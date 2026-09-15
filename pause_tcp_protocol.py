"""Versioned demo framing; operands are payload, opcodes are measured gaps.

TCP is a byte stream, not a packet clock. Fragmentation is handled for framing,
while coalescing/scheduling can still destroy timing information.
"""
import math
import random
import struct
import time
import zlib

MAGIC = b'PLT1'
HEADER = struct.Struct('!4sII')  # magic, operand count, intended-stream CRC32
WORD = struct.Struct('!i')
END_MARKER = WORD.pack(0)
MAX_OPERANDS = 10000
IO_TIMEOUT = 5.0
TCP_TIME_SCALE = 4.0  # 20 ms wire spacing for neighbouring opcodes
TCP_GUARD_BAND = 0.0015  # 6 ms physical tolerance at 4x timing


def program_checksum(data, pauses):
    """Unkeyed integrity check over operands and canonical pause microseconds.

    The digest is not an instruction list and is not authentication/encryption.
    """
    if len(data) != len(pauses):
        raise ValueError('Stream length mismatch')
    checksum = 0
    for value, pause in zip(data, pauses):
        checksum = zlib.crc32(struct.pack('!iI', value, round(pause*1_000_000)), checksum)
    return checksum


def validate_program(data, pauses, expected_checksum, quantizer):
    """Decode and verify the whole frame before any VM side effects."""
    from PauseLang_v0_7_13 import SPEC
    sync_count = len(SPEC['sync_phrase'])
    if not quantizer.calibrate(pauses[:sync_count]):
        raise ValueError('Missing or invalid sync phrase')
    canonical = list(SPEC['sync_phrase'])
    for index, pause in enumerate(pauses[sync_count:], start=sync_count):
        instruction = quantizer.decode(pause)
        if instruction is None:
            raise ValueError(f'Invalid or ambiguous timing at position {index}: {pause:.6f}s')
        canonical.append(instruction.pause)
    if program_checksum(data, canonical) != expected_checksum:
        raise ValueError('Timing stream checksum mismatch; refusing to execute corrupted program')


def recv_exact(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise EOFError(f'Truncated timing frame: expected {n}, received {len(data)} bytes')
        data.extend(chunk)
    return bytes(data)


def send_program(conn, data, pauses, jitter=0.0, *, sleep=time.sleep, rng=None, time_scale=1.0):
    if len(data) != len(pauses) or not 1 <= len(data) <= MAX_OPERANDS:
        raise ValueError('Invalid stream lengths')
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError('time_scale must be finite and positive')
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError('jitter must be finite and non-negative')
    if any(not math.isfinite(p) or not jitter < p*time_scale < IO_TIMEOUT - jitter for p in pauses):
        raise ValueError('Pauses must remain positive and below the I/O timeout')
    # Validate the entire payload before sending any bytes; never truncate words.
    payloads = [WORD.pack(value) for value in data]
    rng = rng or random.Random()
    conn.sendall(HEADER.pack(MAGIC, len(data), program_checksum(data, pauses)))
    for payload, pause in zip(payloads, pauses):
        conn.sendall(payload)
        sleep(pause*time_scale + rng.uniform(-jitter, jitter))
    # A terminal marker lets the receiver measure the LAST opcode's pause.
    conn.sendall(END_MARKER)


def receive_program(conn, *, clock=time.perf_counter, time_scale=1.0):
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError('time_scale must be finite and positive')
    magic, count, checksum = HEADER.unpack(recv_exact(conn, HEADER.size))
    if magic != MAGIC:
        raise ValueError('Unsupported timing protocol (use matching sender and receiver)')
    if not 1 <= count <= MAX_OPERANDS:
        raise ValueError(f'Operand count must be between 1 and {MAX_OPERANDS}')
    current = recv_exact(conn, WORD.size)
    previous_time = clock()
    data, pauses = [], []
    for index in range(count):
        following = recv_exact(conn, WORD.size)
        now = clock()
        data.append(WORD.unpack(current)[0])
        pauses.append((now - previous_time) / time_scale)
        current, previous_time = following, now
    if current != END_MARKER:
        raise ValueError('Invalid terminal timing marker')
    return data, pauses, checksum
