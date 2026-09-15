import io
import socket
import struct
import unittest
from unittest.mock import patch

from PauseLang_v0_7_13 import PauseLangCompiler, PauseLangVM, SPEC
from pause_tcp_protocol import (END_MARKER, HEADER, MAGIC, MAX_OPERANDS, WORD,
                                receive_program, recv_exact, send_program, program_checksum, validate_program)


class FragmentedSocket:
    def __init__(self, raw):
        self.stream = io.BytesIO(raw)

    def recv(self, count):
        return self.stream.read(min(1, count))


class TransportTests(unittest.TestCase):
    def test_fragmented_signed_words_and_final_nonhalt(self):
        pauses, data, _, _ = PauseLangCompiler.compile('CONST -2147483648\nCONST 2147483647\nCONST 70000\nADD2')
        payload = bytearray()
        class Sender:
            def sendall(self, chunk):
                payload.extend(chunk)
        sleeps = []
        send_program(Sender(), data, pauses, sleep=sleeps.append)
        timestamps = [0.0]
        for pause in sleeps:
            timestamps.append(timestamps[-1] + pause)
        measured_data, measured, checksum = receive_program(FragmentedSocket(payload), clock=iter(timestamps).__next__)
        self.assertEqual(measured_data, data)
        validate_program(measured_data, measured, checksum, PauseLangVM().quantizer)
        for actual, expected in zip(measured, pauses):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(measured[-1], .100)  # never fabricated HALT
        result = PauseLangVM().execute(measured_data, measured)
        self.assertEqual(result['traps'], [])
        self.assertEqual(result['final_state']['stack'], [-2147483648, -2147413649])

    def test_real_socket_framing(self):
        sender, receiver = socket.socketpair()
        with sender, receiver:
            sender.settimeout(1)
            receiver.settimeout(1)
            send_program(sender, [-1, 70000], [.045, .150], sleep=lambda _: None)
            data, pauses, checksum = receive_program(receiver, clock=iter([0, .045, .195]).__next__)
        self.assertEqual(data, [-1, 70000])
        self.assertAlmostEqual(pauses[-1], .150)

    def test_every_truncation_rejected(self):
        frame = HEADER.pack(MAGIC, 2, 0) + WORD.pack(42) + WORD.pack(-1) + END_MARKER
        for cutoff in range(len(frame)):
            with self.subTest(cutoff=cutoff), self.assertRaises(EOFError):
                receive_program(FragmentedSocket(frame[:cutoff]), clock=lambda: 0)

    def test_bad_headers_and_marker(self):
        for header in [HEADER.pack(b'nope', 1, 0), HEADER.pack(MAGIC, 0, 0), HEADER.pack(MAGIC, MAX_OPERANDS+1, 0)]:
            with self.assertRaises(ValueError):
                receive_program(FragmentedSocket(header))
        with self.assertRaisesRegex(ValueError, 'terminal'):
            receive_program(FragmentedSocket(HEADER.pack(MAGIC, 1, 0)+WORD.pack(42)+WORD.pack(1)))

    def test_sender_rejects_before_writing(self):
        class NoWrites:
            def sendall(self, _):
                raise AssertionError('Invalid inputs must not be sent')
        for data, pauses, jitter in [([2**31], [.045], 0), ([-2**31-1], [.045], 0),
                                    ([1], [], 0), ([], [], 0), ([1], [float('nan')], 0),
                                    ([1], [.045], -.01), ([1], [.045], .05)]:
            with self.assertRaises((ValueError, struct.error)):
                send_program(NoWrites(), data, pauses, jitter)

    def test_timeout_propagates(self):
        class TimedOut:
            def recv(self, _):
                raise socket.timeout('timed out')
        with self.assertRaises(socket.timeout):
            recv_exact(TimedOut(), 4)

    def test_receiver_import_does_not_mutate_spec(self):
        before = SPEC.copy()
        import importlib
        import pause_tcp_receiver
        importlib.reload(pause_tcp_receiver)
        self.assertEqual(SPEC, before)

    def test_checksum_rejects_wrong_but_valid_opcode(self):
        pauses, data, _, _ = PauseLangCompiler.compile('CONST 42\nHALT')
        checksum = program_checksum(data, pauses)
        pauses[2] = .050  # PUSH corrupted into a perfectly valid POP
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            validate_program(data, pauses, checksum, PauseLangVM().quantizer)

    def test_checksum_rejects_corrupted_operand(self):
        pauses, data, _, _ = PauseLangCompiler.compile('CONST 42\nHALT')
        checksum = program_checksum(data, pauses)
        data[2] = 99
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            validate_program(data, pauses, checksum, PauseLangVM().quantizer)

    def test_invalid_timing_and_sync_rejected(self):
        pauses, data, _, _ = PauseLangCompiler.compile('CONST 42\nHALT')
        for index, value in [(0, .01), (2, .0475)]:
            altered = pauses.copy()
            altered[index] = value
            with self.assertRaises(ValueError):
                validate_program(data, altered, program_checksum(data, pauses), PauseLangVM().quantizer)

    def test_fourfold_timing_preserves_instruction_identity(self):
        pauses, data, _, _ = PauseLangCompiler.compile('CONST -42\nHALT')
        payload = bytearray()
        class Sender:
            def sendall(self, chunk):
                payload.extend(chunk)
        sleeps = []
        send_program(Sender(), data, pauses, sleep=sleeps.append, time_scale=4)
        self.assertEqual(sleeps, [p*4 for p in pauses])
        timestamps = [0.0]
        for pause in sleeps:
            timestamps.append(timestamps[-1]+pause)
        actual, measured, checksum = receive_program(FragmentedSocket(payload), time_scale=4,
                                                     clock=iter(timestamps).__next__)
        validate_program(actual, measured, checksum, PauseLangVM().quantizer)
        result = PauseLangVM().execute(actual, measured)
        self.assertEqual(result['final_state']['stack'], [-42])
        self.assertEqual(result['traps'], ['HALT'])
