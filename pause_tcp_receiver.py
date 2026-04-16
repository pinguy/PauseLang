import socket
import time
import struct
from PauseLang_v0_7_12 import PauseLangVM, SPEC

HOST = "127.0.0.1"
PORT = 65432

# Tolerance for TCP jitter
SPEC['guard_band'] = 0.0040

def recv_exact(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def receive_and_execute():
    vm = PauseLangVM(debug=False, gas_limit=100000)

    data_stream = []
    measured_pauses = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}...")

        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Connection from {addr}")

        # Read length prefix
        length_raw = recv_exact(conn, 4)
        if not length_raw:
            print("Failed to read length")
            return
        num_operands = struct.unpack('<I', length_raw)[0]
        print(f"Expecting {num_operands} operands")

        prev_time = time.perf_counter()
        first = True

        for _ in range(num_operands):
            raw = recv_exact(conn, 2)
            if raw is None:
                print("Connection closed prematurely")
                break

            operand = struct.unpack('<H', raw)[0]
            current_time = time.perf_counter()

            if not first:
                delay = current_time - prev_time
                measured_pauses.append(delay)
            else:
                first = False

            data_stream.append(operand)
            prev_time = current_time

    print(f"\nReceived {len(data_stream)} operands, {len(measured_pauses)} pauses.")

    # Pad if needed
    while len(measured_pauses) < len(data_stream):
        measured_pauses.append(0.150)

    result = vm.execute(
        data_stream=data_stream,
        pause_stream=measured_pauses,
        sync=True,
        strict_sync=False
    )

    print("\n=== PauseLang VM Execution Result ===")
    print(f"Halted: {result['halted']}")
    print(f"Gas used: {result['gas_used']}")
    print(f"Traps: {result.get('traps', [])}")
    print(f"Final stack: {result['final_state']['stack']}")
    print(f"Memory: {result['final_state']['memory']}")
    print(f"IX: {result['final_state']['ix']}")

    # ==================== CLEAN HUMAN-READABLE OUTPUT ====================
    print("\n" + "="*50)
    print("SECRET MESSAGE RECEIVED VIA TIMING CHANNEL")
    print("="*50)

    stack = result['final_state']['stack']
    message = ""

    # Extract printable characters from stack
    for byte in stack:
        if 32 <= byte <= 126:          # printable ASCII
            message += chr(byte)
        elif byte == 1337:             # beacon / end marker
            break

    if message:
        print(f"Hidden message: {message}")
    else:
        print("No readable message found.")

    print(f"Beacon value : {stack[-1] if stack else 'None'}")
    print(f"Total instructions executed: {result['gas_used']}")
    print("="*50)

if __name__ == "__main__":
    receive_and_execute()
