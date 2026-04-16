import socket
import time
import struct
from PauseLang_v0_7_12 import PauseLangVM, SPEC

HOST = "127.0.0.1"
PORT = 65432

# Increase guard band temporarily to be more tolerant of timing jitter
SPEC['guard_band'] = 0.003   # 3 ms instead of 1.5 ms

def recv_exact(conn, n):
    """Receive exactly n bytes from socket."""
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def receive_and_execute():
    vm = PauseLangVM(debug=True, gas_limit=50000)   # debug=True shows opcode decoding

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

        prev_time = time.time()
        first_packet = True

        while True:
            raw = recv_exact(conn, 2)
            if raw is None:
                break

            operand = struct.unpack('<H', raw)[0]
            current_time = time.time()

            if not first_packet:
                delay = current_time - prev_time
                measured_pauses.append(delay)
                print(f"Received operand {operand:5d} | measured pause = {delay:.4f}s")
            else:
                first_packet = False
                print(f"Received operand {operand:5d} (first packet)")

            data_stream.append(operand)
            prev_time = current_time

            if len(data_stream) > 1000:
                break

        # Pad pause stream to match data_stream length
        if len(measured_pauses) < len(data_stream):
            measured_pauses.append(0.150)   # dummy HALT pause
            print(f"Appended dummy HALT pause 0.150s")

    print(f"\nReceived {len(data_stream)} operands, {len(measured_pauses)} pauses.")
    print("Data stream:", data_stream)
    print("Pause stream (raw):", [round(p, 4) for p in measured_pauses])

    result = vm.execute(
        data_stream=data_stream,
        pause_stream=measured_pauses,
        sync=True,
        strict_sync=False
    )

    print("\n=== VM Execution Result ===")
    print(f"Halted: {result['halted']}")
    print(f"Gas used: {result['gas_used']}")
    print(f"Traps: {result['traps']}")
    print(f"Final stack: {result['final_state']['stack']}")
    print(f"Memory: {result['final_state']['memory']}")
    print(f"IX register: {result['final_state']['ix']}")

    if result['final_state']['stack']:
        print(f"\nSecret value from stack: {result['final_state']['stack']}")

if __name__ == "__main__":
    receive_and_execute()
