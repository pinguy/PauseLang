import socket
import time
from PauseLang_v0_7_12 import PauseLangVM, SPEC

HOST = "127.0.0.1"
PORT = 65432
TOLERANCE_EXTRA = 0.002   # Extra safety margin on top of VM's guard_band

def receive_and_execute():
    vm = PauseLangVM(debug=False, gas_limit=50000)

    measured_pauses = []
    cover_text = ""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}...")

        conn, addr = s.accept()
        print(f"Connection from {addr}")

        prev_time = time.time()

        while True:
            data = conn.recv(1)
            if not data:
                break

            char = data.decode('utf-8', errors='ignore')
            cover_text += char

            current_time = time.time()
            delay = current_time - prev_time
            prev_time = current_time

            measured_pauses.append(delay)

            if len(measured_pauses) > 1000:  # safety
                break

        print(f"Received cover message: {cover_text.strip()}")

        # Feed measured pauses into PauseLang VM
        print(f"\nFeeding {len(measured_pauses)} measured pauses into VM...")
        result = vm.execute(
            data_stream=[0] * len(measured_pauses),   # dummy data (we only care about timing)
            pause_stream=measured_pauses,
            sync=True,           # let it calibrate on the sync phrase
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
