import socket
import time
import random
import struct
from PauseLang_v0_7_12 import PauseLangCompiler, SPEC

HOST = "127.0.0.1"
PORT = 65432
JITTER = 0.0005          # smaller jitter for more precise opcode timing

HIDDEN_PROGRAM = """
main:
    CONST 1234
    PUSH 42
    ADD2
    STORE 5
    CONST 7
    STORE 10
    HALT
"""

def send_with_timing(host, port, hidden_source: str):
    pauses, data, comments, labels = PauseLangCompiler.compile(hidden_source)
    print(f"Compiled {len(pauses)-2} hidden instructions (sync phrase included)")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((host, port))
            print(f"Connected to {host}:{port}")

            for i, operand in enumerate(data):
                payload = struct.pack('<H', operand & 0xFFFF)
                s.sendall(payload)

                if i < len(pauses):
                    target_pause = pauses[i]
                    actual_delay = target_pause + random.uniform(-JITTER, JITTER)
                    time.sleep(max(0.0, actual_delay))
                    print(f"Sent operand {operand:5d} (0x{operand:04x}) | pause={actual_delay:.4f}s → {comments[i]}")
                else:
                    time.sleep(0.015)

            print("✅ Hidden program sent.")

    except Exception as e:
        print(f"❌ Sender error: {e}")

if __name__ == "__main__":
    send_with_timing(HOST, PORT, HIDDEN_PROGRAM)
