import socket
import time
import random
import struct
from PauseLang_v0_7_13 import PauseLangCompiler

HOST = "127.0.0.1"
PORT = 65432

JITTER = 0.0008

# Fixed program: stores bytes into MEMORY instead of just pushing to stack
HIDDEN_PROGRAM = """
main:
    CONST 0
    SETIX                    # IX = 0

    CONST 70                    # F
    STOREI
    INCIX
    CONST 117                   # u
    STOREI
    INCIX
    CONST 99                    # c
    STOREI
    INCIX
    CONST 107                   # k
    STOREI
    INCIX
    CONST 32                    # space
    STOREI
    INCIX
    CONST 101                   # e
    STOREI
    INCIX
    CONST 109                   # m
    STOREI
    INCIX
    CONST 32                    # space
    STOREI
    INCIX
    CONST 97                    # a
    STOREI
    INCIX
    CONST 110                   # n
    STOREI
    INCIX
    CONST 100                   # d
    STOREI
    INCIX
    CONST 32                    # space
    STOREI
    INCIX
    CONST 116                   # t
    STOREI
    INCIX
    CONST 104                   # h
    STOREI
    INCIX
    CONST 101                   # e
    STOREI
    INCIX
    CONST 105                   # i
    STOREI
    INCIX
    CONST 114                   # r
    STOREI
    INCIX
    CONST 32                    # space
    STOREI
    INCIX
    CONST 108                   # l
    STOREI
    INCIX
    CONST 97                    # a
    STOREI
    INCIX
    CONST 119                   # w
    STOREI
    INCIX
    CONST 33                    # !
    STOREI
    INCIX

    CONST 1337
    HALT
"""

def send_with_timing(host, port, hidden_source: str):
    pauses, data, comments, labels = PauseLangCompiler.compile(hidden_source)
    print(f"Compiled {len(pauses)-2} instructions (+ sync phrase)")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((host, port))
            print(f"Connected to {host}:{port}")

            length = len(data)
            s.sendall(struct.pack('<I', length))

            for i, operand in enumerate(data):
                payload = struct.pack('<H', operand & 0xFFFF)
                s.sendall(payload)

                if i < len(pauses):
                    target = pauses[i]
                    actual = target + random.uniform(-JITTER, JITTER)
                    time.sleep(max(0.0, actual))
                else:
                    time.sleep(0.012 + random.uniform(-0.001, 0.001))

            print("✅ Hidden PauseLang program sent successfully.")

    except Exception as e:
        print(f"❌ Sender error: {e}")


if __name__ == "__main__":
    send_with_timing(HOST, PORT, HIDDEN_PROGRAM)
