import socket
import time
import random
import struct
from PauseLang_v0_7_12 import PauseLangCompiler

HOST = "127.0.0.1"
PORT = 65432

JITTER = 0.0008

# Fixed program: stores bytes into MEMORY instead of just pushing to stack
HIDDEN_PROGRAM = """
main:
    CONST 0
    SETIX                    # IX = 0

    CONST 70   STOREI INCIX   # F
    CONST 117  STOREI INCIX   # u
    CONST 99   STOREI INCIX   # c
    CONST 107  STOREI INCIX   # k
    CONST 32   STOREI INCIX   # space
    CONST 101  STOREI INCIX   # e
    CONST 109  STOREI INCIX   # m
    CONST 32   STOREI INCIX   # space
    CONST 97   STOREI INCIX   # a
    CONST 110  STOREI INCIX   # n
    CONST 100  STOREI INCIX   # d
    CONST 32   STOREI INCIX   # space
    CONST 116  STOREI INCIX   # t
    CONST 104  STOREI INCIX   # h
    CONST 101  STOREI INCIX   # e
    CONST 105  STOREI INCIX   # i
    CONST 114  STOREI INCIX   # r
    CONST 32   STOREI INCIX   # space
    CONST 108  STOREI INCIX   # l
    CONST 97   STOREI INCIX   # a
    CONST 119  STOREI INCIX   # w
    CONST 33   STOREI INCIX   # !

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
