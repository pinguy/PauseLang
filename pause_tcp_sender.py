import socket
import time
import random
from PauseLang_v0_7_12 import PauseLangCompiler, SPEC

# ====================== SETTINGS ======================
HOST = "127.0.0.1"
PORT = 65432

BASE_DELAY = 0.015          # Natural inter-packet delay for cover traffic
JITTER = 0.0015             # Random noise for stealth (should stay inside guard band)

# Example hidden PauseLang program (you can change this)
HIDDEN_PROGRAM = """
main:
    CONST 1234          # Some secret value
    PUSH 42
    ADD2
    STORE 5             # Store result in memory slot 5
    CONST 7
    STORE 10
    HALT
"""

def send_with_timing(host, port, cover_message: str, hidden_source: str):
    # Compile hidden PauseLang program
    pauses, data, comments, labels = PauseLangCompiler.compile(hidden_source)
    print(f"Compiled {len(pauses)-2} hidden instructions (sync phrase included)")

    # Pad cover message so we have enough packets
    min_packets = len(pauses)
    cover_message = cover_message + " " * max(0, min_packets - len(cover_message))

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print(f"Connected to {host}:{port}")

            for i, char in enumerate(cover_message):
                # Send normal cover character
                s.sendall(char.encode('utf-8'))

                # Insert precise pause (timing = opcode)
                if i < len(pauses):
                    target_pause = pauses[i]
                    # Add small jitter for deniability (keep it inside guard band)
                    actual_delay = target_pause + random.uniform(-JITTER, JITTER)
                    time.sleep(max(0.0, actual_delay))
                    print(f"Sent '{char}' | pause={actual_delay:.4f}s → {comments[i]}")
                else:
                    # Normal traffic after hidden program ends
                    time.sleep(BASE_DELAY + random.uniform(-JITTER*2, JITTER*2))

            print("✅ Cover message + hidden PauseLang program sent.")
            print(f"Hidden program size: {len(pauses)-2} instructions")

    except Exception as e:
        print(f"❌ Sender error: {e}")


if __name__ == "__main__":
    cover = "This is a completely normal TCP message. Nothing to see here."
    send_with_timing(HOST, PORT, cover, HIDDEN_PROGRAM)
