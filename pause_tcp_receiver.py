import socket
from PauseLang_v0_7_13 import PauseLangVM
from pause_tcp_protocol import IO_TIMEOUT, TCP_GUARD_BAND, TCP_TIME_SCALE, receive_program, validate_program

HOST = "127.0.0.1"
PORT = 65432


def receive_and_execute():
    vm = PauseLangVM(debug=False, gas_limit=100000, trap_policy='halt',
                     guard_band=TCP_GUARD_BAND)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}...", flush=True)
        conn, addr = s.accept()
        with conn:
            conn.settimeout(IO_TIMEOUT)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Connection from {addr}")
            # EOF, invalid framing and timeouts abort before any VM execution.
            data_stream, measured_pauses, checksum = receive_program(conn, time_scale=TCP_TIME_SCALE)

    print(f"Received {len(data_stream)} operands and {len(measured_pauses)} measured pauses.")
    validate_program(data_stream, measured_pauses, checksum, vm.quantizer)
    result = vm.execute(data_stream, measured_pauses)
    if 'error' in result:
        raise ValueError(result['error'])
    print("\n=== PauseLang VM Execution Result ===")
    print(f"Halted: {result['halted']}")
    print(f"Gas used: {result['gas_used']}")
    print(f"Traps: {result.get('traps', [])}")
    print(f"Final stack: {result['final_state']['stack']}")
    print(f"Memory: {result['final_state']['memory']}")
    print(f"IX: {result['final_state']['ix']}")

    if result['traps'] != ['HALT']:
        raise RuntimeError(f"Program did not halt cleanly: {result['traps']}")

    # Human-readable demo payload (not encrypted).
    print("\n" + "="*50)
    print("MESSAGE RECEIVED VIA TIMING CHANNEL")
    print("="*50)

    stack = result['final_state']['stack']
    memory = result['final_state']['memory']
    message = ""

    # Preferred format: contiguous printable ASCII bytes in memory starting at 0.
    slot = 0
    while slot in memory and 32 <= memory[slot] <= 126:
        message += chr(memory[slot])
        slot += 1

    # Alternate program convention: printable bytes left on the stack.
    if not message:
        for byte in stack:
            if 32 <= byte <= 126:
                message += chr(byte)
            elif byte == 1337:         # beacon / end marker
                break

    if message:
        print(f"Message: {message}")
    else:
        print("No readable message found.")

    print(f"Beacon value : {stack[-1] if stack else 'None'}")
    print(f"Total instructions executed: {result['gas_used']}")
    print("="*50)
    return result

if __name__ == "__main__":
    receive_and_execute()
