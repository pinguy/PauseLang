# ⏸️ PauseLang

**PauseLang v0.7.13** is a tiny experimental virtual machine where **instruction identity is encoded by pause duration**.

The operand stream still carries ordinary integer values; the timing stream says what to *do* with them. A `45 ms` pause means `PUSH`, `100 ms` means `ADD2`, `150 ms` means `HALT`, and so on. In other words: the data is data, but the opcodes are rhythm.

PauseLang is mainly an experiment in **temporal computing, side-band control, timing channels, and small deterministic supervisors**. It is deliberately strange, bounded, and easy to inspect rather than a replacement for Python, C, or a general-purpose VM.

## Current status

v0.7.13 currently includes:

- a stack-based VM with 32-bit wrapping arithmetic;
- stream, stack, hybrid, control-flow, and system instructions;
- labels, aliases, and compiler macros;
- an indexed `IX` register with `LOADI`, `STOREI`, `INCIX`, and `GETIX`;
- direct and conditional jumps, calls/returns, and bounded loops;
- gas, stack, call-depth, loop-depth, memory, and trap limits;
- timing guard bands, drift estimation, and a two-symbol sync phrase;
- chronological execution traces and disassembly with per-step stack snapshots;
- optional WAV export of a timing program;
- IoT-style demos for a leaky bucket, spike detection, and temporal key delivery;
- a TCP sender/receiver demo that reconstructs and executes a program from packet timing.

The supplied v0.7.13 test suite currently runs **28 torture tests** before the demos.

## How it works

A PauseLang program is represented by two parallel streams:

```text
operand/data stream:  [10, 20, 0, ...]
pause/timing stream:  [45ms, 45ms, 100ms, ...]
                         │     │      │
                         │     │      └─ ADD2
                         │     └──────── PUSH 20
                         └────────────── PUSH 10
```

The compiler adds a sync phrase before the program:

```text
290 ms, 300 ms
```

The VM can use that phrase to estimate clock drift before decoding the actual instructions.

The default specification uses a **5 ms timing quantum** and **1.5 ms guard band**. The TCP demo intentionally widens the receiver guard band to `4 ms` to tolerate scheduler and socket jitter.

## Quick start

Clone the repository and run the main file:

```bash
git clone https://github.com/pinguy/PauseLang
cd PauseLang
python3 PauseLang_v0_7_13.py
```

The main entry point runs the torture suite, then the IoT demos if all tests pass. If NumPy and SciPy are available it also exports the key-delivery timing sequence as a WAV file.

The core VM uses only the Python standard library. WAV export is optional:

```bash
python3 -m pip install numpy scipy
```

## Small example

PauseLang source is **line-oriented: one instruction per line**.

```text
main:
    CONST 10
    CONST 20
    ADD2
    HALT
```

`CONST` is an alias for `PUSH`, so the final stack contains `30`.

Compile and execute it from Python:

```python
from PauseLang_v0_7_13 import PauseLangCompiler, PauseLangVM

source = """
main:
    CONST 10
    CONST 20
    ADD2
    HALT
"""

pauses, data, comments, labels = PauseLangCompiler.compile(source)
vm = PauseLangVM()
result = vm.execute(data, pauses, labels=labels)

print(result["final_state"]["stack"])
# [30]
```

### Important syntax detail

Do **not** write several instructions on one source line:

```text
CONST 70 STOREI INCIX
```

The compiler parses the first opcode and its optional operand from each line. Write this instead:

```text
CONST 70
STOREI
INCIX
```

## TCP timing demo

The included sender and receiver demonstrate a real timing path over loopback TCP.

Terminal 1:

```bash
python3 pause_tcp_receiver.py
```

Terminal 2:

```bash
python3 pause_tcp_sender.py
```

The sender compiles a PauseLang program, sends each operand as a 16-bit value, and spaces the packets according to the compiled pause stream. The receiver measures the inter-arrival timing, reconstructs the timing stream, and executes it in the VM.

The current demo stores a message into VM memory through `STOREI`/`INCIX`, leaves `1337` on the stack as a beacon, and halts. On the tested loopback path it reconstructs all 70 program instructions and the receiver recovers the message from memory.

This is an **experimental timing-channel transport**, not encryption: operand values are still transmitted as packet payloads. What timing hides/encodes is the instruction stream.

## Instruction model

PauseLang currently has five instruction categories:

| Category | Examples | Purpose |
|---|---|---|
| Stream | `ADD`, `MEAN`, `DIFF`, `SQUARE`, `PASS` | Operate on the current/previous streamed values |
| Stack | `PUSH`, `POP`, `DUP`, `ROT`, `ADD2`, `DIV2` | Conventional stack manipulation and arithmetic |
| Hybrid | `STORE`, `LOAD`, `STOREI` | Combine stack state with the operand stream or IX register |
| Control | `JUMP`, `JZ`, `JNZ`, `CALL`, `RET`, loops | Change execution flow |
| System | `SET_META`, `NOP`, `HALT`, `INCIX` | VM/lane/register control |

Instruction timing is defined in the `INSTRUCTIONS` table. v0.7.13 keys that table by **integer milliseconds** to avoid IEEE-754 float-key ambiguity while retaining canonical float durations for decoding and display.

## Labels, aliases, and macros

The compiler supports labels:

```text
start:
    CONST 1
    JUMP end
    CONST 999
end:
    HALT
```

Common aliases include:

```text
CONST -> PUSH
DROP  -> POP
PEEK  -> DUP
DROPS -> CLEAR_STACK
JMP   -> JUMP
JOD   -> JUMP_IF_ODD
JZ    -> JUMP_IF_ZERO
JNZ   -> JUMP_IF_NONZERO
```

Built-in macros include `INC`, `DEC`, `DOUBLE`, `SQUARED`, `ENTER`, `LEAVE`, `NOT`, `LNOT`, `NEG`, and `SETF`.

## Memory and the IX register

PauseLang has up to 256 memory slots by default.

Direct memory operations use the instruction operand as the slot:

```text
CONST 99
STORE 42
LOAD 42
```

Indexed memory uses `IX`:

```text
CONST 0
SETIX
CONST 65
STOREI
INCIX
CONST 66
STOREI
```

`LOADI` returns `0` for an uninitialised slot. `INCIX` wraps around the configured memory size.

## Safety and traps

The VM is bounded rather than "secure" in the cryptographic sense. It has explicit limits and traps for failure cases such as:

- stack underflow/overflow;
- division by zero;
- invalid memory access;
- invalid jumps/calls;
- call-depth and loop-depth exhaustion;
- gas exhaustion;
- return without a call;
- unmatched `LOOP_END`;
- trap storms;
- invalid timing/instructions.

`trap_policy` can continue, halt, or raise depending on how the VM is embedded.

## Timing robustness

v0.7.13 specifically hardens timing decode behaviour:

- integer-millisecond instruction keys avoid float dictionary ambiguity;
- guard-band checks compare the raw adjusted pause against canonical targets;
- guard boundaries are converted to integer microseconds for deterministic inclusive comparisons;
- jitter outside a valid guard band is not silently snapped to the nearest opcode;
- sync calibration estimates drift from the sync phrase;
- `strict_sync=True` disables automatic sync stripping when the caller needs exact control.

Timing is still subject to the host OS, scheduler, transport, and clock behaviour. Loopback TCP is a useful demonstration, not a real-time guarantee.

## Tests and demos

Running:

```bash
python3 PauseLang_v0_7_13.py
```

covers the current torture suite, including labels, aliases, jumps, arithmetic semantics, overflow behaviour, jitter, flags, stack protection, loop handling, macros, `RET`, IX operations, sync handling, fuzzing, gas exhaustion, trace PC accuracy, and guard-boundary stability.

If all tests pass, three demos run:

1. **Leaky Bucket Rate Limiter** — a small temporal/supervisory control-flow example.
2. **Temporal Spike / Dragon Detector** — detects selected anomaly values from a small stored stream.
3. **Temporal Key Delivery** — stores and recovers a four-byte key through VM memory.

## WAV export

`WavExporter` can render a compiled timing stream as clicks separated by silent gaps:

```python
from PauseLang_v0_7_13 import PauseLangCompiler, WavExporter

pauses, _, _, _ = PauseLangCompiler.compile("CONST 42\nHALT")
WavExporter.export_to_wav(pauses, filename="pause_program.wav")
```

The WAV representation makes the timing program audible/inspectable and can be decoded by measuring inter-click intervals.

## What PauseLang is good at

PauseLang is a decent fit for experiments involving:

- temporal or side-band supervision;
- compact deterministic state machines;
- watchdog-style control logic;
- timing-channel research;
- rate limiting and threshold checks;
- sensor/event stream experiments;
- low-complexity control programs where bounded execution matters.

## What it is not

PauseLang is not intended as a general-purpose language. Its model deliberately makes some things awkward:

- large or dynamic data structures;
- rich indirect memory access;
- deep recursion;
- complex array algorithms;
- high-throughput computation;
- hard real-time networking on a normal desktop OS.

That constraint is part of the experiment: **what becomes useful when time itself is part of the instruction encoding?**

## Project files

```text
PauseLang_v0_7_13.py   VM, compiler, tests, demos, WAV exporter
pause_tcp_sender.py     timing-channel TCP sender demo
pause_tcp_receiver.py   timing measurement + VM execution demo
README.md               this file
```

## Philosophy

PauseLang is not about raw power. It is about **doing something while doing almost nothing**: supervising, regulating, and signalling through silence.

A language for **time, rhythm, and control**.

Not Python. Not C. Something stranger and smaller — something that *haunts* the main program while it runs.
