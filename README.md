# ⏸️ PauseLang

**PauseLang v0.7.14** is a tiny experimental virtual machine where **instruction identity is encoded by pause duration**.

The operand stream still carries ordinary integer values; the timing stream says what to *do* with them. A `45 ms` pause means `PUSH`, `100 ms` means `ADD2`, `150 ms` means `HALT`, and so on. In other words: the data is data, but the opcodes are rhythm.

PauseLang is mainly an experiment in **temporal computing, side-band control, timing channels, and small deterministic supervisors**. It is deliberately strange, bounded, and easy to inspect rather than a replacement for Python, C, or a general-purpose VM.

## Current status

v0.7.14 currently includes:

- a stack-based VM with 32-bit wrapping arithmetic;
- stream, stack, hybrid, control-flow, and system instructions;
- labels, aliases, and compiler macros;
- an indexed `IX` register with `LOADI`, `STOREI`, `INCIX`, and `GETIX`;
- direct and conditional jumps, calls/returns, and bounded loops;
- gas, stack, call-depth, loop-depth, memory, and trap limits;
- unique-match timing guard bands, offset calibration, opt-in clock-skew calibration, and a two-symbol sync phrase;
- chronological execution traces and disassembly with per-step stack snapshots;
- optional WAV export of a timing program;
- IoT-style demos for a leaky bucket, spike detection, and temporal key delivery;
- a TCP sender/receiver demo that reconstructs and executes a program from packet timing.

The test suite includes the original **28 torture tests**, independent regression tests, transport framing/integrity tests, and optional WAV tests. Run `python3 -m unittest discover -v`; see [CHANGELOG.md](CHANGELOG.md) for the fixes. The legacy module filename is retained to keep existing imports working.

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

The VM uses that phrase to estimate a constant timing offset before decoding instructions. Optional affine calibration also estimates proportional clock skew; see the timing section below.

The ISA uses a **5 ms timing quantum** and **1.5 ms guard band**. TCP sends pauses at **4× duration**, giving adjacent symbols **20 ms** of wire spacing and a **6 ms physical guard band**. The receiver normalises measurements back to ISA time. Guard windows that match multiple opcodes are rejected.

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

The compiler raises a `ValueError` with the source line number for extra tokens. Write this instead:

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

The sender compiles a PauseLang program and sends signed **32-bit operands**, with pauses after each operand. A final marker lets the receiver measure the final pause too; it never invents a `HALT`. The demo takes roughly 46 seconds at its default 4× timing scale.

The versioned `PLT1` header carries a count and an unkeyed CRC32 of the intended operand/timing stream. The receiver measures gaps between complete operand frames, decodes them, and checks the CRC **before executing any instructions**. Invalid timing, wrong-but-valid opcode substitution, operand corruption, truncation, a bad marker, and socket timeout abort the frame. CRC32 detects accidental corruption; it is not authentication, encryption, or protection against deliberately constructed collisions. Both scripts must be updated together; this framing is incompatible with the old unversioned demo.

TCP is a byte stream: frame boundaries are not network packet boundaries. Coalescing and scheduling can destroy timing even with `TCP_NODELAY`. The receiver limits frames to 10,000 operands, uses a 5-second socket I/O timeout, and halts on VM errors. These are research-demo bounds, not a hardened network service.

The current demo stores a message into VM memory through `STOREI`/`INCIX`, leaves `1337` on the stack as a beacon, and halts. The loopback integration check expects 70 executed instructions, only the normal `HALT` event, the exact message in memory, and beacon `1337`. A noisy run may be rejected; a successful send alone does not prove successful execution.

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

`STOREI_POP` is a compatibility alias macro for `STOREI`, which already consumes one stack value. To discard another value, write a separate `POP`. `NOT` computes `-1 - x`; `NEG` computes `0 - x`; `LNOT` computes `1 - x` and is only logical negation for boolean inputs `0` and `1`.

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

The decoder compares calibrated pauses directly against canonical targets, with inclusive integer-microsecond guard boundaries. No match or multiple matches means `INVALID_INSTRUCTION`; it never chooses the first instruction in an overlapping window. Non-finite and non-positive pauses are rejected.

Each VM accepts explicit configuration:

```python
vm = PauseLangVM(guard_band=0.0015, calibration="offset")
# Optional experiment for a clock-skewed, low-jitter channel:
vm = PauseLangVM(guard_band=0.0015, calibration="affine")
```

Defaults are read from `SPEC` when a quantizer is created. The TCP receiver passes its configuration explicitly and does not mutate global settings. `vm.reset()` clears calibration as well as VM state; reset before executing a separate program on a reused VM.

- `offset` fits `observed = expected + offset`.
- `affine` fits `observed = scale * expected + offset` and corrects both components.
- Sync acquisition allows up to ±20 ms offset and, in affine mode, scale 0.9–1.1. Residuals must fit the guard band. Invalid calibration leaves the existing fit unchanged.
- `sync=False` disables calibration, but retains the legacy auto-strip of an unadjusted sync phrase. `strict_sync=True` disables both auto-stripping and automatic calibration.

**Affine correction is deliberately opt-in.** The existing sync targets are only 10 ms apart. A 1 ms difference in sync errors can imply a 10% scale error, badly distorting shorter opcodes. The reproducible [timing measurements](benchmarks/RESULTS.md) compare offset/affine correction, guard widths, and a first-match decoder across 3,000 synthetic traces, with full [opcode confusion matrices](benchmarks/timing_results.json). They measure wrong accepted instructions separately from rejected instructions.

Noise can land entirely inside a different opcode's window. A timing guard cannot detect that by itself; the TCP demo also verifies a checksum. Synthetic results and loopback demonstrations are not guarantees about real networks.

## Tests and demos

Running:

```bash
python3 PauseLang_v0_7_13.py
```

runs the discoverable test suite before the demos, including labels, aliases, jumps, arithmetic semantics, overflow behaviour, jitter, flags, stack protection, loop handling, macros, `RET`, IX operations, sync handling, fuzzing, gas exhaustion, trace PC accuracy, and guard-boundary stability.

If all tests pass, three demos run:

1. **Leaky Bucket Rate Limiter** — a small temporal/supervisory control-flow example.
2. **Temporal Spike / Dragon Detector** — detects selected anomaly values from a small stored stream.
3. **Temporal Key Delivery** — stores and recovers a four-byte key through VM memory.

Run tests without demos or WAV output:

```bash
python3 -m unittest discover -v
```

GitHub Actions runs the tests and demos on Linux and Windows with Python 3.10 and 3.13. A separate job installs NumPy/SciPy and checks WAV output. Timing tests use seeded synthetic data and controlled clocks; host scheduler jitter is not made into a flaky CI pass/fail test.

Regenerate the timing report:

```bash
python3 benchmarks/timing_benchmark.py --output benchmarks/timing_results.json --markdown benchmarks/RESULTS.md
```

## WAV export

`WavExporter` can render a compiled timing stream as clicks separated by silent gaps:

```python
from PauseLang_v0_7_13 import PauseLangCompiler, WavExporter

pauses, _, _, _ = PauseLangCompiler.compile("CONST 42\nHALT")
WavExporter.export_to_wav(pauses, filename="pause_program.wav")
```

The WAV representation includes a terminal click so **every** pause, including the last one, has a measurable inter-click interval. Empty streams produce one reference click. Export rejects pauses shorter than 1 ms and sample rates below 4 kHz.

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
PauseLang_v0_7_13.py    VM, compiler, demos, WAV exporter (legacy filename)
pause_tcp_sender.py    timing-channel TCP sender demo
pause_tcp_receiver.py  timing measurement + checked execution demo
pause_tcp_protocol.py  framing, signed operands, checksum, timing scale
tests/                 original torture suite and focused regressions
benchmarks/            seeded channel benchmark, results, confusion matrices
.github/workflows/     automated tests
CHANGELOG.md           fixes and compatibility notes
README.md              this file
```

## Philosophy

PauseLang is not about raw power. It is about **doing something while doing almost nothing**: supervising, regulating, and signalling through silence.

A language for **time, rhythm, and control**.

Not Python. Not C. Something stranger and smaller — something that *haunts* the main program while it runs.
