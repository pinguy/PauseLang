# ⏸️ PauseLang

**PauseLang** is a tiny virtual machine and language that encodes instructions in **pause durations**.  
Instead of clock ticks and numeric opcodes, PauseLang runs on **time itself**.  

It’s designed for **temporal + neuromorphic computing patterns** — where supervision, rhythm, and timing matter more than raw algorithmic crunching.

---

## ✨ Why PauseLang?

- 🥁 **Unique Paradigm — Time as Code**  
  Instructions are encoded as *pauses*. You don’t just execute opcodes; you play rhythms.  
  Perfect for neuromorphic chips (Intel Loihi, SpiNNaker), analog spikes, or any hardware where *time is a first-class citizen*.

- 🛡️ **Robust to Noise**  
  Guard bands, quantization, and drift compensation mean PauseLang can tolerate jittery signals and imperfect clocks.  
  It’s been torture-tested under noise and still decodes cleanly.

- 🔒 **Minimal, Secure VM**  
  - Stack-based execution  
  - Gas limits to prevent infinite loops  
  - Flags + traps for safe error handling  
  - Concise ISA covering arithmetic, bitwise ops, control flow, and I/O  
  Small enough to audit, safe enough to embed.

- 🐉 **Temporal Supervision**  
  PauseLang is ideal as a **side-band supervisor**, running alongside a main program.  
  Example: the *Dragon Detector Demo*, where it evaluates probabilistic decisions in parallel at near-zero overhead.  
  Think watchdog timers, ML edge inference, sensor fusion, or power-aware regulation.

- 🧩 **Extensible**  
  - Macro system in the compiler for higher-level patterns (Fibonacci, leaky buckets, sliding windows)  
  - Clean I/O via port-mapped `MemoryController`  
  - Debugger with single-step and trace mode for *time-aware inspection*

---

## 🧰 What It Can Do

✔ **Fixed control-flow templates** — supervision loops, event ticks, deterministic sequences  
✔ **Temporal supervision** — rate-limiting, leaky bucket regulators, spike encodings  
✔ **Bitwise operations** — feature gates, masks, pattern detectors  
✔ **Simple pipelines** — sums, comparisons, threshold checks  
✔ **Side-band watchdogs** — run *alongside* real programs, enforcing safety or constraints cheaply  

⚠️ **What It’s *not* for**  
PauseLang is not a general-purpose algorithmic language.  
It struggles with:  
- Indirect/dynamic memory access  
- Complex recursive patterns  
- Array-based algorithms  
- Data-dependent control flow  

---

## 🚀 Demos

- **Dragon Detector** — probabilistic supervision in real time  
- **Fibonacci with IX Register** — indexed memory access pattern (conceptual)  
- **Leaky Bucket** — rate limiter for event streams  
- **Spike Encoding Bridge** — maps binary patterns into temporal spikes  
- **MemoryController** — clean I/O separation between VM and world  

---

## 🔬 Stress Tested

- ✅ Runs 100s–1000s of iterations without leaks or slowdowns  
- ✅ Robust under timing jitter and clock drift  
- ✅ Safe under invalid ops (traps instead of crashes)  
- ✅ Gas and stack safety invariants hold  

---

## 📜 Philosophy

PauseLang isn’t about raw power.  
It’s about **doing something while doing almost nothing** —  
supervising, regulating, and signaling through silence.  

A language for **time, rhythm, and control**.  
Not Python. Not C. Something stranger. Something smaller.  
Something that *haunts* your main program while it runs.

---

## 🚀 Quick Start

Absolute addressing and loop-stack leak fix in v6.4

Clone the repo:

```bash
git clone https://github.com/pinguy/PauseLang
cd PauseLang
python3 PauseLang_v0.6.4.py

