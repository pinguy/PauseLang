# Maintenance validation

Local environment: Linux, Python 3.12.14. NumPy and SciPy were available.

- `python3 -m unittest discover -v`: 65 tests passed, including the original 28 torture tests and three WAV tests.
- `python3 PauseLang_v0_7_13.py`: tests passed; all three IoT demos passed; WAV exported successfully.
- `python3 benchmarks/timing_benchmark.py --output benchmarks/timing_results.json --markdown benchmarks/RESULTS.md`: 3,000 seeded synthetic traces, 96,000 instruction symbols per decoder strategy. Four strategies receive identical traces. See the report for wrong-opcode and rejection rates.
- Live TCP sender/receiver integration, using the default 4× timing scale and ±0.8 ms sender jitter, passed with these receiver observations:

```text
Received 72 operands and 72 measured pauses.
Halted: True
Gas used: 70
Traps: ['HALT']
Final stack: [1337]
IX: 22
Message: Fuck em and their law!
Beacon value : 1337
Total instructions executed: 70
```

An earlier 1× live run during development decoded scheduler-delayed pauses into other valid opcodes. This motivated both the wider wire spacing and the pre-execution checksum, rather than relying only on timing-window rejection. Regression tests explicitly corrupt a `PUSH` timing into a valid `POP` timing and verify checksum rejection, as well as testing corrupt operands, every byte truncation of a sample frame, fragmented reads, invalid markers and timeouts.

This successful loopback run is not a measured network reliability rate. Synthetic benchmark percentages are symbol-level results for the stated noise models, not whole-program delivery guarantees. GitHub Actions provides separate Linux/Windows and Python-version checks; local verification alone does not establish their results.
