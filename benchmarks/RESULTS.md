# Timing decoder measurements

Seed 714; 500 traces per scenario; 32 instruction symbols per trace, plus two sync symbols.

Synthetic independent traces, including noisy sync. The 4ms strategy is a first-match comparison, not the old effective 1.5ms default. No claim of real network reliability.

Each strategy sees identical traces. Wrong means a different opcode was accepted; rejected includes all symbols in a trace whose sync failed. These are symbol-level counts, not whole-program success rates.

| Scenario | Decoder | Correct | Wrong | Rejected | Wrong % | Rejected % |
|---|---|---:|---:|---:|---:|---:|
| clean | offset_unique_1.5ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| clean | offset_unique_2ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| clean | affine_unique_1.5ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| clean | first_match_4ms_comparison | 16000 | 0 | 0 | 0.00 | 0.00 |
| offset_10ms | offset_unique_1.5ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| offset_10ms | offset_unique_2ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| offset_10ms | affine_unique_1.5ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| offset_10ms | first_match_4ms_comparison | 16000 | 0 | 0 | 0.00 | 0.00 |
| skew_2pct | offset_unique_1.5ms | 450 | 9575 | 5975 | 59.84 | 37.34 |
| skew_2pct | offset_unique_2ms | 2230 | 11649 | 2121 | 72.81 | 13.26 |
| skew_2pct | affine_unique_1.5ms | 16000 | 0 | 0 | 0.00 | 0.00 |
| skew_2pct | first_match_4ms_comparison | 425 | 15120 | 455 | 94.50 | 2.84 |
| gaussian_0.6ms | offset_unique_1.5ms | 15349 | 0 | 651 | 0.00 | 4.07 |
| gaussian_0.6ms | offset_unique_2ms | 15899 | 1 | 100 | 0.01 | 0.62 |
| gaussian_0.6ms | affine_unique_1.5ms | 1367 | 3968 | 10665 | 24.80 | 66.66 |
| gaussian_0.6ms | first_match_4ms_comparison | 14718 | 1282 | 0 | 8.01 | 0.00 |
| burst_8ms | offset_unique_1.5ms | 13337 | 188 | 2475 | 1.18 | 15.47 |
| burst_8ms | offset_unique_2ms | 13934 | 419 | 1647 | 2.62 | 10.29 |
| burst_8ms | affine_unique_1.5ms | 1128 | 3658 | 11214 | 22.86 | 70.09 |
| burst_8ms | first_match_4ms_comparison | 13023 | 2228 | 749 | 13.93 | 4.68 |
| scheduler_0.7ms | offset_unique_1.5ms | 12351 | 107 | 3542 | 0.67 | 22.14 |
| scheduler_0.7ms | offset_unique_2ms | 14107 | 254 | 1639 | 1.59 | 10.24 |
| scheduler_0.7ms | affine_unique_1.5ms | 1298 | 2968 | 11734 | 18.55 | 73.34 |
| scheduler_0.7ms | first_match_4ms_comparison | 12777 | 3217 | 6 | 20.11 | 0.04 |

## Interpretation

Offset correction handles additive bias but does not remove proportional clock error. Affine correction removes noiseless scale error, but the existing 10 ms sync baseline amplifies sync jitter and extrapolates it across the ISA. It remains opt-in.

A 4 ms first-match window overlaps neighbouring symbols and accepts some pauses as an earlier opcode. Unique-match decoding removes this table-order bias. It cannot detect noise that lands wholly inside another symbol’s window; neither guard bands nor calibration are an integrity check.

Gaussian noise has standard deviation 0.6 ms. Burst noise adds ±8 ms to 5% of intervals on that Gaussian background. Scheduler noise is the difference between consecutive exponential arrival delays (mean 0.7 ms). Skew is +2% with a +1 ms offset. Sync is perturbed by the same scenario as instruction timing.

Full intended-opcode → decoded-opcode counts are in `timing_results.json`. Regenerate both files with `python3 benchmarks/timing_benchmark.py --output benchmarks/timing_results.json --markdown benchmarks/RESULTS.md`.
