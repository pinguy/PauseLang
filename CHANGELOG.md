# Changes

## 0.7.14

The module stays named `PauseLang_v0_7_13.py` for import compatibility.

- Resolve timing defaults at construction; expose per-VM guard/calibration settings.
- Reject ambiguous timing windows instead of accepting the first matching opcode.
- Reject NaN, infinity and non-positive instruction pauses without a Python crash.
- Acquire additive sync offsets up to ±20 ms; offer bounded affine clock-skew correction explicitly. Preserve offset correction as the default because the short sync baseline amplifies jitter.
- Clear calibration on VM reset; validate stream lengths before changing calibration.
- Reject extra compiler tokens and malformed labels with accurate source line numbers; parse signed decimal operands explicitly.
- Make `STOREI_POP` consume exactly one value. An additional discard now requires an explicit `POP`; label positions follow the corrected expansion.
- Correct the uninitialised `LOADI` test to push 42 before `SETIX` and verify the IX value and absence of errors.
- Check META memory bounds for both loads and stores; invalid stores retain the stack value.
- Allow re-entry of an existing loop at maximum loop depth.
- Report `TRAP_STORM`; keep gas-used statistics within the configured budget.
- Compute integer division without a floating-point intermediate, including large input integers.
- Extract the original 28-test suite from the VM and expose independent discoverable tests. Add focused VM/compiler/timing, framed transport, checksum and WAV regressions; add Linux/Windows GitHub Actions coverage.
- Add seeded timing-noise measurements and per-opcode confusion matrices. Report accepted wrong opcodes separately from rejects, including sync failures.
- Escape unsupported CLI decoration on legacy output encodings (including redirected Windows cp1252) and write benchmark reports as UTF-8.
- Encode the final WAV pause with a terminal click; validate WAV timing and sample-rate inputs; handle empty streams.

### TCP compatibility change

Both sender and receiver must be updated. The new versioned `PLT1` framing uses signed 32-bit operands instead of silently masking values to unsigned 16-bit. A final timing marker measures the last instruction instead of padding with a fabricated `HALT`.

The demo sends at 4× timing scale (20 ms adjacent-symbol spacing; 6 ms physical tolerance). Frames carry an unkeyed CRC32 over operands and canonical timing microseconds. The receiver validates all timings and the checksum before VM execution, then halts on runtime traps. It rejects truncated frames, bad headers/markers, out-of-range counts and socket timeouts. The CRC is accidental-corruption detection, not encryption or authentication.

### Remaining experimental limits

Timing symbols can be corrupted into other valid symbols. The VM alone cannot distinguish those; the transport checksum can detect accidental stream changes but is not adversary-resistant. Clock-rate estimation with only two closely spaced sync symbols is noise-sensitive. Wider/repeated sync, error-correcting codes, and authenticated transport remain future experiments. No licence grant is introduced by this maintenance change; the repository owner still needs to choose licensing terms.
