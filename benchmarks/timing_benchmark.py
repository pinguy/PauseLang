"""Seeded synthetic channel benchmark; not evidence of real network reliability."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PauseLang_v0_7_13 import INSTRUCTIONS, SPEC, TimeQuantizer

SCENARIOS = ('clean', 'offset_10ms', 'skew_2pct', 'gaussian_0.6ms',
             'burst_8ms', 'scheduler_0.7ms')
STRATEGIES = {
    'offset_unique_1.5ms': ('offset', .0015, False),
    'offset_unique_2ms': ('offset', .002, False),
    'affine_unique_1.5ms': ('affine', .0015, False),
    'first_match_4ms_comparison': ('offset', .004, True),
}


def generate(rng, scenario, symbols):
    intended = [rng.choice(symbols) for _ in range(32)]
    expected = SPEC['sync_phrase'] + [i.pause for i in intended]
    scale = 1.02 if scenario == 'skew_2pct' else 1.0
    offset = .010 if scenario == 'offset_10ms' else .001 if scenario == 'skew_2pct' else 0
    observed = []
    previous_delay = 0
    for pause in expected:
        noise = 0
        if scenario in ('gaussian_0.6ms', 'burst_8ms'):
            noise = rng.gauss(0, .0006)
        if scenario == 'burst_8ms' and rng.random() < .05:
            noise += rng.choice([-.008, .008])
        if scenario == 'scheduler_0.7ms':
            delay = rng.expovariate(1/.0007)
            noise = delay - previous_delay  # differences of delayed arrival times
            previous_delay = delay
        observed.append(max(0, scale*pause + offset + noise))
    return intended, observed


def benchmark(traces=500, seed=714):
    rng = random.Random(seed)
    symbols = list(INSTRUCTIONS.values())
    report = {'seed': seed, 'traces_per_scenario': traces, 'symbols_per_trace': 32,
              'note': 'Synthetic independent traces, including noisy sync. The 4ms strategy is a first-match comparison, not the old effective 1.5ms default. No claim of real network reliability.',
              'scenarios': {}}
    for scenario in SCENARIOS:
        rows = {name: {'correct': 0, 'wrong': 0, 'rejected': 0, 'sync_rejected_traces': 0,
                       'confusion_matrix': defaultdict(Counter)} for name in STRATEGIES}
        for _ in range(traces):
            intended, observed = generate(rng, scenario, symbols)
            for name, (mode, guard, first_match) in STRATEGIES.items():
                row = rows[name]
                q = TimeQuantizer(calibration=mode, guard_band=guard)
                synced = q.calibrate(observed[:2])
                if not synced:
                    row['sync_rejected_traces'] += 1
                for target, pause in zip(intended, observed[2:]):
                    if not synced:
                        decoded = None
                    elif first_match:
                        decoded = next((i for i in symbols if q.in_guard_band(pause, i.pause)), None)
                    else:
                        decoded = q.decode(pause)
                    actual = decoded.opcode if decoded else 'REJECT'
                    row['confusion_matrix'][target.opcode][actual] += 1
                    row['rejected' if decoded is None else 'correct' if actual == target.opcode else 'wrong'] += 1
        for row in rows.values():
            total = traces*32
            row['total'] = total
            row['wrong_rate'] = row['wrong']/total
            row['rejection_rate'] = row['rejected']/total
            row['confusion_matrix'] = {k: dict(v) for k, v in row['confusion_matrix'].items()}
        report['scenarios'][scenario] = rows
    return report


def summary(report):
    lines = ['# Timing decoder measurements', '',
             f"Seed {report['seed']}; {report['traces_per_scenario']} traces per scenario; 32 instruction symbols per trace, plus two sync symbols.", '',
             report['note'], '',
             'Each strategy sees identical traces. Wrong means a different opcode was accepted; rejected includes all symbols in a trace whose sync failed. These are symbol-level counts, not whole-program success rates.', '',
             '| Scenario | Decoder | Correct | Wrong | Rejected | Wrong % | Rejected % |',
             '|---|---|---:|---:|---:|---:|---:|']
    for scenario, rows in report['scenarios'].items():
        for name, row in rows.items():
            lines.append(f"| {scenario} | {name} | {row['correct']} | {row['wrong']} | {row['rejected']} | {100*row['wrong_rate']:.2f} | {100*row['rejection_rate']:.2f} |")
    lines += ['', '## Interpretation', '',
              'Offset correction handles additive bias but does not remove proportional clock error. Affine correction removes noiseless scale error, but the existing 10 ms sync baseline amplifies sync jitter and extrapolates it across the ISA. It remains opt-in.', '',
              'A 4 ms first-match window overlaps neighbouring symbols and accepts some pauses as an earlier opcode. Unique-match decoding removes this table-order bias. It cannot detect noise that lands wholly inside another symbol’s window; neither guard bands nor calibration are an integrity check.', '',
              'Gaussian noise has standard deviation 0.6 ms. Burst noise adds ±8 ms to 5% of intervals on that Gaussian background. Scheduler noise is the difference between consecutive exponential arrival delays (mean 0.7 ms). Skew is +2% with a +1 ms offset. Sync is perturbed by the same scenario as instruction timing.', '',
              'Full intended-opcode → decoded-opcode counts are in `timing_results.json`. Regenerate both files with `python3 benchmarks/timing_benchmark.py --output benchmarks/timing_results.json --markdown benchmarks/RESULTS.md`.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=int, default=500)
    parser.add_argument('--seed', type=int, default=714)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    if args.traces <= 0:
        parser.error('--traces must be positive')
    report = benchmark(args.traces, args.seed)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if args.markdown:
        args.markdown.write_text(summary(report))
    print(summary(report))
