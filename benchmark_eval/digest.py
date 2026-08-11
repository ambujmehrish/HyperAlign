"""Compact digest of a batch of runs: one screen instead of hundreds of lines.

  usage:  python3 benchmark_eval/digest.py  name=path/to/train.out  [name=path ...]

Prints, per run: the validation peak and where it fell in training, the first/last graph-health
line, and the gate drift. Then a side-by-side of the peaks, which is the comparison that matters
(stage A vs stage B, pre-fix vs post-fix).
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_curve import parse  # noqa: E402

EDGES = re.compile(r'\[EDGES\][^\n]*')
GATE = re.compile(r'\[GATE\][^\n]*\[([^\]]*)\]')


def scan(path):
    out = {'edges_first': None, 'edges_last': None, 'gate_first': None, 'gate_last': None}
    if not os.path.exists(path):
        return None
    for line in open(path, errors='ignore'):
        m = EDGES.search(line)
        if m:
            out['edges_first'] = out['edges_first'] or m.group(0)
            out['edges_last'] = m.group(0)
        g = GATE.search(line)
        if g:
            try:
                v = [float(x) for x in g.group(1).split(',')]
            except ValueError:
                continue
            out['gate_first'] = out['gate_first'] or v
            out['gate_last'] = v
    out['curve'] = parse(path)
    return out


def peak(curve, sec='ret_itm_area'):
    pts = curve.get(sec) or []
    if not pts:
        return None
    best = max(p[1] for p in pts)
    step = next(s for s, v in pts if v == best)
    last = pts[-1][0]
    return best, step, last, len(pts)


def main(args):
    runs = {}
    for a in args:
        name, _, path = a.partition('=')
        d = scan(path)
        if d is None:
            print(f"  !! missing log: {path}")
            continue
        runs[name] = d

    for name, d in runs.items():
        print('=' * 70)
        print(f"  {name}")
        print('=' * 70)
        p = peak(d['curve'])
        if p:
            best, step, last, n = p
            print(f"  val peak (MSR-VTT tvas, volume-ITM R@1): {best:.1f} at step {step}"
                  f"  = {step/last*100:.0f}% of {last}   [{n} validations]")
        else:
            print("  val peak: no validation blocks parsed")
        for k, label in (('edges_first', 'graph @start'), ('edges_last', 'graph @end')):
            if d[k]:
                print(f"  {label}: {d[k].split('] ',1)[-1]}")
        if d['gate_first'] and d['gate_last']:
            a = ', '.join(f"{x:.3f}" for x in d['gate_first'])
            b = ', '.join(f"{x:.3f}" for x in d['gate_last'])
            print(f"  gates: [{a}] -> [{b}]")
        print()

    peaks = {n: peak(d['curve']) for n, d in runs.items()}
    peaks = {n: p for n, p in peaks.items() if p}
    if len(peaks) > 1:
        print('=' * 70)
        print("  SIDE BY SIDE  (MSR-VTT tvas, volume-ITM R@1)")
        print('=' * 70)
        print(f"  {'run':<26}{'best R@1':>10}{'at step':>10}{'% of run':>10}")
        for n, (best, step, last, _) in peaks.items():
            print(f"  {n:<26}{best:>10.1f}{step:>10}{step/last*100:>9.0f}%")
        win = max(peaks, key=lambda n: peaks[n][0])
        print(f"\n  highest: {win} ({peaks[win][0]:.1f})")
        early = [n for n, p in peaks.items() if p[1] / p[2] < 0.4 and p[3] >= 3]
        if early:
            print(f"  peaked in the first 40% of training: {', '.join(early)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
