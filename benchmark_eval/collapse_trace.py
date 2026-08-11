"""Track representation collapse across runs from the [EDGES] lines.

The diagnostic that matters is batch_cos_std: the spread of caption similarities. When it falls,
the embedding space is contracting, the similarity floor (mean + sigma*sd) drops with it, and the
semantic graph degenerates into connecting everything -- which is what preceded the audio-bearing
settings losing 6+ R@1 in run 2. Comparing the trajectory between runs shows whether a change to
w_reg / w_doc actually slowed it.

  usage:  python3 benchmark_eval/collapse_trace.py name=log [name=log ...]
"""
import os
import re
import sys

LINE = re.compile(
    r'\[EDGES\] step~(\d+):.*?edges=(\d+).*?isolated=(\d+)/(\d+).*?'
    r'edge_cos=([\d.]+).*?batch_cos=([\d.]+)\(sd ([\d.]+)\)')


def trace(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors='ignore'):
        m = LINE.search(line)
        if m:
            step, edges, iso, B, ecos, bcos, sd = m.groups()
            out.append(dict(step=int(step), edges=int(edges), iso=int(iso), B=int(B),
                            ecos=float(ecos), bcos=float(bcos), sd=float(sd)))
    return out


def main(args):
    runs = {}
    for a in args:
        name, _, path = a.partition('=')
        t = trace(path)
        if not t:
            print(f"  !! no [EDGES] lines in {path}")
            continue
        runs[name] = t

    for name, t in runs.items():
        sd0 = t[0]['sd']
        print('=' * 76)
        print(f"  {name}   ({len(t)} samples)")
        print('=' * 76)
        print(f"  {'step':>7}{'sd':>8}{'% of start':>12}{'edge_cos':>10}{'isolated':>11}{'edges':>8}")
        # show first few, then every ~10th, then the last
        idx = list(range(min(4, len(t)))) + list(range(4, len(t), max(1, len(t) // 8)))
        idx = sorted(set(idx + [len(t) - 1]))
        for i in idx:
            r = t[i]
            print(f"  {r['step']:>7}{r['sd']:>8.3f}{r['sd']/sd0*100:>11.0f}%"
                  f"{r['ecos']:>10.3f}{r['iso']:>7}/{r['B']:<4}{r['edges']:>8}")
        last = t[-1]
        print(f"  -> sd {sd0:.3f} -> {last['sd']:.3f} ({last['sd']/sd0*100:.0f}% retained)"
              f"   isolated {t[0]['iso']}/{t[0]['B']} -> {last['iso']}/{last['B']}")
        print()

    if len(runs) > 1:
        print('=' * 76)
        print("  COLLAPSE RATE  (sd as % of its own start, at matched steps)")
        print('=' * 76)
        names = list(runs)
        steps = sorted(set.intersection(*[{r['step'] for r in runs[n]} for n in names]))
        show = steps[:3] + steps[3::max(1, len(steps) // 6)]
        print(f"  {'step':>7}" + ''.join(f"{n:>20}" for n in names))
        for s in sorted(set(show)):
            row = f"  {s:>7}"
            for n in names:
                r = next(x for x in runs[n] if x['step'] == s)
                row += f"{r['sd']/runs[n][0]['sd']*100:>19.0f}%"
            print(row)
        print("\n  higher = spread preserved = collapse resisted")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
