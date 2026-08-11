"""Side-by-side zero-shot comparison of several checkpoints, against the GRAM paper.

Each argument is name=<eval results dir> (the EVAL_RES_DIR a run wrote to). Prints one table of
T2V R@1 per (benchmark, mode) with a column per run and the paper's number, plus each run's mean
gap. This is the table the paper needs; running eval_summary.py once per run prints three screens
that cannot be read together.

  usage:  python3 benchmark_eval/compare_runs.py gram_base=path/to/dir hyperalign=path/to/dir ...
          python3 benchmark_eval/compare_runs.py --ref=gram_official gram_official=... run2=...

The paper column is NOT a fair reference for our numbers. GRAM's own released checkpoint, run
through this harness, scores ~3.6 R@1 below its published table -- and by benchmark that shortfall
is ~1.4 (msrvtt), ~3.2 (didemo, activitynet), ~6.3 (vatex). Measuring our runs against the paper
therefore charges them for a harness difference they did not cause. --ref=<name> re-baselines the
gap column onto a run evaluated HERE, which is the apples-to-apples comparison.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_summary import merge_metrics, mode_keys, pick, PAPER, RET_ORDER  # noqa: E402


def collect(res_dir):
    """-> {(bench, mode): (t2v_r1, v2t_r1)}"""
    out = {}
    for bench, modes in RET_ORDER:
        for mode in modes:
            log = os.path.join(res_dir, f"{bench}_{mode}.log")
            merged = merge_metrics(log)
            if not merged:
                continue
            t2v_r1k, t2v_reck, v2t_r1k, v2t_reck = mode_keys(mode)
            t2v, _ = pick(merged, t2v_r1k, t2v_reck)
            v2t, _ = pick(merged, v2t_r1k, v2t_reck)
            out[(bench, mode)] = (t2v, v2t)
    return out


def main(args):
    ref = None
    rest = []
    for a in args:
        if a.startswith('--ref='):
            ref = a.split('=', 1)[1]
        else:
            rest.append(a)
    runs = {}
    for a in rest:
        name, _, path = a.partition('=')
        if not os.path.isdir(path):
            print(f"  !! not a directory: {path}")
            continue
        runs[name] = collect(path)
    if not runs:
        return 1
    if ref and ref not in runs:
        print(f"  !! --ref={ref} is not one of the runs given: {', '.join(runs)}")
        return 1

    names = list(runs)
    w = max(12, max(len(n) for n in names) + 2)
    print('=' * (22 + w * (len(names) + 1)))
    print("  ZERO-SHOT T2V R@1  ·  all runs vs GRAM (paper)")
    print('=' * (22 + w * (len(names) + 1)))
    print(f"  {'bench':<13}{'mode':<7}" + ''.join(f"{n:>{w}}" for n in names) + f"{'PAPER':>{w}}")
    print('-' * (22 + w * (len(names) + 1)))

    gaps = {n: [] for n in names}
    for bench, modes in RET_ORDER:
        for mode in modes:
            paper = PAPER.get((bench, mode), {}).get('T2V', (None,))[0]
            row = f"  {bench:<13}{mode:<7}"
            for n in names:
                v = runs[n].get((bench, mode), (None, None))[0]
                row += f"{v:>{w}.1f}" if v is not None else f"{'—':>{w}}"
                if v is not None and paper is not None:
                    gaps[n].append(v - paper)
            row += f"{paper:>{w}.1f}" if paper is not None else f"{'—':>{w}}"
            print(row)
    print('-' * (22 + w * (len(names) + 1)))
    row = f"  {'mean gap vs paper':<20}"
    for n in names:
        g = gaps[n]
        row += f"{(sum(g)/len(g)):>{w}.1f}" if g else f"{'—':>{w}}"
    print(row + f"{'0.0':>{w}}")
    print()
    for n in names:
        if gaps[n]:
            print(f"  {n}: {len(gaps[n])} settings, mean {sum(gaps[n])/len(gaps[n]):+.1f}, "
                  f"worst {min(gaps[n]):+.1f}, best {max(gaps[n]):+.1f}")

    if ref:
        print()
        print('=' * (22 + w * (len(names) + 1)))
        print(f"  SAME-HARNESS DELTA vs {ref}   (both evaluated here; no paper involved)")
        print('=' * (22 + w * (len(names) + 1)))
        others = [n for n in names if n != ref]
        print(f"  {'bench':<13}{'mode':<7}" + ''.join(f"{n:>{w}}" for n in others))
        print('-' * (22 + w * (len(names) + 1)))
        rgaps = {n: [] for n in others}
        for bench, modes in RET_ORDER:
            for mode in modes:
                base = runs[ref].get((bench, mode), (None, None))[0]
                row = f"  {bench:<13}{mode:<7}"
                for n in others:
                    v = runs[n].get((bench, mode), (None, None))[0]
                    if v is None or base is None:
                        row += f"{'—':>{w}}"
                    else:
                        row += f"{v - base:>+{w}.1f}"
                        rgaps[n].append(v - base)
                print(row)
        print('-' * (22 + w * (len(names) + 1)))
        row = f"  {'mean delta':<20}"
        for n in others:
            g = rgaps[n]
            row += f"{(sum(g)/len(g)):>+{w}.1f}" if g else f"{'—':>{w}}"
        print(row)
        print(f"\n  positive = better than {ref} on the SAME evaluation code, galleries and configs.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
