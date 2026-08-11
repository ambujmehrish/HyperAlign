"""Print the validation curve from a training log, and where the best step actually fell.

The key question when comparing runs is not the final number but the SHAPE: run 1's best-val
checkpoint fired ~21% into training while the gates kept growing, which is either general
overfitting or the hypergraph causing it. Comparing this curve between stage A (gram_base) and
stage B (hyperalign) separates the two.

  usage:  python3 benchmark_eval/val_curve.py <train log> [<train log> ...]
"""
import ast
import os
import re
import sys

# section banner -> which metric key that section reports
SECTIONS = {
    'ret_itm_area': 'volume_ITM_T2D_r1',   # >=3-modal: Gramian-volume ITM (what eval_summary uses)
    'ret_area_forward': 'volume_T2D_r1',   # raw volume, no ITM rerank
    'cosine_TV': 'forward_r1',             # plain text-video cosine
}
STEP_RE = re.compile(r'step[: ]+(\d+)')
SEC_RE = re.compile(r'evaluation--[^-]*--\w+?_(' + '|'.join(SECTIONS) + r')[=\s]')


def parse(path):
    """-> {section: [(step, value), ...]} in log order."""
    out = {s: [] for s in SECTIONS}
    seen = {s: set() for s in SECTIONS}
    step, sec = None, None
    for line in open(path, errors='ignore'):
        m = SEC_RE.search(line)
        if m:
            # The log emits TWO banners per validation: the current "step N" result, and a
            # "history best step: N" recap that repeats the best-so-far. Only the first is a
            # point on the curve. Dropping the recap by "same step as the previous row" fails
            # as soon as the best is not the most recent step -- the two then alternate
            # (524, 124, 549, 124, ...) and every recap survives, which corrupts pts[-1] and
            # with it every "% of run" figure and the peaked-early warning.
            if 'history' in line.lower():
                sec = None
                continue
            sec = m.group(1)
            s = STEP_RE.search(line.split(sec, 1)[1])
            if s:
                step = int(s.group(1))
            continue
        if sec and step is not None:
            d = re.search(r"\{.*\}", line)
            if not d:
                continue
            try:
                vals = ast.literal_eval(d.group(0))
            except Exception:
                continue
            key = SECTIONS[sec]
            if isinstance(vals, dict) and key in vals:
                if step not in seen[sec]:          # belt and braces: one row per step
                    seen[sec].add(step)
                    out[sec].append((step, float(vals[key])))
                sec = None
    return out


def report(path):
    name = os.path.basename(path)
    data = parse(path)
    print('=' * 72)
    print(f"  {name}")
    print('=' * 72)
    any_rows = False
    for sec, key in SECTIONS.items():
        pts = data[sec]
        if not pts:
            continue
        any_rows = True
        best_v = max(p[1] for p in pts)
        best_s = next(s for s, v in pts if v == best_v)
        last_s = pts[-1][0]
        frac = best_s / last_s * 100 if last_s else 0
        print(f"\n  {sec}  ({key})")
        for s, v in pts:
            bar = '#' * int(round(v / 2))
            print(f"    step {s:>6}  {v:>6.1f}  {bar}{'   <- best' if v == best_v else ''}")
        print(f"    best {best_v:.1f} at step {best_s} = {frac:.0f}% of {last_s} steps seen")
        if frac < 40 and len(pts) >= 3:
            print("    ^ peaked early: validation degraded over the remaining ~%.0f%% of training"
                  % (100 - frac))
    if not any_rows:
        print("  no validation blocks parsed yet (job may still be before its first validation)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
        print()
