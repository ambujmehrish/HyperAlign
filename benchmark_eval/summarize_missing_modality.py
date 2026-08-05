"""Build the missing-modality degradation table from an eval_missing_modality.sh sweep.

Reads <sweep_dir>/<bench>_drop<M>_rate<R>_masked<ON|OFF>.log and prints R@1/R@10 as a function of
the drop rate, for the masked Gramian volume ON vs OFF, plus the gap between them.

The expected shape: with masking OFF (vanilla GRAM) every clip missing the modality has volume 0,
so all incomplete clips tie and R@1 falls roughly in proportion to the drop rate. With masking ON
those clips are scored at their own lower arity and the curve degrades gently.

  usage: python3 summarize_missing_modality.py <sweep_dir>
"""
import ast
import glob
import os
import re
import sys

# Same key selection as eval_summary.py: 2-modal tasks report pairwise ITM, >=3-modal report the
# Gramian-volume ITM. Keeping this identical matters -- otherwise the drop curve would be measuring
# a change of metric rather than a change of robustness.
KEYS_2MODAL = (['video_r1'], ['video_recall'])
KEYS_NMODAL = (['volume_ITM_T2D_r1'], ['volume_ITM_T2D_recall'])

NAME_RE = re.compile(r'^(?P<bench>.+)_drop(?P<mod>[vasd])_rate(?P<rate>[0-9.]+)_masked(?P<masked>ON|OFF)$')


def merge_metrics(path):
    merged = {}
    for line in open(path, errors='ignore'):
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict):
            merged.update(d)
    return merged


def pick(merged, mode):
    r1_keys, rec_keys = KEYS_2MODAL if mode in ('tv', 'ta') else KEYS_NMODAL
    r1 = next((merged[k] for k in r1_keys if k in merged), None)
    rec = next((merged[k] for k in rec_keys if k in merged), None)
    r10 = None
    if isinstance(rec, str) and rec.count('/') == 2:
        try:
            r10 = float(rec.split('/')[2])
        except ValueError:
            r10 = None
    return (float(r1) if r1 is not None else None), r10


def main(sweep_dir):
    rows = {}
    mods = set()
    bench = None
    for log in sorted(glob.glob(os.path.join(sweep_dir, '*_masked*.log'))):
        m = NAME_RE.match(os.path.basename(log)[:-4])
        if not m:
            continue
        bench = bench or m.group('bench')
        mods.add(m.group('mod'))
        mode = m.group('bench').rsplit('_', 1)[-1]
        r1, r10 = pick(merge_metrics(log), mode)
        rows[(float(m.group('rate')), m.group('masked'))] = (r1, r10)

    if not rows:
        print(f"no sweep logs found in {sweep_dir}")
        return 1

    rates = sorted({r for r, _ in rows})
    dropped = '/'.join(sorted(mods)).upper()
    print('=' * 78)
    print(f"  MISSING-MODALITY ROBUSTNESS  ·  {bench}  ·  dropped modality: {dropped}")
    print("  masked ON  = volume_computation_masked (clip scored at its own lower arity)")
    print("  masked OFF = vanilla GRAM (missing modality => singular Gram => volume 0)")
    print('=' * 78)
    print(f"{'drop rate':>10} | {'masked ON':>16} | {'masked OFF':>16} | {'Δ R@1':>7}")
    print(f"{'':>10} | {'R@1 / R@10':>16} | {'R@1 / R@10':>16} |")
    print('-' * 78)

    def fmt(v):
        a, b = v if v else (None, None)
        return f"{a:.1f} / {b:.1f}" if a is not None and b is not None else \
               (f"{a:.1f} / --" if a is not None else "-- / --")

    for r in rates:
        on, off = rows.get((r, 'ON')), rows.get((r, 'OFF'))
        d = ''
        if on and off and on[0] is not None and off[0] is not None:
            d = f"{on[0] - off[0]:+.1f}"
        print(f"{r:>10.2f} | {fmt(on):>16} | {fmt(off):>16} | {d:>7}")
    print('-' * 78)

    base = rows.get((0.0, 'ON'))
    if base and base[0]:
        print(f"  retention vs complete gallery (R@1 as % of rate=0, masked ON):")
        line = '   '.join(f"{r:.0%}:{(rows[(r,'ON')][0]/base[0]*100):.0f}%"
                          for r in rates if rows.get((r, 'ON')) and rows[(r, 'ON')][0])
        print(f"    {line}")
    print("\n  Note: the same clips are dropped in every cell (seeded permutation), so the two")
    print("  columns differ only in how a missing modality is scored.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '.'))
