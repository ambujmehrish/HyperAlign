"""Read the per-(benchmark,mode) eval logs and print ONE paper-format zero-shot table.
Ours vs GRAM (paper), R@1/R@10 for T2V and V2T, Δ@1. Robust to metric-key naming:
merges every {..} dict logged for a benchmark and picks R@1/R@10 by candidate keys.
Also dumps the raw keys it found (so the mapping can be confirmed on the first real run).
"""
import os, re, ast, glob

RES = os.environ.get('EVAL_RES_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_results')

# GRAM paper zero-shot numbers  (Table 8 = T2V R@1/R@10 ; Table 9 = V2T R@1/R@10)
PAPER = {
 ('msrvtt','tv'):     {'T2V': (52.8, 82.9), 'V2T': (49.5, 81.7)},
 ('msrvtt','tva'):    {'T2V': (54.2, 83.9), 'V2T': (50.5, 82.2)},
 ('msrvtt','tvas'):   {'T2V': (54.8, 82.9), 'V2T': (52.9, 82.9)},
 ('didemo','tv'):     {'T2V': (54.0, 80.7), 'V2T': (52.3, 80.3)},
 ('didemo','tva'):    {'T2V': (54.2, 79.3), 'V2T': (52.2, 78.9)},
 ('activitynet','tv'):{'T2V': (58.9, 91.2), 'V2T': (50.9, 85.4)},
 ('activitynet','tva'):{'T2V':(59.0, 91.1), 'V2T': (50.4, 85.8)},
 ('vatex','tv'):      {'T2V': (81.1, 99.5), 'V2T': (79.0, 98.3)},
 ('vatex','tva'):     {'T2V': (83.9, 98.6), 'V2T': (79.2, 99.0)},
 ('vatex','tvas'):    {'T2V': (83.5, 98.8), 'V2T': (82.7, 98.1)},
}
PAPER_AUDIO = {'audiocaps': (33.2, 75.3), 'vggsound': (40.6, 78.1)}   # GRAM T-AV
RET_ORDER = [('msrvtt', ['tv','tva','tvas']), ('didemo', ['tv','tva']),
             ('activitynet', ['tv','tva']),  ('vatex',  ['tv','tva','tvas'])]

# metric keys are MODE-AWARE:
#  tv / ta  -> 2-modal pairwise (ret_itm_tv/ta): video_r1 (fwd=T2V/T2A), txt_r1 (bwd=V2T/A2T).
#             (the Gramian volume for tv still loads audio -> 3-modal, WRONG for the 'tv' column,
#              so we must NOT use volume_* here.)
#  tva/tvas -> the Gramian-volume ITM retrieval: volume_ITM_T2D_r1 / volume_ITM_D2T_r1.
def mode_keys(mode):
    # -> (t2v_r1_cands, t2v_rec_cands, v2t_r1_cands, v2t_rec_cands)
    if mode in ('tv', 'ta'):
        return (['video_r1'], ['video_recall'], ['txt_r1'], ['txt_recall'])
    return (['volume_ITM_T2D_r1'], ['volume_ITM_T2D_recall'],
            ['volume_ITM_D2T_r1'], ['volume_ITM_D2T_recall'])


def merge_metrics(logpath):
    """Merge every dict literal logged in the file into one flat key->value map."""
    if not os.path.exists(logpath):
        return None
    merged = {}
    for line in open(logpath, errors='ignore'):
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                merged[k] = v
    return merged or None


def pick(merged, r1_keys, rec_keys):
    """Return (r1, r10) as floats, or (None,None). r10 parsed from 'r1/r5/r10' recall."""
    r1 = next((merged[k] for k in r1_keys if k in merged), None)
    r10 = None
    rec = next((merged[k] for k in rec_keys if k in merged), None)
    if isinstance(rec, str) and rec.count('/') == 2:
        try: r10 = float(rec.split('/')[2])
        except Exception: pass
    if r1 is not None:
        try: r1 = float(r1)
        except Exception: r1 = None
    return r1, r10


def fmt(v):        return '  —  ' if v is None else f'{v:.1f}'
def pair(r1, r10): return f'{fmt(r1)}/{fmt(r10)}'
def delta(o, g):   return '  — ' if o is None else f'{o-g:+.1f}'


def main():
    L = []
    P = L.append
    P('═' * 92)
    P('  ZERO-SHOT SUMMARY  ·  HyperAlign 4-model (Ours, best-val ckpt)  vs  GRAM (paper)')
    P('  Δ@1 = Ours − GRAM   (+ = beat paper)      [—] = not run / metric not found')
    P('═' * 92)
    P('')
    P(f'VIDEO RETRIEVAL                    T2V (R@1/R@10)              V2T (R@1/R@10)')
    P(f'{"bench":12} {"mode":5} {"Ours":13} {"GRAM":11} {"Δ@1":>5}   {"Ours":13} {"GRAM":11} {"Δ@1":>5}')
    P('─' * 92)
    raw = []
    for bench, modes in RET_ORDER:
        for i, mode in enumerate(modes):
            merged = merge_metrics(f'{RES}/{bench}_{mode}.log')
            g = PAPER[(bench, mode)]
            if merged is None:
                o_t2v = (None, None); o_v2t = (None, None)
            else:
                r1t, rect, r1v, recv = mode_keys(mode)
                o_t2v = pick(merged, r1t, rect)     # T2V (tv/ta: video_r1 ; tva/tvas: volume_ITM_T2D)
                o_v2t = pick(merged, r1v, recv)     # V2T (tv/ta: txt_r1   ; tva/tvas: volume_ITM_D2T)
                raw.append((f'{bench}_{mode}', sorted(k for k in merged if 'r1' in k or 'recall' in k)))
            bcol = bench if i == 0 else ''
            P(f'{bcol:12} {mode:5} '
              f'{pair(*o_t2v):13} {g["T2V"][0]:.1f}/{g["T2V"][1]:<5} {delta(o_t2v[0], g["T2V"][0]):>5}   '
              f'{pair(*o_v2t):13} {g["V2T"][0]:.1f}/{g["V2T"][1]:<5} {delta(o_v2t[0], g["V2T"][0]):>5}')
    P('')
    P('AUDIO (T-AV)                        Ours              GRAM        Δ@1')
    P('─' * 92)
    # AudioCaps: T-V-A (videos downloaded from YouTube) -> like-for-like with GRAM T-AV
    ac = merge_metrics(f'{RES}/audiocaps_tva.log')
    ac_r1t, ac_rect, _, _ = mode_keys('tva')                # Gramian-volume ITM (volume_ITM_T2D)
    ac_r1, ac_r10 = pick(ac, ac_r1t, ac_rect) if ac else (None, None)
    g = PAPER_AUDIO['audiocaps']
    P(f'{"AudioCaps (R@1/R@10, T-V-A)":34} {pair(ac_r1, ac_r10):17} {g[0]:.1f}/{g[1]:<5}  {delta(ac_r1, g[0]):>5}')
    # VGGSound classification reports via the same volume_T2D_* keys (rank-based) -> Acc@1 / Acc@10
    vg = merge_metrics(f'{RES}/vggsound_tav.log')          # classification -> ret_area volume_T2D_* = Acc
    vg_a1, vg_a10 = pick(vg, ['volume_T2D_r1'], ['volume_T2D_recall']) if vg else (None, None)
    g = PAPER_AUDIO['vggsound']
    P(f'{"VGGSound 5K (Acc@1/Acc@10)":34} {pair(vg_a1, vg_a10):17} {g[0]:.1f}/{g[1]:<5}  {delta(vg_a1, g[0]):>5}')
    P('─' * 92)
    P('  AudioCaps: T-V-A (videos re-downloaded from YouTube); like-for-like with GRAM T-AV (33.2/75.3).')
    P('  VGGSound: classification, run via eval_vggsound (T-AV, Acc@1/Acc@10).')
    P('')
    # raw key dump (confirm the metric-key mapping on the first real run)
    P('· raw metric keys found per run (for mapping check):')
    for name, keys in raw:
        P(f'    {name:22} {keys}')

    txt = '\n'.join(L)
    print(txt)


if __name__ == '__main__':
    main()
