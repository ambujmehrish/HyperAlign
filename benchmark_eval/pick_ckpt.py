"""Pick the checkpoint to init downstream stages from a training run's ckpt dir.

GRAM trains with save_best=True on an MSR-VTT val set. That writes TWO kinds of files:
  best_<evalname>.pt   <- weights at the best-val step (save_best output)
  model_step_<N>.pt    <- latest/final weights (remove_before_ckpt keeps only the last)

GRAM's *released* 4-model checkpoint is model_step_249.pt, but 150k//128*5 ~= 5865 steps,
so 249 << final => GRAM released the BEST-val checkpoint, not the final one. To reproduce,
we prefer best_*.pt and only fall back to the latest model_step_*.pt (then a placeholder).
"""
import os, glob


def pick_ckpt(ckpt_dir, placeholder):
    best = sorted(glob.glob(os.path.join(ckpt_dir, 'best_*.pt')))
    if best:
        return best[0]
    steps = sorted(glob.glob(os.path.join(ckpt_dir, 'model_step_*.pt')),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    return steps[-1] if steps else placeholder
