import os
import torch
from os.path import join
from utils.logger import LOGGER




def _step_of(fname):
    """model_step_1234.pt -> 1234 (-1 if unparseable, so odd files sort first and get pruned)."""
    try:
        return int(fname.rsplit('_', 1)[-1].split('.')[0])
    except (ValueError, IndexError):
        return -1


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt',remove_before_ckpt=True,
                 keep_last_n=1):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix
        self.remove_before_ckpt = remove_before_ckpt
        # How many recent model checkpoints to retain. 1 = original behaviour (only the newest
        # survives). >1 keeps a rolling window so the best checkpoint can be chosen post-hoc on
        # the real benchmarks instead of trusting a single save_best hit on one val set.
        # Optimizer states are always pruned to the newest -- they exist only for resume.
        self.keep_last_n = max(1, int(keep_last_n))
    def save(self, model, step, optimizer=None, best_indicator=None, save_best=False):
        ###remove previous model, keeping the most recent keep_last_n
        previous_state = [i  for i in os.listdir(self.output_dir) if i.startswith('model')]
        # if not self.pretraining:
        if self.remove_before_ckpt:
            # the checkpoint about to be written counts towards the window
            stale = sorted(previous_state, key=_step_of)[:-(self.keep_last_n - 1)] \
                    if self.keep_last_n > 1 else previous_state
            for p in stale:
                os.remove(os.path.join(self.output_dir,p))
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        torch.save(state_dict, output_model_file)

        if save_best:
            for k in best_indicator:
                if best_indicator[k]:
                    torch.save(state_dict, join(self.output_dir,
                                 f"best_{k}.{self.suffix}"))

        if optimizer is not None:
            if hasattr(optimizer, '_amp_stash'):
                pass  # fp16/amp optimizer state is not separately saved
            previous_state = [i  for i in os.listdir(self.output_dir) if i.startswith('optimizer')]
            if self.remove_before_ckpt:
                for p in previous_state:
                    os.remove(os.path.join(self.output_dir,p))
            torch.save(optimizer.state_dict(), f'{self.output_dir}/optimizer_step_{step}.pt')
