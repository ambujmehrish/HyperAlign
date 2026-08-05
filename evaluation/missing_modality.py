"""Synthetic missing-modality evaluation harness.

Motivation
----------
GRAM scores a clip by the Gramian volume of its modality vectors. The loader zero-fills any
modality it could not load, and a zero vector makes that modality's Gram row/column zero, so the
matrix is singular: det(G) == 0 and the volume collapses to 0 for EVERY query. Incomplete clips do
not merely score worse -- they all tie at zero and become unrankable.

volume_computation_masked replaces a missing modality's row/column with an orthonormal phantom
axis, which contributes a factor of exactly 1 to the determinant, so the clip is scored at its own
lower arity (a tva clip missing audio is scored exactly as tv).

This module drops modalities from a controlled fraction of GALLERY clips at eval time so the two
behaviours can be measured on the same checkpoint and the same benchmark. The query text is never
dropped -- it is the anchor.

Usage (environment-driven; no config plumbing, eval-harness concern only):
    HA_DROP_MOD    modality letter to drop from the gallery: v | a | s | d   (unset => disabled)
    HA_DROP_RATE   fraction of gallery clips to drop it from, 0.0 .. 1.0     (default 0.0)
    HA_DROP_SEED   RNG seed selecting WHICH clips lose it                    (default 1234)

The selection is a seeded permutation of clip indices, so for a fixed (seed, gallery size) the same
clips are dropped across every run. That is what makes masked-volume-on vs -off, and one checkpoint
vs another, comparable rather than differently-corrupted.
"""
import os

import torch

from utils.logger import LOGGER


def drop_config():
    """Read the harness settings from the environment. Returns None when disabled."""
    mod = (os.environ.get('HA_DROP_MOD') or '').strip().lower()
    if not mod:
        return None
    if mod not in ('v', 'a', 's', 'd'):
        raise ValueError(f"HA_DROP_MOD must be one of v/a/s/d, got {mod!r}")
    try:
        rate = float(os.environ.get('HA_DROP_RATE', '0.0'))
    except ValueError:
        raise ValueError(f"HA_DROP_RATE must be a float, got {os.environ.get('HA_DROP_RATE')!r}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"HA_DROP_RATE must be in [0, 1], got {rate}")
    seed = int(os.environ.get('HA_DROP_SEED', '1234'))
    return {'mod': mod, 'rate': rate, 'seed': seed}


def dropped_indices(n_clips, rate, seed):
    """Deterministic set of gallery rows that lose the modality: first round(rate*n) of a seeded
    permutation. Same (n, rate, seed) -> same rows, on any machine, in any run."""
    n_drop = int(round(rate * n_clips))
    if n_drop <= 0:
        return None
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_clips, generator=g)[:n_drop]


def apply(feats, mods, cfg=None):
    """Zero-fill cfg['mod'] for a deterministic fraction of gallery clips.

    feats : list of (B, dim) gallery tensors, ordered as the letters of `mods`.
    mods  : the modality letters for this task, e.g. 'va' for ret%tva.
    Returns (feats, info) with feats a NEW list when anything was dropped (inputs untouched).

    Zeroing is exactly how a genuinely missing modality reaches the volume, so downstream
    presence detection (norm > 0.5) and the masked/vanilla volume paths need no special casing.
    """
    cfg = drop_config() if cfg is None else cfg
    if not cfg or cfg['rate'] <= 0:
        return feats, None
    order = [m for m in 'vasd' if m in mods]
    if cfg['mod'] not in order:
        LOGGER.info(f"[MISSING-MOD] task modalities '{mods}' do not include '{cfg['mod']}' "
                    f"-> nothing dropped for this run")
        return feats, None
    if len(order) <= 1:
        # dropping the only gallery modality would leave the volume with text alone
        LOGGER.info(f"[MISSING-MOD] task '{mods}' has a single gallery modality -> skipping drop")
        return feats, None

    pos = order.index(cfg['mod'])
    n = feats[pos].shape[0]
    idx = dropped_indices(n, cfg['rate'], cfg['seed'])
    if idx is None:
        return feats, None

    out = list(feats)
    f = out[pos].clone()
    f[idx.to(f.device)] = 0.0
    out[pos] = f
    info = {'mod': cfg['mod'], 'rate': cfg['rate'], 'seed': cfg['seed'],
            'n_dropped': int(idx.numel()), 'n_clips': int(n)}
    LOGGER.info(f"[MISSING-MOD] dropped '{cfg['mod']}' from {info['n_dropped']}/{n} gallery clips "
                f"(rate={cfg['rate']}, seed={cfg['seed']})")
    return out, info
