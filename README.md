# HyperAlign

## Storage layout on Leonardo

Home has a 50 GB quota, so the project is split across three locations:

| What | Where | Env var |
|---|---|---|
| Code (this repo) | `/leonardo/home/userexternal/amehrish/HyperAlign` | `HA_CODE` |
| Big folders: `pretrained_weights`, `datasets`, `data`, `workdir_smoke_ha`, `workdir_ftsmoke`, `workdir_v2full` (+ new training outputs `workdir/`, `workdir_pretrain/`) | `/leonardo_work/AIFAC_S07_041/HyperAlign` | `HA_WORK` |
| Raw dataset (videos / audios / annotations) | `/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset` | `HA_DATA` |

All absolute paths in the configs and slurm scripts follow this layout. Model code and
some configs also use paths **relative to the repo root** (`./pretrained_weights/...`,
`datasets/annotations/...`), which are resolved through symlinks created by the setup
script below. All training outputs (checkpoints) are written to `$HA_WORK`, never to home.

## One-time setup

```bash
cd /leonardo/home/userexternal/amehrish/HyperAlign
bash setup_paths.sh
```

This symlinks the big folders from `$HA_WORK` into the repo root (symlinks cost no
quota), creates the log directories, and sanity-checks that the key files exist
(VAST foundation checkpoint, BERT weights, annotations, dataset root). Fix anything
it reports as `MISSING` before submitting jobs.

The slurm scripts source `slurm_scripts/env.sh`, which activates the
`Multimodal_hypergraph` conda env. It looks for conda in the usual places
(`$HOME/miniconda3`, `$HOME/anaconda3`, `$HA_WORK/...`); if yours is elsewhere:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
```

## Running

```bash
# 24-step training smoke (debug QOS, ~30 min)
sbatch slurm_scripts/smoke_train.sh

# full pretraining (auto-resumes from $HA_WORK/workdir_pretrain/4model)
sbatch slurm_scripts/run_pretrain.sh

# finetuning: all 5 benchmarks (msrvtt_depth waits on msrvtt via slurm dependency)
bash slurm_scripts/finetune_all.sh

# zero-shot eval (12 benchmark/mode configs) and finetuned-checkpoint eval
sbatch benchmark_eval/eval_zeroshot.sh
sbatch benchmark_eval/eval_finetune.sh msrvtt
```

Notes:
- The VAST foundation checkpoint referenced by the pretrain configs is expected at
  `$HA_WORK/pretrained_weights/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt`
  (`setup_paths.sh` searches `pretrained_weights/` and symlinks it if it lives elsewhere).
- SLURM jobs are billed to account `AIFAC_S07_041` (`#SBATCH -A`); change it in
  `slurm_scripts/*.sh` and `benchmark_eval/*.sh` if your account differs.
- Job logs go to `slurm_scripts/logs/` and `benchmark_eval/{logs,smoke_logs}/` in home
  (small text files only).
