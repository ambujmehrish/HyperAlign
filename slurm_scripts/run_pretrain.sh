#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=pretrain
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/train4_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/train4_%j.out
# VAST-150k pretraining launcher — GRAM recipe (epoch1/bs256/lr2e-5/frames2) with hypergraph
# alignment.
# The recipe is read off GRAM's own released checkpoint, whose training directory is named
# `finetuneVolume256batchlossonlyvolume4Mod120k` and whose file is `model_step_459.pt`:
# batch 256, volume-only loss, 4 modalities, 120k samples => 468 steps = ONE epoch. The earlier
# "epoch5/bs128" in this header was a guess and both halves of it were wrong.
#
# Usage:
#   sbatch slurm_scripts/run_pretrain.sh                       # -> workdir_pretrain
#   HA_RUN_NAME=distill sbatch slurm_scripts/run_pretrain.sh   # -> workdir_distill
#   HA_CFG=config/gram/pretrain_cfg/gram_base.json HA_RUN_NAME=base sbatch ...
#
# HA_RUN_NAME exists because the workdir used to be hardcoded, so every experiment wrote to
# workdir_pretrain and collided with the previous one: two concurrent runs corrupt each other's
# save_best, and renaming the directory to make room breaks the job still writing to it. Name the
# run instead of moving directories afterwards.
RUN=${HA_RUN_NAME:-pretrain}
CFG=${HA_CFG:-./config/gram/pretrain_cfg/hyperalign.json}
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
WD="$W/workdir_$RUN/4model"
echo "START=$(date +%T) [HyperAlign pretrain: run='$RUN' cfg='$CFG' -> $WD]"
source "$H/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env
cd "$H"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }

# Refuse to start if another live job is already training into this workdir. Two writers silently
# interleave their checkpoints and the resulting best_*.pt belongs to neither run. A stale lock
# from a crashed job is ignored, since squeue no longer knows that id.
mkdir -p "$W/workdir_$RUN"
LOCK="$W/workdir_$RUN/.running"
OTHER=$(cat "$LOCK" 2>/dev/null)
if [ -n "$OTHER" ] && [ "$OTHER" != "$SLURM_JOB_ID" ] \
   && [ -n "$(squeue -h -j "$OTHER" 2>/dev/null)" ]; then
  echo "ERROR: job $OTHER is already training into $WD." >&2
  echo "       Use HA_RUN_NAME=<something-else> rather than sharing a workdir." >&2
  exit 1
fi
echo "$SLURM_JOB_ID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# auto-resume: continue from the latest optimizer checkpoint after a crash
RESUME=""
if ls "$WD"/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  # --resume is store_true; passing a value makes argparse reject the bare "true"
  RESUME="--resume"; echo "RESUME: checkpoint found -> continue"
else
  echo "FRESH"
fi
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9893 \
  ./run.py --config "$CFG" \
  --output_dir "$WD" --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
