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
# alignment (gate initialized to 1.0, signed refinement, w_doc 1.0).
# The recipe is read off GRAM's own released checkpoint, whose training directory is named
# `finetuneVolume256batchlossonlyvolume4Mod120k` and whose file is `model_step_459.pt`:
# batch 256, volume-only loss, 4 modalities, 120k samples => 468 steps = ONE epoch. The earlier
# "epoch5/bs128" in this header was a guess and both halves of it were wrong.
# Writes to workdir_pretrain/4model. Resume only triggers off that directory, so a first run
# always starts from the VAST pretrained weights.
echo "START=$(date +%T) [HyperAlign pretrain, GRAM recipe + hypergraph -> ./workdir_pretrain/4model]"
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
source "$H/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env
cd "$H"
# auto-resume: continue from the latest optimizer checkpoint after a crash
RESUME=""
if ls "$W"/workdir_pretrain/4model/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  RESUME="--resume true"; echo "RESUME: checkpoint found -> continue"
else
  echo "FRESH"
fi
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9893 \
  ./run.py --config ./config/gram/pretrain_cfg/hyperalign.json \
  --output_dir "$W/workdir_pretrain/4model" --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
