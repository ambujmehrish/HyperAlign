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
#SBATCH --job-name=ft_vatex
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/ft_vatex_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/ft_vatex_%j.out
# Finetune HyperAlign (from the pretrained checkpoint) on VATEX — GRAM recipe (epoch 3, bs 64), save_best.
set -uo pipefail
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
source "$H/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env
cd "$H"; mkdir -p slurm_scripts/logs "$W/workdir/finetune_vatex"
INIT="$W/workdir_v2full/4model/ckpt/best_ret%tvas--msrvtt_ret_ret_area_forward.pt"
RESUME=""; ls "$W"/workdir/finetune_vatex/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true"
echo "START $(date +%T)  finetune VATEX from pretrained checkpoint  (init=$INIT)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9898 \
  ./run.py --config ./config/gram/finetune_cfg/retrieval-vatex.json \
  --output_dir "$W/workdir/finetune_vatex" --checkpoint "$INIT" --save_best true --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
