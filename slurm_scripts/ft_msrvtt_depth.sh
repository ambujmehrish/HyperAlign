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
#SBATCH --job-name=ft_msrvtt_depth
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/ft_msrvtt_depth_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/ft_msrvtt_depth_%j.out
# 5th finetuning: MSR-VTT WITH depth (5-modal, tvasd). Depth is STACKED on the msrvtt finetune,
# so this inits from the ALREADY-FINETUNED msrvtt checkpoint (run ft_msrvtt.sh FIRST). GRAM recipe
# (epoch 4, bs 32) + depth data -> volume_computation5. save_best on val (ret%tvasd).
set -uo pipefail
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
source "$H/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env
cd "$H"; mkdir -p slurm_scripts/logs "$W/workdir/finetune_msrvtt_depth"
# init = the finetuned msrvtt checkpoint (not the pretrained one) — depth stacks on it
INIT=$(ls -t "$W"/workdir/finetune_msrvtt/ckpt/best_*.pt 2>/dev/null | head -1)
[ -z "$INIT" ] && { echo "ERROR: msrvtt finetune not done — run 'sbatch ft_msrvtt.sh' FIRST (depth stacks on it)"; exit 1; }
RESUME=""; ls "$W"/workdir/finetune_msrvtt_depth/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume"
echo "START $(date +%T)  finetune MSR-VTT+DEPTH from msrvtt-finetuned  (init=$INIT)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9899 \
  ./run.py --config ./config/gram/finetune_cfg/retrieval-msrvtt_depth.json \
  --output_dir "$W/workdir/finetune_msrvtt_depth" --checkpoint "$INIT" --save_best true --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
