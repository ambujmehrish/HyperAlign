#!/bin/bash
#SBATCH -A IscrC_CASPER-A_0
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=ft_didemo
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/slurm_scripts/logs/ft_didemo_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/slurm_scripts/logs/ft_didemo_%j.out
# Finetune HyperAlign (from the pretrained checkpoint) on DiDeMo — GRAM recipe (epoch 40, bs 64), save_best.
set -uo pipefail
H=/leonardo_work/IscrC_GMEG/anag0000/HyperAlign
source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
cd "$H"; mkdir -p slurm_scripts/logs workdir/finetune_didemo
INIT="$H/workdir_v2full/4model/ckpt/best_ret%tvas--msrvtt_ret_ret_area_forward.pt"
RESUME=""; ls workdir/finetune_didemo/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true"
echo "START $(date +%T)  finetune DiDeMo from pretrained checkpoint  (init=$INIT)"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9896 \
  ./run.py --config ./config/gram/finetune_cfg/retrieval-didemo.json \
  --output_dir ./workdir/finetune_didemo --checkpoint "$INIT" --save_best true --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
