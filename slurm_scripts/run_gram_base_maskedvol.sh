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
#SBATCH --job-name=gram_mv
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/gram_base_mv_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/gram_base_mv_%j.out
# GRAM-BASE ABLATION: identical data, recipe and VAST init as run_pretrain.sh, but stage A
# (hypergraph OFF, all graph aux losses zeroed). This is the attribution control -- comparing
# it against the stage-B run isolates what the hypergraph contributes from what the setup does.
# Writes to workdir_gram_base_mv/4model. Resume only triggers off that directory, so a first run
# always starts from the VAST pretrained weights.
echo "START=$(date +%T) [GRAM-BASE ablation (stage A, hypergraph OFF) -> ./workdir_gram_base_mv/4model]"
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
source "$H/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env
cd "$H"
# auto-resume: continue from the latest optimizer checkpoint after a crash
RESUME=""
if ls "$W"/workdir_gram_base_mv/4model/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  # --resume is store_true; passing a value makes argparse reject the bare "true"
  RESUME="--resume"; echo "RESUME: checkpoint found -> continue"
else
  echo "FRESH"
fi
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9893 \
  ./run.py --config ./config/gram/pretrain_cfg/gram_base_maskedvol.json \
  --output_dir "$W/workdir_gram_base_mv/4model" --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
