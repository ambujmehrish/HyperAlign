#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:28:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=ha_smoke
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/ha_smoke_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/slurm_scripts/logs/ha_smoke_%j.out
# HyperAlign training smoke: 24 steps of the real recipe (bs256/lr2e-5/frames2) + hypergraph + validation + save_best.
# Fresh output_dir; best_531 (workdir_v2full) and workdir_pretrain untouched.
echo "START=$(date +%T) [HyperAlign training smoke, 24 steps + validate]"
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
source "$H/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env
cd "$H"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9877 \
  ./run.py --config ./config/gram/pretrain_cfg/hyperalign_smoke.json \
  --output_dir "$W/workdir_smoke_ha" --checkpointing true 2>&1
echo "EXIT=$? DONE=$(date +%T)"
