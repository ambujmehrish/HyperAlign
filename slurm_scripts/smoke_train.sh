#!/bin/bash
#SBATCH -A IscrC_CASPER-A_0
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:28:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=ha_smoke
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/slurm_scripts/logs/ha_smoke_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/slurm_scripts/logs/ha_smoke_%j.out
# HyperAlign training smoke: 24 steps of the real recipe (bs256/lr2e-5/frames2) + hypergraph + validation + save_best.
# Fresh output_dir; best_531 (workdir_v2full) and workdir_pretrain untouched.
echo "START=$(date +%T) [HyperAlign training smoke, 24 steps + validate]"
source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
cd /leonardo_work/IscrC_GMEG/anag0000/HyperAlign
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9877 \
  ./run.py --config ./config/gram/pretrain_cfg/hyperalign_smoke.json \
  --output_dir ./workdir_smoke_ha --checkpointing true 2>&1
echo "EXIT=$? DONE=$(date +%T)"
