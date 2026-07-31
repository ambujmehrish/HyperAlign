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
#SBATCH --job-name=ft_smoke
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/slurm_scripts/logs/ft_smoke_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/slurm_scripts/logs/ft_smoke_%j.out
# Finetune smoke: run 15 steps of finetuning on each dataset arg, init from the pretrained best-val
# checkpoint. Verifies dataset load + finetune training runs error-free. Fresh throwaway output dir.
set -uo pipefail
H=/leonardo_work/IscrC_GMEG/anag0000/HyperAlign
INIT="$H/workdir_v2full/4model/ckpt/best_ret%tvas--msrvtt_ret_ret_area_forward.pt"
source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
cd "$H"; mkdir -p slurm_scripts/logs workdir_ftsmoke
FAIL=0
for ds in "$@"; do
  echo "===================== FT SMOKE ${ds} (15 steps) ====================="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9844 \
    ./run.py --config ./config/gram/finetune_cfg/retrieval-${ds}_ftsmoke.json \
    --output_dir ./workdir_ftsmoke/${ds} --checkpoint "$INIT" --checkpointing true 2>&1
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then echo "!!!!! FAILED ${ds} rc=$rc !!!!!"; FAIL=1; else echo "----- OK ${ds} $(date +%T) -----"; fi
done
echo "==== FT SMOKE DONE $(date +%T)  FAIL=$FAIL ===="
