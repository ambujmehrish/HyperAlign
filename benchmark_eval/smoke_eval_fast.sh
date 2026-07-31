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
#SBATCH --job-name=ha_zsfast
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/smoke_logs/zsfast_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/smoke_logs/zsfast_%j.out
# Load-once zero-shot eval smoke: one model load evaluates every val entry in the config.
# Arg = config basename under benchmark_eval/configs (e.g. zs_COMBINED_retrieval or zs_vggsound_tav).
set -uo pipefail
CFG=${1:?need config basename}
E2E=/leonardo_work/IscrC_GMEG/anag0000/HyperAlign
EVAL=$E2E/benchmark_eval
mkdir -p "$EVAL/smoke_logs" "$EVAL/smoke_results"
source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
cd "$E2E"
echo "==== HA zs-eval FAST smoke $(date +%T)  config=$CFG (one load, all val entries) ===="
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9855 \
     "$EVAL/run_eval.py" --config "$EVAL/configs/$CFG.json" --output_dir "$EVAL/smoke_results/out_$CFG" 2>&1
echo "==== EXIT=$? DONE=$(date +%T) ===="
