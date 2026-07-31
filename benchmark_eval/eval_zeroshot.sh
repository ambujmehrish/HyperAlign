#!/bin/bash
#SBATCH -A IscrC_CASPER-A_0
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=zs
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/logs/zs_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/logs/zs_%j.out
#
# Zero-shot retrieval evaluation with the hypergraph model (gate 1.0, signed refinement, graph ON).
# Regenerates the 12 zs_*.json configs (checkpoint baked into workdir_v2full), loops run_eval.py,
# then eval_summary.py -> paper-format table.
#
# Usage:  sbatch eval_zeroshot.sh            # all 12 benchmark/mode
#         sbatch eval_zeroshot.sh didemo     # one benchmark first
set -uo pipefail
E2E=/leonardo_work/IscrC_GMEG/anag0000/HyperAlign
EVAL=$E2E/benchmark_eval
CFG=$EVAL/configs
RES=$EVAL/eval_results
mkdir -p "$RES" "$EVAL/logs"

source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver

# regenerate the 12 zero-shot configs against the best-val checkpoint (workdir_v2full)
python3 "$EVAL/make_configs.py"

cd "$E2E"                              # relative paths (datasets/..., ./config/...) resolve from repo root

FILTER="${*:-}"
echo "START $(date +%T)   filter='${FILTER:-ALL}'   [hypergraph]"

for cfg in "$CFG"/zs_*.json; do
  name=$(basename "$cfg" .json | sed 's/^zs_//')
  bench=${name%_*}
  if [ -n "$FILTER" ] && ! { echo "$FILTER" | grep -qw "$bench" || echo "$FILTER" | grep -qw "$name"; }; then continue; fi
  echo "===================== EVAL $name ====================="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
       --master_port 9899 \
       "$EVAL/run_eval.py" --config "$cfg" --output_dir "$RES/out_$name" 2>&1 | tee "$RES/$name.log"
  echo "----- done $name $(date +%T) -----"
done

echo "===================== SUMMARY ====================="
python3 "$EVAL/eval_summary.py" 2>&1 | tee "$RES/RESULTS_zeroshot.txt"
echo "DONE $(date +%T)"
