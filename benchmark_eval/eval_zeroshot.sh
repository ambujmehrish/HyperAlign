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
#SBATCH --job-name=zs
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/benchmark_eval/logs/zs_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/benchmark_eval/logs/zs_%j.out
#
# Zero-shot retrieval evaluation with the hypergraph model (gate 1.0, signed refinement, graph ON).
# Regenerates the 12 zs_*.json configs (checkpoint baked into workdir_v2full), loops run_eval.py,
# then eval_summary.py -> paper-format table.
#
# Usage:  sbatch eval_zeroshot.sh            # all 12 benchmark/mode
#         sbatch eval_zeroshot.sh didemo     # one benchmark first
set -uo pipefail
E2E=/leonardo/home/userexternal/amehrish/HyperAlign
EVAL=$E2E/benchmark_eval
# EVAL_RES_DIR lets one checkpoint's results live beside another's. Without it, evaluating a
# second checkpoint silently OVERWRITES the first in eval_results/ and leaves a directory
# that mixes checkpoints -- eval_summary.py then reports a table built from two models.
RES=${EVAL_RES_DIR:-$EVAL/eval_results}
# Configs must be per-run too, not just results: make_configs.py regenerates them with the
# checkpoint baked in, so two concurrent evals sharing one config dir race and both end up
# scoring the same checkpoint. Default path is unchanged when EVAL_RES_DIR is not set.
if [ -n "${EVAL_RES_DIR:-}" ]; then CFG=${GRAM_CFG_DIR:-$RES/configs}; else CFG=${GRAM_CFG_DIR:-$EVAL/configs}; fi
export GRAM_CFG_DIR="$CFG"
mkdir -p "$CFG"
mkdir -p "$RES" "$EVAL/logs"

source "$E2E/slurm_scripts/env.sh" || exit 1   # conda + WANDB/GRAM env

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
EVAL_RES_DIR="$RES" python3 "$EVAL/eval_summary.py" 2>&1 | tee "$RES/RESULTS_zeroshot.txt"
echo "DONE $(date +%T)"
