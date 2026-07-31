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
#SBATCH --job-name=ha_fteval
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/benchmark_eval/smoke_logs/fteval_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/benchmark_eval/smoke_logs/fteval_%j.out
#
# 10-mode FT-EVAL smoke (HyperAlign) — each FINETUNED checkpoint eval'd on its OWN benchmark.
# Configs in configs_ft/ already have the finetuned checkpoint baked + smoke-truncated val.
# GRAM-faithful eval (no hypergraph), per-task volume.  CHUNKABLE by bench name.
set -uo pipefail
H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
EVAL=$H/benchmark_eval
cd "$H"; mkdir -p "$EVAL/smoke_logs" "$EVAL/ft_eval_results"
source "$H/slurm_scripts/env.sh"   # conda + WANDB/GRAM env
BENCHES="${*:-msrvtt vatex didemo activitynet}"
echo "==== HA FT-EVAL SMOKE START $(date +%T)  benches='$BENCHES' ===="
FAIL=0
for bench in $BENCHES; do
  for cfg in "$EVAL"/configs_ft/zs_${bench}_*.json; do
    [ -e "$cfg" ] || continue
    name=$(basename "$cfg" .json | sed 's/^zs_//')
    echo "===================== HA FT-EVAL $name  (ckpt=$(python3 -c "import json;print(json.load(open('$cfg'))['run_cfg']['checkpoint'].split('/')[-1])")) ====================="
    srun python3 -u -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
         --master_port 9898 \
         "$EVAL/run_eval.py" --config "$cfg" --output_dir "$EVAL/ft_eval_results/out_$name" \
         2>&1 | tee "$EVAL/ft_eval_results/$name.log"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then echo "!!!!! FAILED $name rc=$rc !!!!!"; FAIL=1; else echo "----- OK $name $(date +%T) -----"; fi
  done
done
echo "==== HA FT-EVAL SMOKE DONE $(date +%T)  FAIL=$FAIL ===="
