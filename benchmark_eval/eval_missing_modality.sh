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
#SBATCH --job-name=miss_mod
#SBATCH -o /leonardo/home/userexternal/amehrish/HyperAlign/benchmark_eval/logs/missmod_%j.out
#SBATCH -e /leonardo/home/userexternal/amehrish/HyperAlign/benchmark_eval/logs/missmod_%j.out
#
# MISSING-MODALITY ROBUSTNESS SWEEP
#
# Drops one modality from a growing fraction of GALLERY clips and measures retrieval at each rate,
# with the masked Gramian volume ON and OFF. Vanilla GRAM (masked off) sends every incomplete clip
# to volume 0 -- they all tie and become unrankable -- while the masked volume scores each clip at
# its own lower arity. The two curves are the experiment.
#
# The SAME clips are dropped in every cell (seeded permutation), so masked-on vs masked-off differ
# only in how a missing modality is handled, not in which clips lost it.
#
#   Usage:  sbatch eval_missing_modality.sh <benchmark_mode> [drop_modality] [rates...]
#   e.g.    sbatch eval_missing_modality.sh msrvtt_tva a
#           sbatch eval_missing_modality.sh vatex_tvas s 0 0.5 1.0
#
# Requires the zs_<benchmark_mode>.json config to exist (run make_configs.py first, or
# eval_zeroshot.sh which regenerates them). GRAM_CKPT selects the checkpoint as usual.
set -uo pipefail
BENCH=${1:?usage: sbatch eval_missing_modality.sh <benchmark_mode> [drop_modality] [rates...]}
DROP=${2:-a}
shift $(( $# > 1 ? 2 : 1 ))
RATES=${*:-"0 0.25 0.5 0.75 1.0"}

H=/leonardo/home/userexternal/amehrish/HyperAlign
W=/leonardo_work/AIFAC_S07_041/HyperAlign
EVAL=$H/benchmark_eval
source "$H/slurm_scripts/env.sh" || exit 1
cd "$H"

SRC="$EVAL/configs/zs_${BENCH}.json"
[ -f "$SRC" ] || { echo "ERROR: no config $SRC -- run 'python3 $EVAL/make_configs.py' first"; exit 1; }

RES=$EVAL/missing_modality/${BENCH}
mkdir -p "$RES" "$EVAL/logs"
export HA_DROP_MOD=$DROP
export HA_DROP_SEED=${HA_DROP_SEED:-1234}

# two config variants differing ONLY in masked_volume
python3 - "$SRC" "$RES" <<'PYEOF'
import json, sys
src, res = sys.argv[1], sys.argv[2]
cfg = json.load(open(src))
for flag in (True, False):
    c = json.loads(json.dumps(cfg))
    c['model_cfg']['masked_volume'] = flag
    json.dump(c, open(f"{res}/cfg_masked{'ON' if flag else 'OFF'}.json", 'w'), indent=1)
print(f"  configs written -> {res}  (ckpt: {cfg['run_cfg']['checkpoint'].split('/')[-1]})")
PYEOF

echo "==== MISSING-MODALITY SWEEP  bench=$BENCH  drop='$DROP'  rates='$RATES'  seed=$HA_DROP_SEED ===="
FAIL=0
for masked in ON OFF; do
  for rate in $RATES; do
    export HA_DROP_RATE=$rate
    tag="${BENCH}_drop${DROP}_rate${rate}_masked${masked}"
    echo "===================== $tag ====================="
    srun python3 -u -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
         --master_port 9871 \
         "$EVAL/run_eval.py" --config "$RES/cfg_masked${masked}.json" \
         --output_dir "$RES/out_$tag" 2>&1 | tee "$RES/$tag.log"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "!!!!! FAILED $tag rc=$rc !!!!!"; FAIL=1; } || echo "----- OK $tag $(date +%T) -----"
  done
done

echo "===================== SUMMARY ====================="
python3 "$EVAL/summarize_missing_modality.py" "$RES" 2>&1 | tee "$RES/RESULTS_missing_modality.txt"
echo "==== DONE $(date +%T)  FAIL=$FAIL ===="
