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
#SBATCH --job-name=ha_zseval
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/smoke_logs/zseval_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/smoke_logs/zseval_%j.out
# HyperAlign zero-shot EVAL smoke: run the benchmark_eval pipeline (make_configs -> run_eval) on the
# 24-step smoke checkpoint, ~80 clips/mode, debug QOS. Verifies the eval pipeline runs error-free.
# Args = benchmark names to run this chunk (e.g. "msrvtt vatex"). No args = all 12.
set -uo pipefail
E2E=/leonardo_work/IscrC_GMEG/anag0000/HyperAlign
EVAL=$E2E/benchmark_eval
RES=$EVAL/smoke_results
mkdir -p "$RES" "$EVAL/smoke_logs" "$EVAL/smoke_annos"
source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
# point make_configs at the smoke checkpoint (24-step trained model)
export GRAM_CKPT=$E2E/workdir_smoke_ha/ckpt/model_step_22.pt

echo "==== HA zero-shot EVAL smoke $(date +%T)  ckpt=$(basename $GRAM_CKPT)  benches='${*:-ALL}' ===="
# 1) generate the 12 zs_*.json configs against the smoke checkpoint
python3 "$EVAL/make_configs.py"
# 2) truncate each config's val (and train) annotation to ~80 clips for a fast smoke
python3 - <<'PYEOF'
import json, glob, os
N=80; E2E="/leonardo_work/IscrC_GMEG/anag0000/HyperAlign"
CFG=f"{E2E}/benchmark_eval/configs"; SM=f"{E2E}/benchmark_eval/smoke_annos"
for cf in sorted(glob.glob(f"{CFG}/zs_*.json")):
    c=json.load(open(cf)); ok=True
    for sec in ('train','val'):
        for item in c['data_cfg'][sec]:
            txt=item['txt']; full=txt if os.path.isabs(txt) else f"{E2E}/{txt}"
            try: ann=json.load(open(full))
            except Exception as e: ok=False; break
            sp=f"{SM}/{os.path.basename(cf)[:-5]}_{sec}.json"
            json.dump(ann[:N], open(sp,'w')); item['txt']=sp
    if ok: json.dump(c, open(cf,'w'), indent=1); print("  truncated", os.path.basename(cf), "->", N)
PYEOF

cd "$E2E"
FILTER="${*:-}"
FAIL=0
for cfg in "$EVAL"/configs/zs_*.json; do
  name=$(basename "$cfg" .json | sed 's/^zs_//'); bench=${name%_*}
  if [ -n "$FILTER" ] && ! echo "$FILTER" | grep -qw "$bench"; then continue; fi
  echo "===================== EVAL $name ====================="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9866 \
       "$EVAL/run_eval.py" --config "$cfg" --output_dir "$RES/out_$name" 2>&1 | tee "$RES/$name.log"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then echo "!!!!! FAILED $name rc=$rc !!!!!"; FAIL=1; else echo "----- OK $name $(date +%T) -----"; fi
done
echo "==== HA zs-eval smoke DONE $(date +%T)  FAIL=$FAIL ===="
