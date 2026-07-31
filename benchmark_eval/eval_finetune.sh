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
#SBATCH --job-name=ft_eval
#SBATCH -o /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/logs/fteval_%j.out
#SBATCH -e /leonardo_work/IscrC_GMEG/anag0000/HyperAlign/benchmark_eval/logs/fteval_%j.out
#
# Evaluate a FINETUNED HyperAlign checkpoint (workdir/finetune_<bench>) on its benchmark.
# Same eval path as zero-shot (make_configs bakes the checkpoint, run_eval.py per mode),
# but GRAM_CKPT points to the finetuned best-val checkpoint instead of the pretrained one.
#
#   Usage:  sbatch eval_finetune.sh msrvtt      # or  didemo | activitynet | vatex
#
set -uo pipefail
BENCH=${1:?usage: sbatch eval_finetune.sh <msrvtt|didemo|activitynet|vatex>}
H=/leonardo_work/IscrC_GMEG/anag0000/HyperAlign
EVAL=$H/benchmark_eval
FTDIR=$H/workdir/finetune_${BENCH}/ckpt

# pick the finetuned best-val checkpoint (fallback: latest model_step)
FT_CKPT=$(ls -t "$FTDIR"/best_*.pt 2>/dev/null | head -1)
[ -z "$FT_CKPT" ] && FT_CKPT=$(ls -t "$FTDIR"/model_step_*.pt 2>/dev/null | head -1)
[ -z "$FT_CKPT" ] && { echo "ERROR: no finetuned checkpoint in $FTDIR — run finetune.sh $BENCH first"; exit 1; }
echo "FT eval ${BENCH} using checkpoint: $FT_CKPT"

source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
export GRAM_CKPT="$FT_CKPT"             # make_configs.py bakes THIS checkpoint into the eval configs
case "$BENCH" in *depth*) export DEPTH_EVAL=1;; esac   # depth bench -> make_configs adds the 5-modal msrvtt_depth (tvasd) config
python3 "$EVAL/make_configs.py"          # regenerate eval configs against the finetuned checkpoint

cd "$H"
RES=$EVAL/eval_results_ft/${BENCH}; mkdir -p "$RES" "$EVAL/logs"
for cfg in "$EVAL"/configs/zs_${BENCH}_*.json; do
  [ -e "$cfg" ] || continue
  name=$(basename "$cfg" .json | sed 's/^zs_//')
  echo "===================== FT-EVAL $name ====================="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
       --master_port 9897 \
       "$EVAL/run_eval.py" --config "$cfg" --output_dir "$RES/out_$name" 2>&1 | tee "$RES/$name.log"
  echo "----- done $name $(date +%T) -----"
done
echo "DONE ft-eval ${BENCH} $(date +%T)"
