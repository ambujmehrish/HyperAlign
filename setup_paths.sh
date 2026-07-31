#!/bin/bash
# One-time setup after cloning HyperAlign into home.
#
# Home quota is only 50 GB, so the big folders live in the project WORK area and
# the raw dataset lives on scratch. The code (python + json configs) still uses
# relative paths like ./pretrained_weights/... and datasets/annotations/... that
# expect those folders inside the repo root — this script symlinks them in
# (symlinks cost no quota) and creates the log/output directories.
#
#   Usage:  bash setup_paths.sh
set -u
CODE="${HA_CODE:-/leonardo/home/userexternal/amehrish/HyperAlign}"
WORK="${HA_WORK:-/leonardo_work/AIFAC_S07_041/HyperAlign}"
DATA="${HA_DATA:-/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset}"

echo "CODE = $CODE"
echo "WORK = $WORK"
echo "DATA = $DATA"
FAIL=0

link() {  # link <name>: $CODE/<name> -> $WORK/<name>
  local name="$1" src="$WORK/$1" dst="$CODE/$1"
  if [ -L "$dst" ]; then
    [ "$(readlink "$dst")" = "$src" ] && { echo "  ok      $name -> $src"; return; }
    rm "$dst"
  elif [ -e "$dst" ]; then
    echo "  SKIP    $name: a real file/dir already exists at $dst (move it into $WORK first)"; FAIL=1; return
  fi
  if [ ! -e "$src" ]; then
    mkdir -p "$src"     # output dirs (workdir*) may not exist yet — create in WORK
  fi
  ln -s "$src" "$dst" && echo "  linked  $name -> $src"
}

echo "Symlinking big folders into the repo root:"
for d in pretrained_weights datasets data workdir_smoke_ha workdir_ftsmoke workdir_v2full workdir_pretrain workdir; do
  link "$d"
done

echo "Creating log/result directories:"
mkdir -p "$CODE/slurm_scripts/logs" \
         "$CODE/benchmark_eval/logs" "$CODE/benchmark_eval/smoke_logs" \
         "$CODE/benchmark_eval/smoke_annos" "$CODE/benchmark_eval/smoke_results"
echo "  done"

echo "Sanity checks:"
[ -d "$DATA" ] && echo "  ok      dataset root $DATA" \
               || { echo "  MISSING dataset root $DATA"; FAIL=1; }
VAST="$WORK/pretrained_weights/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt"
if [ -f "$VAST" ]; then
  echo "  ok      VAST foundation checkpoint"
else
  echo "  MISSING VAST foundation checkpoint: $VAST"
  echo "          (pretrain configs point there; searching pretrained_weights...)"
  found=$(find "$WORK/pretrained_weights" -name 'model_step_204994.pt' 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    mkdir -p "$(dirname "$VAST")" && ln -s "$found" "$VAST" \
      && echo "          found at $found — symlinked into place"
  else
    echo "          if it lives elsewhere: mkdir -p $(dirname "$VAST") && ln -s /path/to/model_step_204994.pt $VAST"
    FAIL=1
  fi
fi
for f in "$WORK/pretrained_weights/bert/bert-base-uncased" \
         "$WORK/pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt" \
         "$WORK/datasets/annotations/msrvtt/descs_ret_test.json"; do
  [ -e "$f" ] && echo "  ok      ${f#$WORK/}" || { echo "  MISSING $f"; FAIL=1; }
done

[ $FAIL -eq 0 ] && echo "All good — submit jobs from $CODE (e.g. sbatch slurm_scripts/smoke_train.sh)" \
                || echo "Finished with warnings above — fix the MISSING items before running."
