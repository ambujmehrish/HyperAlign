#!/bin/bash
# Create the Multimodal_hypergraph conda env in the WORK area.
# Home quota is full, so the env AND all download caches go to WORK.
# Run on a LOGIN node (compute nodes have no internet):
#
#   bash setup_env.sh
#
set -e
WORK="${HA_WORK:-/leonardo_work/AIFAC_S07_041/HyperAlign}"
ENV_PREFIX="$WORK/envs/Multimodal_hypergraph"

# find conda (same candidates as slurm_scripts/env.sh)
for _c in "${CONDA_SH:-}" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && source "$_c" && break
done
command -v conda >/dev/null || { echo "ERROR: conda not found (set CONDA_SH)"; exit 1; }

# keep every cache off the home quota
export CONDA_PKGS_DIRS="$WORK/.cache/conda_pkgs"
export PIP_CACHE_DIR="$WORK/.cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

if [ -d "$ENV_PREFIX" ]; then
  echo "env already exists at $ENV_PREFIX — activating and (re)installing packages"
else
  conda create -y -p "$ENV_PREFIX" python=3.10
fi
conda activate "$ENV_PREFIX"

# ffmpeg binary (video/audio preprocessing in utils/offline_process_data.py)
conda install -y -c conda-forge ffmpeg

# CUDA torch (Leonardo A100 -> cu121 wheels)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

HERE="$(cd "$(dirname "$0")" && pwd)"
pip install -r "$HERE/requirements.txt"

# let `conda activate Multimodal_hypergraph` work by name from anywhere
conda config --append envs_dirs "$WORK/envs" 2>/dev/null || true

echo
echo "Done. Env at: $ENV_PREFIX"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available:', torch.cuda.is_available())" || true
echo "(cuda 'available: False' is EXPECTED on a login node — GPUs exist only on compute nodes)"
echo "slurm_scripts/env.sh will pick this env up automatically."
