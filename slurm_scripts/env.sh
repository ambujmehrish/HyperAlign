#!/bin/bash
# Shared environment for HyperAlign jobs (sourced by every slurm script).
#
# Storage layout (home quota is 50 GB, so everything big lives elsewhere):
#   HA_CODE : the code checkout (this repo)               — home
#   HA_WORK : big folders (pretrained_weights, workdir_*, datasets, data)
#   HA_DATA : raw multimodal dataset (videos/audios/annotations)
export HA_CODE="${HA_CODE:-/leonardo/home/userexternal/amehrish/HyperAlign}"
export HA_WORK="${HA_WORK:-/leonardo_work/AIFAC_S07_041/HyperAlign}"
export HA_DATA="${HA_DATA:-/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset}"

# ---- conda ---------------------------------------------------------------
# Override with:  export CONDA_SH=/path/to/etc/profile.d/conda.sh
_ha_conda_candidates=(
  "${CONDA_SH:-}"
  "$HOME/miniconda3/etc/profile.d/conda.sh"
  "$HOME/anaconda3/etc/profile.d/conda.sh"
  "/leonardo_work/AIFAC_S07_041/miniconda3/etc/profile.d/conda.sh"
  "/leonardo_work/AIFAC_S07_041/amehrish/miniconda3/etc/profile.d/conda.sh"
  "/leonardo_work/AIFAC_S07_041/HyperAlign/miniconda3/etc/profile.d/conda.sh"
)
_ha_found=""
for _c in "${_ha_conda_candidates[@]}"; do
  if [ -n "$_c" ] && [ -f "$_c" ]; then _ha_found="$_c"; break; fi
done
if [ -z "$_ha_found" ]; then
  echo "ERROR: conda.sh not found. Set CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "$_ha_found"
conda activate "${HA_CONDA_ENV:-Multimodal_hypergraph}" || {
  echo "ERROR: could not activate conda env '${HA_CONDA_ENV:-Multimodal_hypergraph}'" >&2; exit 1; }

export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
