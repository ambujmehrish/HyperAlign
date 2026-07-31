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

# ---- conda installation --------------------------------------------------
# The conda install + env live in the WORK area. Override the install with:
#   export CONDA_SH=/path/to/etc/profile.d/conda.sh
_ha_conda_candidates=(
  "${CONDA_SH:-}"
  "/leonardo_work/AIFAC_S07_041/miniconda3/etc/profile.d/conda.sh"
  "/leonardo_work/AIFAC_S07_041/anaconda3/etc/profile.d/conda.sh"
  "/leonardo_work/AIFAC_S07_041/$USER/miniconda3/etc/profile.d/conda.sh"
  "/leonardo_work/AIFAC_S07_041/$USER/anaconda3/etc/profile.d/conda.sh"
  "$HA_WORK/miniconda3/etc/profile.d/conda.sh"
  "$HA_WORK/anaconda3/etc/profile.d/conda.sh"
  "$HOME/miniconda3/etc/profile.d/conda.sh"
  "$HOME/anaconda3/etc/profile.d/conda.sh"
)
_ha_found=""
for _c in "${_ha_conda_candidates[@]}"; do
  if [ -n "$_c" ] && [ -f "$_c" ]; then _ha_found="$_c"; break; fi
done
if [ -n "$_ha_found" ]; then
  source "$_ha_found"
elif ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Set CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

# ---- conda environment ---------------------------------------------------
# HA_CONDA_ENV may be an env NAME (registered with this conda install) or a
# full PATH to the env prefix (e.g. /leonardo_work/AIFAC_S07_041/envs/hyperalign).
_ha_env="${HA_CONDA_ENV:-Multimodal_hypergraph}"
if [ -d "$_ha_env" ]; then
  conda activate "$_ha_env"
elif ! conda activate "$_ha_env" 2>/dev/null; then
  # name not registered — look for an env prefix with that name in the WORK area
  _ha_env_found=""
  for _p in "/leonardo_work/AIFAC_S07_041/envs/$_ha_env" \
            "/leonardo_work/AIFAC_S07_041/conda_envs/$_ha_env" \
            "/leonardo_work/AIFAC_S07_041/$USER/envs/$_ha_env" \
            "$HA_WORK/envs/$_ha_env" \
            "$HA_WORK/conda_envs/$_ha_env"; do
    if [ -d "$_p" ]; then _ha_env_found="$_p"; break; fi
  done
  if [ -n "$_ha_env_found" ]; then
    conda activate "$_ha_env_found"
  else
    echo "ERROR: could not activate conda env '$_ha_env'." >&2
    echo "  Set HA_CONDA_ENV to your env name or its full path, e.g." >&2
    echo "    export HA_CONDA_ENV=/leonardo_work/AIFAC_S07_041/envs/<name>" >&2
    echo "  (find it with: conda env list)" >&2
    exit 1
  fi
fi
echo "conda env: $CONDA_DEFAULT_ENV ($(command -v python3))"

export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
