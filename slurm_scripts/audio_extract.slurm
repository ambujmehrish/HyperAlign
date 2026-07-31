#!/bin/bash
#SBATCH -A IscrC_GMEG
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=aud_ex
#SBATCH -o slurm_scripts/logs/%x_%j.out
#SBATCH -e slurm_scripts/logs/%x_%j.err
# audio->wav for shard $1 of $2 (CPU-only, 32-parallel Pool inside). skip-if-exists.
source /leonardo_work/IscrC_GMEG/anag0000/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
cd /leonardo_work/IscrC_GMEG/anag0000/GRAM/code
echo "START=$(date +%T) shard=${1:-0}/${2:-2}"
python3 data/extract_audio.py ${1:-0} ${2:-2}
DS=/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset
echo "DONE=$(date +%T) total_wav=$(ls $DS/vast27m_150k/audios_wav/*.wav 2>/dev/null | wc -l)"
