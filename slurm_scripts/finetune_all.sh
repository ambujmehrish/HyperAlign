#!/bin/bash
# Submit all 5 finetunings. The 4 base benchmarks finetune from the pretrained checkpoint and run in PARALLEL
# (separate jobs, each own 24h walltime). ft_msrvtt_depth STACKS depth on the msrvtt finetune, so
# it is submitted with a SLURM dependency (afterok) on the msrvtt job — auto-starts only after
# msrvtt finetuning finishes. LOGIN-NODE launcher — run with `bash`, NOT sbatch.
#
#   Usage:  bash finetune_all.sh
#
HERE=$(cd "$(dirname "$0")" && pwd)
echo "Submitting base finetunes (parallel):"
jm=$(sbatch  --parsable "$HERE/ft_msrvtt.sh");       echo "  ft_msrvtt       -> job $jm"
jd=$(sbatch  --parsable "$HERE/ft_didemo.sh");       echo "  ft_didemo       -> job $jd"
ja=$(sbatch  --parsable "$HERE/ft_activitynet.sh");  echo "  ft_activitynet  -> job $ja"
jv=$(sbatch  --parsable "$HERE/ft_vatex.sh");        echo "  ft_vatex        -> job $jv"
# 5th: msrvtt+depth stacks on the msrvtt finetune -> wait for it (afterok)
jdd=$(sbatch --parsable --dependency=afterok:$jm "$HERE/ft_msrvtt_depth.sh")
echo "  ft_msrvtt_depth -> job $jdd   (waits for ft_msrvtt job $jm)"
echo "Watch:  squeue -u \$USER   |  logs: slurm_scripts/logs/ft_*_<jobid>.out"
