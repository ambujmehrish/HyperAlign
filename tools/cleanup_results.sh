#!/bin/bash
# Separate results by validity so a broken number cannot re-enter a table by accident.
#
#   bash tools/cleanup_results.sh              # DRY RUN: print the plan, touch nothing
#   bash tools/cleanup_results.sh --apply      # do it
#
# Three buckets:
#
#   DELETE     — measured wrongly; the numbers are not recoverable and not worth keeping.
#   QUARANTINE — measured correctly, but under a protocol that cannot be compared with GRAM
#                (lr 2e-5 instead of the paper's 1e-4, and save_best instead of GRAM's final
#                checkpoint). Renamed with a _lr2e5 suffix, not deleted: these still support the
#                one conclusion that survives — that the hypergraph is neutral vs plain GRAM —
#                because that comparison is internal, all runs sharing the same protocol. Deleting
#                them would destroy the evidence for the only finding we have.
#   KEEP       — GRAM's own checkpoint and the VAST init, evaluated in our harness. Not trained by
#                us, so the lr/checkpoint-protocol defects do not apply.
set -uo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
H=${HA_CODE:-/leonardo/home/userexternal/amehrish/HyperAlign}
W=${HA_WORK:-/leonardo_work/AIFAC_S07_041/HyperAlign}
R=$W/benchmark_eval
STAMP=$(date +%Y%m%d)
Q=$W/DEPRECATED_lr2e5_$STAMP

say() { printf '  %-12s %s\n' "$1" "$2"; }
doit() { if [ $APPLY -eq 1 ]; then eval "$@"; else echo "      would: $*"; fi }

echo "=============================================================="
echo "  DELETE — measured wrongly"
echo "=============================================================="
# 431-clip VATEX gallery: inflated recall above the paper's FINETUNED numbers.
[ -d "$R/eval_results_v2" ] && { say DELETE "eval_results_v2 (VATEX gallery 431, not 1358)"; doit rm -rf "$R/eval_results_v2"; }
# Sweep read $EVAL/configs, which held a smoke config: 22-step checkpoint, 80-clip gallery. On top
# of that the has_audio override replaced the norm test instead of ANDing with it, so synthetic
# dropout was invisible and masked ON/OFF computed identical volumes.
[ -d "$H/benchmark_eval/missing_modality" ] && { say DELETE "benchmark_eval/missing_modality/* (smoke config + presence bug)"; doit rm -rf "$H/benchmark_eval/missing_modality"; }
# The shared config dir is the vector for that failure: whatever ran last wins. Every script that
# matters now regenerates into its own dir, so this one only exists to be picked up by mistake.
[ -d "$H/benchmark_eval/configs" ] && { say DELETE "benchmark_eval/configs/* (shared dir; smoke configs land here)"; doit rm -rf "$H/benchmark_eval/configs"; }

echo
echo "=============================================================="
echo "  QUARANTINE -> $Q"
echo "  (correct measurements, wrong protocol: lr 2e-5 + save_best)"
echo "=============================================================="
doit mkdir -p "$Q"
for d in eval_results_grambase eval_results_base1ep eval_results_base1ep_maskedvol \
         eval_results_ha1ep eval_results_ha_wraw eval_results_distill \
         eval_results_run2 eval_results_run3 \
         eval_results_gram_base_presfix eval_results_gram_base_maskedvol_presfix; do
  [ -d "$R/$d" ] && { say QUARANTINE "$d"; doit mv "$R/$d" "$Q/"; }
done
for d in "$W"/workdir_pretrain* "$W"/workdir_gram_base* "$W"/workdir_distill "$W"/workdir_v2full; do
  [ -d "$d" ] && { say QUARANTINE "$(basename "$d")"; doit mv "$d" "$Q/"; }
done

echo
echo "=============================================================="
echo "  KEEP — not trained by us, protocol defects do not apply"
echo "=============================================================="
for d in eval_results_gramofficial eval_results_gramofficial_upstreameval eval_results_vastinit; do
  [ -d "$R/$d" ] && say KEEP "$d"
done
say KEEP "pretrained_weights/ (GRAM release, VAST init, BERT)"

echo
if [ $APPLY -eq 0 ]; then
  echo "  DRY RUN — nothing changed. Re-run with --apply to execute."
else
  echo "  done. Quarantined material is in $Q (delete it yourself once the"
  echo "  paper-recipe runs have replaced the findings that rest on it)."
fi
