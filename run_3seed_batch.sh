#!/bin/bash
# Runs seeds 123 and 2026 for Phases 1 (MLP only), 2, 3, 4.
# seed=42 is already done (the original single-seed results already in
# results/). Sequential, not parallel -- this box has 2 CPU cores and
# Phase 3/4's Flower simulation runs are themselves multi-process, so
# running two of these scripts at once would just contend with each other.
set -e
cd /home/claude/fl-edge-ids

for SEED in 123 2026; do
  echo "=============================================="
  echo "SEED $SEED -- Phase 1 (MLP only)"
  echo "=============================================="
  python3 src/baselines/centralized.py --seed $SEED --models MLP

  echo "=============================================="
  echo "SEED $SEED -- Phase 2 (local-only)"
  echo "=============================================="
  python3 src/baselines/local_only.py --seed $SEED

  echo "=============================================="
  echo "SEED $SEED -- Phase 3 (federated FedAvg/FedProx)"
  echo "=============================================="
  python3 src/federated/run_phase3.py --seed $SEED

  echo "=============================================="
  echo "SEED $SEED -- Phase 4 (zero-day)"
  echo "=============================================="
  python3 src/federated/run_phase4.py --seed $SEED
done

echo "=============================================="
echo "ALL SEEDS DONE"
echo "=============================================="
