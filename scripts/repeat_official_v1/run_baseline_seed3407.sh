#!/usr/bin/env bash
set -euo pipefail
cd /root/epfs/MSTGMF
exec /root/epfs/miniconda3/envs/klane_h100/bin/python tools/run_seed_experiment.py \
  --config configs/repeat_official_v1/baseline_seed3407.py \
  --output-dir /root/epfs/MSTGMF/runs/repeat_official_v1/baseline/seed_3407 \
  --model baseline --expected-seed 3407 --gpu 0
