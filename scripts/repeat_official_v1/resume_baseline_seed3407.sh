#!/usr/bin/env bash
set -euo pipefail
cd /root/epfs/MSTGMF
previous=/root/epfs/MSTGMF/runs/repeat_official_v1/baseline/seed_3407
checkpoint=$(find "$previous/ckpt" -maxdepth 1 -type f -regextype posix-extended -regex '.*/[0-9]+\.pth' | sort -V | tail -n 1)
[[ -n "$checkpoint" ]] || { echo 'No numbered checkpoint found' >&2; exit 1; }
attempt=$(date -u +%Y%m%dT%H%M%SZ)
output="${previous}_resume_${attempt}"
log="/root/epfs/MSTGMF/runs/repeat_official_v1/nohup/baseline_seed3407_resume_${attempt}.log"
echo "Resume checkpoint: $checkpoint"
echo "New output: $output"
exec bash scripts/repeat_official_v1/launch_once.sh "$log" \
  /root/epfs/miniconda3/envs/klane_h100/bin/python tools/run_seed_experiment.py \
  --config "$previous/config.py" --resume-from "$checkpoint" \
  --output-dir "$output" --model baseline --expected-seed 3407 --gpu 0
