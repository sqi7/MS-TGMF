# Repeated-seed training and recovery

The repeated-seed configs use 35 epochs, fixed DCRB alpha 0.7, and the official
Train/Test protocol. `Runner.validate()` selects checkpoints on the official Test
set; it is not an independent validation set. See
`experiments/repeat_official_v1/AUDIT.md` for the existing-run audit.

The supplied server launchers assume the checkout is `/root/epfs/MSTGMF`, the
dataset is `/root/epfs/data/KLane`, and Python is
`/root/epfs/miniconda3/envs/klane_h100/bin/python`. Adjust these paths for another
machine. Shell scripts may also be invoked with `bash`.

Run `bash scripts/repeat_official_v1/resume_baseline_seed3407.sh` to resume the
latest numbered checkpoint from the original seed-3407 output directory.
The script creates a new output directory and a separate nohup log. Do not run it
twice while a previous attempt is active. To resume a later interrupted attempt,
pass that attempt's saved config and checkpoint to the generic entry point:

```bash
python tools/run_seed_experiment.py \
  --config /path/to/previous_run/config.py \
  --resume-from /path/to/previous_run/ckpt/29.pth \
  --output-dir /path/to/new_run \
  --model baseline --expected-seed 3407 --gpu 0
```

Recovery requires the numbered checkpoint, previous `ckpt/best.pth`, saved
`config.py` and `common.py`, and `test_selection.txt`. Restore their directory
layout together. The checkpoint restores model weights, optimizer, scheduler and
completed epochs. A selection evaluation interrupted after saving the numbered
checkpoint is repeated before training proceeds. Legacy checkpoints do not
include random-number-generator states, so resumed and uninterrupted runs are
not guaranteed to be bitwise identical.

Git excludes datasets, model checkpoints, experiment output directories and logs.
Back these files up separately. Source code in this repository cannot restore
trained weights or deleted result logs.

After a resumed run completes, update the corresponding result path in
`experiments/repeat_official_v1/results_manifest.json` to its actual
`test_metrics.json`, then run `python tools/summarize_repeated_seeds.py`.
