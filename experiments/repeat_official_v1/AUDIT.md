# Repeated-seed protocol audit

Audit date: 2026-10-07 (Asia/Shanghai)

## Fixed protocol

- Conda environment: `klane_h100` (Python 3.10.20, PyTorch 2.6.0+cu124).
- Hardware observed during audit: NVIDIA H100 PCIe, driver 550.90.12.
- Training/evaluation: official `train` and official `test`, 35 epochs, batch size 24,
  Adam at 1e-4, cosine schedule, evaluation/checkpointing every 5 epochs.
- `Runner.validate()` reads `cfg.dataset.test`; `best.pth` is selected by confidence F1
  on the official test set. This set is **not** an independent validation set.
- Baseline flags: temporal/MSCE/DCRB all disabled.
- Final MS-TGMF flags: temporal/MSCE/DCRB all enabled; DCRB alpha fixed at 0.7.
- Paired seeds: 2021, 3407, 42. Only the seed and model flags differ between the
  per-model configs.

The top-level working copy has no `.git` directory, so its Git commit is unverified.
Each new run records SHA-256 fingerprints of relevant source files. The nested clean
upstream reference `MS-TGMF-release` is at commit
`5b1ea3fcbe5b2023a5a11681d5ba8b38e4a7885e`; it is not claimed as the top-level
working-copy commit.

The environment lacked the optional config formatter `yapf` and external `einops`.
Config formatting now has a no-format fallback, and the repository contains a
torch-native implementation of only the fixed rearrangement patterns used by these
models. No package was installed and no Conda environment was modified.

## Existing-run audit

| Existing run | Actual config | Checkpoint/log evidence | Comparable to this protocol? |
|---|---|---|---|
| `logs/baseline/2026-06-21-09-36-42` | seed 2021, 35 epochs, Baseline | `best.pth` exists; training test-selection log and `logs/baseline/vis/val.txt` exist | **Yes; reused, not retrained** |
| `logs/baseline60/2026-06-24-13-19-23` | seed 2021, 60 epochs, Baseline | complete checkpoints/logs | No: epoch count differs |
| `logs/MS-TGMF/2026-06-28-09-04-14` | seed 2021, 60 epochs, alpha 0.7 | complete checkpoints/logs | No: epoch count differs and temporal branch was inactive |
| `logs/proj28_temporal_ms_dcrb_alpha07_seed2021_35ep/2026-06-16-15-10-05` | seed 2021, 35 epochs, alpha 0.7 | `best.pth` exists | No: checkpoint temporal BN counters are zero; dataset supplied no temporal tensors |
| `logs/proj28_temporal_ms_dcrb_seed3407_35ep/2026-06-05-05-01-03` | seed 3407, 35 epochs, DCRB alpha omitted (code default 1.0) | `best.pth` exists | No: alpha differs and temporal branch was inactive |
| `logs/proj28_t_ms_dcrb_alpha05_seed2021_35ep/2026-06-14-12-00-56` | seed 2021, 35 epochs, alpha 0.5 | `best.pth` exists | No: alpha differs and temporal branch was inactive |
| Other `proj28_*` runs | seeds read from saved configs, 35 epochs | checkpoints/logs exist | No: ablation flags differ from Baseline and final MS-TGMF |

Directory names were not used as seed evidence; saved config contents were used.
The user-reported deleted result is not reconstructed, listed as evidence, or inserted
into the manifest.

## Temporal correction and boundary rule

Legacy dataset samples only contained `proj`; the detector therefore took its
single-frame fallback even when `use_temporal=True`. The corrected loader supplies
`proj_t`, `proj_tm1`, and `proj_tm2`. History is drawn only from the same driving
sequence and the same official split. At a split-local sequence boundary, the earliest
available frame is repeated. The launcher also refuses to complete an MS-TGMF run if
temporal BatchNorm counters remain zero.

The legacy temporal config also declared 1,024 input channels, while this configured
VitSegNet (`is_with_shared_mlp=False`) actually emits 8 channels. That mismatch was
previously hidden by the single-frame fallback. The repeated-run common config corrects
only `temporal.in_channels` to 8 and retains `hidden_channels=256`.

## Minimum missing work

Baseline seed 2021 is reused. The only training tasks are:

1. Baseline: seeds 3407 and 42.
2. Final MS-TGMF (alpha 0.7): seeds 2021, 3407, and 42.

Every launcher refuses an existing output directory, and `launch_once.sh` refuses an
existing nohup log. Full training and full evaluation are intentionally not started by
the preparation step.

## Preparation-time verification

- Python compilation and shell syntax checks passed.
- All five missing-run configs resolve to 35 epochs, batch size 24, evaluation/save
  interval 5, Adam 1e-4, cosine `T_max=4235`, official Train/Test, and alpha 0.7.
- Dataset scan passed for 7,687 Train samples and 7,695 official Test samples across
  15 sequences, with no history frame crossing sequence or split membership.
- One Train and one official Test sample loaded successfully; all four projection keys
  had shape `(3, 1152, 1152)`.
- Baseline and corrected MS-TGMF model construction passed (58,528,808 and 58,614,816
  parameters). A two-sample MS-TGMF training-mode forward pass returned a finite loss,
  and all three temporal BatchNorm counters advanced from 0 to 1.
- The summary script currently fails loudly on exactly the five missing result JSON
  files; it does not substitute zero or silently omit them.
