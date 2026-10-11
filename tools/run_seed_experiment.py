#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import re
from pathlib import Path

from repeat_utils import (
    ROOT,
    create_output_dir,
    result_payload,
    started_payload,
    write_json_new,
)

sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one non-overwriting repeated-seed training and final official-test evaluation.'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True, choices=('baseline', 'mstgmf'))
    parser.add_argument('--expected-seed', required=True, type=int)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume-from', help='Completed-epoch checkpoint; resume into a NEW output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    import torch.backends.cudnn as cudnn

    from baseline.engine.runner import Runner
    from baseline.utils.config import Config

    config_path = Path(args.config).resolve()
    cfg = Config.fromfile(str(config_path))
    if int(cfg.seed) != args.expected_seed:
        raise ValueError(
            'Actual config seed {} does not match --expected-seed {}'.format(
                cfg.seed, args.expected_seed))

    expected_flags = (False, False, False) if args.model == 'baseline' else (True, True, True)
    actual_flags = (bool(cfg.use_temporal), bool(cfg.use_ms_context), bool(cfg.use_dcrb))
    if actual_flags != expected_flags:
        raise ValueError('Unexpected model flags: {} != {}'.format(actual_flags, expected_flags))
    if float(cfg.dcrb.alpha) != 0.7:
        raise ValueError('Protocol requires fixed DCRB alpha=0.7')
    if cfg.dataset.train.split != 'train' or cfg.dataset.test.split != 'test':
        raise ValueError('Protocol requires official train/test splits')
    if int(cfg.epochs) != 35 or int(cfg.eval_ep) != 5:
        raise ValueError('Protocol requires 35 epochs and evaluation every 5 epochs')

    configured_output = (ROOT / cfg.log_dir).resolve()
    requested_output = Path(args.output_dir).resolve()
    if configured_output != requested_output and not args.resume_from:
        raise ValueError(
            'Config log_dir resolves to {}, not {}'.format(configured_output, requested_output))

    output_dir = create_output_dir(requested_output)
    shutil.copyfile(config_path, output_dir / 'config.py')
    common_path = config_path.parent / 'common.py'
    if common_path.is_file():
        shutil.copyfile(common_path, output_dir / 'common.py')
    write_json_new(
        output_dir / 'run_started.json',
        started_payload(config_path, cfg.seed, args.model, output_dir),
    )

    cfg.log_dir = str(output_dir)
    cfg.work_dirs = str(output_dir / cfg.dataset.train.type)
    cfg.gpus = 1
    cudnn.benchmark = bool(cfg.cudnn_benchmark)
    cudnn.deterministic = bool(cfg.cudnn_deterministic)

    print('Protocol:', cfg.protocol_name, flush=True)
    print('Actual seed:', cfg.seed, flush=True)
    print('Official train split:', cfg.dataset.train.split, flush=True)
    print('Checkpoint selection split:', cfg.selection_split_name, flush=True)
    print('Output:', output_dir, flush=True)

    runner = Runner(cfg)
    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        previous_dir = resume_path.parent.parent
        if requested_output == previous_dir:
            raise ValueError('Resume requires a new output directory to preserve the interrupted run')
        if not resume_path.stem.isdigit():
            raise ValueError('Resume from a numbered completed-epoch checkpoint, not best.pth')
        # Compare the saved, resolved configuration, rather than trusting names.
        previous_cfg = Config.fromfile(str(previous_dir / 'config.py'))
        for key in previous_cfg._cfg_dict:
            if key == 'log_dir':
                continue
            if previous_cfg[key] != cfg[key]:
                raise ValueError('Resume configuration differs: {}'.format(key))
        state = torch.load(str(resume_path), map_location='cpu', weights_only=False)
        runner.net.load_state_dict(state['net'], strict=True)
        runner.optimizer.load_state_dict(state['optim'])
        runner.scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['epoch']) + 1
        if start_epoch >= cfg.epochs:
            raise ValueError('Checkpoint already finished the requested training epochs')
        selection_log = previous_dir / cfg.selection_log_name
        selections = []
        if selection_log.is_file():
            for line in selection_log.read_text().splitlines():
                match = re.match(r'epoch = (\d+), ([^,]+),', line)
                if match and int(match.group(1)) < start_epoch:
                    selections.append((int(match.group(1)), float(match.group(2)), line))
        if selections:
            runner.metric = max(row[1] for row in selections)
            (output_dir / cfg.selection_log_name).write_text(
                '\n'.join(row[2] for row in selections) + '\n')
        previous_best = previous_dir / 'ckpt' / 'best.pth'
        if not previous_best.is_file():
            raise FileNotFoundError('Previous best checkpoint is required: {}'.format(previous_best))
        (output_dir / 'ckpt').mkdir()
        shutil.copyfile(previous_best, output_dir / 'ckpt' / 'best.pth')
        write_json_new(output_dir / 'resume_provenance.json', {
            'checkpoint': str(resume_path),
            'previous_output_dir': str(previous_dir),
            'start_epoch_zero_based': start_epoch,
            'rng_restored': False,
            'note': 'Legacy checkpoint lacks RNG state; interrupted and continuous runs are not bitwise identical.',
        })
        # Numbered checkpoints are written before selection evaluation. Finish
        # that evaluation if power loss occurred between these two operations.
        last_epoch = start_epoch - 1
        if start_epoch % cfg.eval_ep == 0 and not any(row[0] == last_epoch for row in selections):
            runner.validate(last_epoch)
        print('Resuming at epoch {}/{}'.format(start_epoch + 1, cfg.epochs), flush=True)
    runner.train(start_epoch=start_epoch)

    checkpoint_path = output_dir / 'ckpt' / 'best.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Training finished without best.pth')

    if args.model == 'mstgmf':
        counters = [
            module.num_batches_tracked.item()
            for module in runner.net.module.temporal_module.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        ]
        if not counters or min(counters) <= 0:
            raise RuntimeError('Temporal branch was not exercised during training')

    runner.load_ckpt(str(checkpoint_path))
    metrics = runner.eval_conditional(save_path=str(output_dir / 'official_test_metrics.txt'))
    write_json_new(
        output_dir / 'test_metrics.json',
        result_payload(
            metrics, output_dir / 'config.py', checkpoint_path, cfg.seed, args.model,
            source='new_training_and_best_checkpoint_evaluation',
        ),
    )
    write_json_new(output_dir / 'run_complete.json', {'status': 'complete'})


if __name__ == '__main__':
    main()
