#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path

from repeat_utils import ROOT, create_output_dir, result_payload, started_payload, write_json_new

sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate one existing checkpoint with explicit provenance and no overwrites.'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True, choices=('baseline', 'mstgmf'))
    parser.add_argument('--expected-seed', required=True, type=int)
    parser.add_argument('--gpu', default='0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch.backends.cudnn as cudnn

    from baseline.engine.runner import Runner
    from baseline.utils.config import Config

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = Config.fromfile(str(config_path))
    if int(cfg.seed) != args.expected_seed:
        raise ValueError('Config seed does not match --expected-seed')
    if cfg.dataset.test.split != 'test':
        raise ValueError('Evaluation must use the official test split')

    output_dir = create_output_dir(args.output_dir)
    shutil.copyfile(config_path, output_dir / 'config.py')
    common_path = config_path.parent / 'common.py'
    if common_path.is_file():
        shutil.copyfile(common_path, output_dir / 'common.py')
    write_json_new(
        output_dir / 'evaluation_started.json',
        started_payload(config_path, cfg.seed, args.model, output_dir),
    )

    cfg.log_dir = str(output_dir)
    cfg.work_dirs = str(output_dir / cfg.dataset.train.type)
    cfg.gpus = 1
    cudnn.benchmark = bool(cfg.cudnn_benchmark)
    cudnn.deterministic = bool(cfg.cudnn_deterministic)

    runner = Runner(cfg)
    runner.load_ckpt(str(checkpoint_path))
    metrics = runner.eval_conditional(save_path=str(output_dir / 'official_test_metrics.txt'))
    write_json_new(
        output_dir / 'test_metrics.json',
        result_payload(
            metrics, output_dir / 'config.py', checkpoint_path, cfg.seed, args.model,
            source='verified_legacy_checkpoint_reevaluation',
        ),
    )
    write_json_new(output_dir / 'evaluation_complete.json', {'status': 'complete'})


if __name__ == '__main__':
    main()
