#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def timestamp(path):
    return Path(path).stem.rsplit('_', 1)[-1]


def sequence_from_path(path):
    return next(part for part in Path(path).parts if part.startswith('seq_'))


def split_times(dataset):
    by_sequence = {}
    for info in dataset.data_infos:
        by_sequence.setdefault(info['sequence'], set()).add(timestamp(info['pc']))
    return by_sequence


def check_dataset(dataset, expected_split):
    times = split_times(dataset)
    for index, info in enumerate(dataset.data_infos):
        if info['split'] != expected_split:
            raise AssertionError('sample {} has split {}'.format(index, info['split']))
        sequence = info['sequence']
        if sequence_from_path(info['pc']) != sequence:
            raise AssertionError('current frame sequence mismatch at {}'.format(index))
        for key in ('pc_tm1', 'pc_tm2'):
            history = info[key]
            if sequence_from_path(history) != sequence:
                raise AssertionError('{} crosses sequence at {}'.format(key, index))
            if timestamp(history) not in times[sequence]:
                raise AssertionError('{} crosses official split at {}'.format(key, index))
            if not Path(history).is_file():
                raise FileNotFoundError(history)
    return len(dataset.data_infos), len(times)


def parse_args():
    parser = argparse.ArgumentParser(description='Small static/data preflight for repeat protocol.')
    parser.add_argument(
        '--config',
        default=str(ROOT / 'configs/repeat_official_v1/mstgmf_seed2021.py'),
    )
    parser.add_argument('--load-samples', type=int, default=0, choices=(0, 1, 2))
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    from baseline.datasets import build_dataset
    from baseline.utils.config import Config

    cfg = Config.fromfile(args.config)
    cfg.is_eval_conditional = False
    train = build_dataset(cfg.dataset.train, cfg)
    test = build_dataset(cfg.dataset.test, cfg)
    train_count, train_sequences = check_dataset(train, 'train')
    test_count, test_sequences = check_dataset(test, 'test')
    print('train samples={} sequences={}'.format(train_count, train_sequences))
    print('official test samples={} sequences={}'.format(test_count, test_sequences))

    candidates = ((train, 0, 'train'), (test, 0, 'official_test'))
    for dataset, index, label in candidates[:args.load_samples]:
        sample = dataset[index]
        required = ('proj', 'proj_t', 'proj_tm1', 'proj_tm2')
        missing = [key for key in required if key not in sample]
        if missing:
            raise AssertionError('{} sample lacks {}'.format(label, missing))
        shapes = {key: tuple(sample[key].shape) for key in required}
        print('{} sample {} projections {}'.format(label, index, shapes))


if __name__ == '__main__':
    main()
