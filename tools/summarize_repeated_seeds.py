#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ('total', 'Total'),
    ('night', 'Night'),
    ('curve', 'Curve'),
    ('merging', 'Merging'),
    ('occ4', 'Occ4'),
    ('occ5', 'Occ5'),
    ('occ6', 'Occ6'),
    ('occ4_6', 'Occ4-6'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize comparable repeated-seed results.')
    parser.add_argument(
        '--manifest',
        default=str(ROOT / 'experiments/repeat_official_v1/results_manifest.json'),
    )
    parser.add_argument('--json-out', help='Optional new JSON output path (must not exist).')
    return parser.parse_args()


def mean_std(values):
    if len(values) < 2:
        raise ValueError('At least two values are required for a standard deviation')
    return statistics.mean(values), statistics.stdev(values)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_legacy_config(path, expected_seed):
    text = path.read_text(encoding='utf-8')
    required = {
        'seed': r'^seed\s*=\s*(\d+)',
        'epochs': r'^epochs\s*=\s*(\d+)',
        'use_temporal': r'^use_temporal\s*=\s*(True|False)',
        'use_ms_context': r'^use_ms_context\s*=\s*(True|False)',
        'use_dcrb': r'^use_dcrb\s*=\s*(True|False)',
    }
    values = {}
    for key, pattern in required.items():
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError('{} lacks {}'.format(path, key))
        values[key] = match.group(1)
    if int(values['seed']) != int(expected_seed):
        raise ValueError('{} has actual seed {}, expected {}'.format(
            path, values['seed'], expected_seed))
    if int(values['epochs']) != 35:
        raise ValueError('{} is not a 35-epoch run'.format(path))
    if any(values[key] != 'False' for key in ('use_temporal', 'use_ms_context', 'use_dcrb')):
        raise ValueError('{} is not a Baseline config'.format(path))


def parse_legacy_metrics(path):
    text = path.read_text(encoding='utf-8').strip()
    labels = {
        'total': 'overall',
        'night': 'night',
        'curve': 'curve',
        'merging': 'merging',
        'occ4': 'occ4',
        'occ5': 'occ5',
        'occ6': 'occ6',
        'occ4_6': 'Occ456',
    }
    metrics = {}
    for key, label in labels.items():
        pattern = r'(?:^|\s)' + re.escape(label) + r':?\s+(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if not match:
            raise ValueError('{} lacks metric {}'.format(path, label))
        metrics[key] = float(match.group(1))
    return metrics


def load_runs(manifest_path):
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    errors = []
    runs = {'baseline': {}, 'mstgmf': {}}
    for item in manifest.get('runs', []):
        model = item.get('model')
        seed = item.get('seed')
        if model not in runs:
            errors.append('unknown model {!r}'.format(model))
            continue
        if not item.get('comparable', False):
            errors.append('{} seed {} is marked incomparable'.format(model, seed))
            continue
        result_path = resolve_path(item['result'])
        if not result_path.is_file():
            errors.append('missing result: {}'.format(result_path))
            continue
        if item.get('result_format') == 'legacy_conditional_log':
            try:
                config_path = resolve_path(item['config'])
                checkpoint_path = resolve_path(item['checkpoint'])
                if not config_path.is_file() or not checkpoint_path.is_file():
                    raise FileNotFoundError('legacy config or checkpoint is missing')
                parse_legacy_config(config_path, seed)
                metrics = parse_legacy_metrics(result_path)
            except (KeyError, OSError, ValueError) as exc:
                errors.append('invalid legacy source {}: {}'.format(result_path, exc))
                continue
        else:
            with result_path.open(encoding='utf-8') as stream:
                payload = json.load(stream)
            if payload.get('model') != model or int(payload.get('seed', -1)) != int(seed):
                errors.append('identity mismatch in {}'.format(result_path))
                continue
            if payload.get('protocol') != manifest.get('protocol'):
                errors.append('protocol mismatch in {}'.format(result_path))
                continue
            if payload.get('evaluation_split') != 'official_test':
                errors.append('non-official-test result in {}'.format(result_path))
                continue
            try:
                config_path = Path(payload['config_path'])
                checkpoint_path = Path(payload['checkpoint_path'])
                if not config_path.is_file() or not checkpoint_path.is_file():
                    raise FileNotFoundError('recorded config or checkpoint is missing')
                if sha256_file(config_path) != payload.get('config_sha256'):
                    raise ValueError('config SHA-256 mismatch')
                if sha256_file(checkpoint_path) != payload.get('checkpoint_sha256'):
                    raise ValueError('checkpoint SHA-256 mismatch')
            except (KeyError, OSError, ValueError) as exc:
                errors.append('invalid provenance in {}: {}'.format(result_path, exc))
                continue
            metrics = payload.get('metrics_percent', {})
        missing = [key for key, _ in METRICS if key not in metrics]
        if missing:
            errors.append('{} lacks metrics {}'.format(result_path, ', '.join(missing)))
            continue
        if int(seed) in runs[model]:
            errors.append('duplicate {} seed {}'.format(model, seed))
            continue
        runs[model][int(seed)] = metrics

    if errors:
        raise RuntimeError('Cannot summarize:\n- ' + '\n- '.join(errors))
    baseline_seeds = set(runs['baseline'])
    mstgmf_seeds = set(runs['mstgmf'])
    if baseline_seeds != mstgmf_seeds:
        raise RuntimeError(
            'Paired seeds differ: baseline={} mstgmf={}'.format(
                sorted(baseline_seeds), sorted(mstgmf_seeds)))
    if len(baseline_seeds) < 3:
        raise RuntimeError('At least three paired comparable seeds are required')
    return runs, sorted(baseline_seeds)


def main():
    args = parse_args()
    runs, seeds = load_runs(Path(args.manifest).resolve())
    summary = {'seeds': seeds, 'metrics': {}}
    print('Paired seeds:', ', '.join(map(str, seeds)))
    print('| Metric | Baseline mean±std | MS-TGMF mean±std | Paired Δ mean±std |')
    print('|---|---:|---:|---:|')
    for key, label in METRICS:
        baseline = [float(runs['baseline'][seed][key]) for seed in seeds]
        mstgmf = [float(runs['mstgmf'][seed][key]) for seed in seeds]
        paired = [right - left for left, right in zip(baseline, mstgmf)]
        base_mean, base_std = mean_std(baseline)
        final_mean, final_std = mean_std(mstgmf)
        diff_mean, diff_std = mean_std(paired)
        summary['metrics'][key] = {
            'baseline_mean': base_mean,
            'baseline_std': base_std,
            'mstgmf_mean': final_mean,
            'mstgmf_std': final_std,
            'paired_delta_mean': diff_mean,
            'paired_delta_std': diff_std,
        }
        print('| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:+.3f}±{:.3f} |'.format(
            label, base_mean, base_std, final_mean, final_std, diff_mean, diff_std))

    if args.json_out:
        output_path = Path(args.json_out)
        with output_path.open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write('\n')


if __name__ == '__main__':
    main()
