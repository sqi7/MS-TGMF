import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(path):
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ['git', '-C', str(path), 'status', '--porcelain'], text=True
        ).strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def code_identity():
    tracked = [
        'baseline/datasets/klane.py',
        'baseline/datasets/registry.py',
        'baseline/engine/runner.py',
        'baseline/models/net/detector.py',
        'baseline/utils/config.py',
        'baseline/utils/einops_compat.py',
        'baseline/utils/metric_utils.py',
        'tools/run_seed_experiment.py',
        'tools/evaluate_checkpoint.py',
    ]
    return {
        'workspace_git': git_identity(ROOT),
        'upstream_reference_git': git_identity(ROOT / 'MS-TGMF-release'),
        'files': {
            rel: sha256_file(ROOT / rel)
            for rel in tracked
            if (ROOT / rel).is_file()
        },
    }


def create_output_dir(path):
    output_dir = Path(path).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json_new(path, payload):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')


def started_payload(config_path, seed, model, output_dir):
    config_path = Path(config_path).resolve()
    common_path = config_path.parent / 'common.py'
    payload = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': model,
        'seed': int(seed),
        'config_path': str(config_path),
        'config_sha256': sha256_file(config_path),
        'output_dir': str(Path(output_dir).resolve()),
        'protocol': 'official_train_test_repeat_v1',
        'checkpoint_selection': (
            'best official-test confidence F1 evaluated every 5 epochs; '
            'official test is not an independent validation set'
        ),
        'code_identity': code_identity(),
    }
    if common_path.is_file():
        payload['common_config_path'] = str(common_path)
        payload['common_config_sha256'] = sha256_file(common_path)
    return payload


def result_payload(metrics, config_path, checkpoint_path, seed, model, source):
    config_path = Path(config_path).resolve()
    common_path = config_path.parent / 'common.py'
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = {
        'model': model,
        'seed': int(seed),
        'source': source,
        'protocol': 'official_train_test_repeat_v1',
        'evaluation_split': 'official_test',
        'checkpoint_selection': (
            'best official-test confidence F1 evaluated every 5 epochs; '
            'official test is not an independent validation set'
        ),
        'config_path': str(config_path),
        'config_sha256': sha256_file(config_path),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'metrics_percent': {key: float(value) for key, value in metrics.items()},
    }
    if common_path.is_file():
        payload['common_config_path'] = str(common_path)
        payload['common_config_sha256'] = sha256_file(common_path)
    return payload
