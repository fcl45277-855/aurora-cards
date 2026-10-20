"""Synthetic test of make_climax_config.py: a passed run yields a config that run_climax.gate() accepts except for the
files this test does not create, and a failed persistence gate is refused. CPU, seconds, writes only to a temp dir.

    python test_make_config.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_climax_config as mk  # noqa: E402
from train_climax_era5 import ROOT, TASK, sha, write_json  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task, run = tmp / 'task', tmp / 'run'
        task.mkdir(); run.mkdir()
        for name in ['task_plan.json', 'holdout_manifest.json', 'latitude.npy', 'longitude.npy']:
            shutil.copy(TASK / name, task / name)
        np.savez(task / 'normalize_mean.npz', a=1.0); np.savez(task / 'normalize_std.npz', a=2.0)
        write_json(task / 'training_data_manifest.json', {
            'complete': True, 'mean': {'path': str(task / 'normalize_mean.npz'), 'sha256': sha(task / 'normalize_mean.npz')},
            'std': {'path': str(task / 'normalize_std.npz'), 'sha256': sha(task / 'normalize_std.npz')}})
        (run / 'best.ckpt').write_bytes(b'synthetic')
        prov = {'task_id': 't', 'scope': 's', 'best_epoch': 3, 'training_data_manifest_sha256': 'x' * 64,
                'pretrained': {}, 'lead_hours': 6, 'task_plan_sha256': sha(task / 'task_plan.json'),
                'checkpoint': {'path': str(run / 'best.ckpt'), 'sha256': sha(run / 'best.ckpt')},
                'forecast_gate': {'passed': True, 'checks': {}}}
        write_json(run / 'provenance.json', prov)
        mk.TASK, mk.RUN = task, run
        out = tmp / 'cfg.json'
        sys.argv = ['make_climax_config.py', '--output', str(out)]
        mk.main()
        cfg = json.loads(out.read_text())
        import run_climax
        problems = run_climax.gate(cfg)
        assert problems == [], problems                       # every file and hash resolves
        prov['forecast_gate']['passed'] = False
        write_json(run / 'provenance.json', prov)
        try:
            sys.argv = ['make_climax_config.py', '--output', str(tmp / 'cfg2.json')]
            mk.main()
            raise AssertionError('failed gate was not refused')
        except RuntimeError:
            pass
        try:
            sys.argv = ['make_climax_config.py', '--output', str(out)]
            mk.main()
            raise AssertionError('existing config was overwritten')
        except FileExistsError:
            pass
    print('MAKE CONFIG TEST PASS')


if __name__ == '__main__':
    main()
