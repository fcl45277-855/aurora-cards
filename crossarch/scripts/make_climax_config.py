"""Write the ClimaX evaluation config that run_climax.py's asset gate accepts, from a finished fine-tuning run.

Refuses unless the run passed its frozen forecast gate, every referenced file still matches its SHA256 and the holdout
inputs are the fixed dates of the cross-architecture protocol. Never overwrites an existing config.

    python make_climax_config.py [--output ../configs/climax_config.json]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_climax_era5 import RUN, ROOT, TASK, sha, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'configs/climax_config.json')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError('Existing config retained: ' + str(a.output))
    prov_path = RUN / 'provenance.json'
    prov = json.loads(prov_path.read_text())
    if not prov['forecast_gate']['passed']:
        raise RuntimeError('Fine-tuned ClimaX did not beat 6 h persistence; quantisation runs are not authorised')
    if sha(prov['checkpoint']['path']) != prov['checkpoint']['sha256']:
        raise RuntimeError('best.ckpt changed after training')
    if sha(TASK / 'task_plan.json') != prov['task_plan_sha256']:
        raise RuntimeError('task_plan.json changed after training')
    data = json.loads((TASK / 'training_data_manifest.json').read_text())
    holdout = json.loads((TASK / 'holdout_manifest.json').read_text())
    if not holdout.get('complete') or holdout['task_plan_sha256'] != prov['task_plan_sha256']:
        raise RuntimeError('Holdout inputs incomplete or prepared for another plan')
    template = json.loads((ROOT / 'configs/climax_config.template.json').read_text())
    inputs = {}
    for split, entries in template['inputs'].items():
        got = holdout['inputs'][split]
        if [e['date'] for e in got] != [e['date'] for e in entries]:
            raise RuntimeError(split + ': holdout dates differ from the protocol')
        for e in got:
            if sha(e['path']) != e['sha256']:
                raise RuntimeError('Holdout input changed: ' + e['path'])
        inputs[split] = [{'date': e['date'], 'path': e['path'], 'sha256': e['sha256']} for e in got]

    def item(path):
        return {'path': str(path), 'sha256': sha(path)}
    config = {
        'task': 'era5_finetuned',
        'training_provenance': {'provenance': item(prov_path), 'task_id': prov['task_id'], 'scope': prov['scope'],
                                'best_epoch': prov['best_epoch'], 'forecast_gate': prov['forecast_gate'],
                                'training_data_manifest_sha256': prov['training_data_manifest_sha256'],
                                'pretrained': prov['pretrained']},
        'lead_hours': prov['lead_hours'], 'lead_embedding_divisor': 100,
        'checkpoint': prov['checkpoint'],
        'mean': {'path': data['mean']['path'], 'sha256': data['mean']['sha256']},
        'std': {'path': data['std']['path'], 'sha256': data['std']['sha256']},
        'lat': item(TASK / 'latitude.npy'), 'lon': item(TASK / 'longitude.npy'),
        'inputs': inputs}
    for k in ('mean', 'std'):
        if sha(config[k]['path']) != config[k]['sha256']:
            raise RuntimeError(k + ' statistics changed after training')
    write_json(a.output, config)
    print('wrote', a.output)


if __name__ == '__main__':
    main()
