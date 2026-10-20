"""Freeze January-only predictor values before running confirmation dates."""
import argparse, datetime, json
from pathlib import Path
from model_support import ROOT,sha
from attention_adapter import pooled_ratios

parser=argparse.ArgumentParser()
parser.add_argument('predictor',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'metadata/frozen_predictor.json')
args=parser.parse_args()
result=json.loads(args.predictor.read_text())
if result['split']!='predictor' or result['mode']!='W4A4' or [r['date'] for r in result['dates']]!=['2024-01-04','2024-01-06','2024-01-08']:
    raise ValueError('Only the fixed January W4A4 predictor set can be frozen')
modules={}
for name in result['dates'][0]['modules']:
    rows=[r['modules'][name]['key_replay']['1.0'] for r in result['dates']]
    # Discard derived ratios; preserve the additive raw quantities only.
    raw=[{k:v for k,v in row.items() if k.endswith('_sum') or k.endswith('_count')} for row in rows]
    modules[name]=pooled_ratios(raw)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'synthetic_only':bool(result.get('synthetic_only',False)),
        'predictor_path':str(args.predictor.resolve()),'predictor_sha256':sha(args.predictor),
        'config_sha256':result['config_sha256'],'modules':modules,
        'err_in_flip_order':sorted(modules,key=lambda n:modules[n]['err_in_flip'] if modules[n]['err_in_flip'] is not None else -1,reverse=True)}
target=args.output
if target.exists():
    raise FileExistsError('Existing frozen predictor is retained')
target.write_text(json.dumps(record,indent=2,allow_nan=False))
print('Frozen',target)
