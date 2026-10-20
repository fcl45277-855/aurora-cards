"""Synthetic fixtures live outside results/ and never authorize a GPU run."""
from pathlib import Path
import json,subprocess,sys

from crossarch_paths import ROOT
tmp=ROOT/'test_fixtures/analysis'
tmp.mkdir(parents=True,exist_ok=True)
names=['var_agg']+[f'blocks.{i}.attn' for i in range(8)]
predictor={'split':'predictor','mode':'W4A4','synthetic_only':True,'config_sha256':'0'*64,'dates':[]}
for date in ['2024-01-04','2024-01-06','2024-01-08']:
    modules={}
    for index,name in enumerate(names):
        raw={'error_sum':100.0,'flip_error_sum':90.0-10*index,'linear_error_sum':50.0,
             'query_count':100,'flip_count':10+index*10,'flip_value_gap_sum':1.0+index,
             'sharp_count':0,'key_error_sum':1.0,'key_reference_sum':100.0,
             'centered_delta_sum':1.0,'centered_logit_sum':100.0,
             'local_error_sum':1.0+index,'local_reference_sum':100.0}
        modules[name]={'key_replay':{'1.0':raw}}
    predictor['dates'].append({'date':date,'modules':modules})
path=tmp/'synthetic_predictor.json'
path.write_text(json.dumps(predictor))
frozen=tmp/'synthetic_frozen.json'
if frozen.exists():
    frozen.unlink()
subprocess.run([sys.executable,str(ROOT/'scripts/freeze_predictor.py'),str(path),'--output',str(frozen)],check=True)
assert json.loads(frozen.read_text())['synthetic_only']
for reverse in [False,True]:
    result={'split':'confirmation','mode':'W4A4','synthetic_only':True,'config_sha256':'0'*64,'dates':[]}
    for day in range(1,13):
        modules={name:{'mse_vs_reference':float((index+1 if reverse else 9-index)*(1+0.01*day)),
                       'projection_parameters':4194304} for index,name in enumerate(names)}
        result['dates'].append({'date':f'2024-03-{day:02d}','modules':modules})
    inp=tmp/f'synthetic_confirmation_{reverse}.json'
    inp.write_text(json.dumps(result))
    out=tmp/f'synthetic_analysis_{reverse}.json'
    if out.exists():
        out.unlink()
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_climax.py'),str(inp),'--frozen',str(frozen),'--output',str(out)],check=True)
    analysis=json.loads(out.read_text())
    assert analysis['synthetic_only']
    assert analysis['H1_top_quartile_supported']==(not reverse)
    assert analysis['H2_supported']==(not reverse)
    assert analysis['spearman_rho']['parameter_count'] is None
    print('PASS synthetic positive/reversed analysis',reverse,flush=True)
print('PASS: no real frozen predictor exists',not (ROOT/'metadata/frozen_predictor.json').exists(),flush=True)
