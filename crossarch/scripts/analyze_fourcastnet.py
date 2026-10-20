"""Descriptive AFNO damage concentration; no uniformity null or switching score."""
import argparse,json
from pathlib import Path
import numpy as np
from model_support import sha

parser=argparse.ArgumentParser()
parser.add_argument('result',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():
    raise FileExistsError('Existing analysis retained')
data=json.loads(args.result.read_text())
names=sorted(data['groups'])
damage=np.array([[date['damage'][name]['mse_vs_reference'] for name in names] for date in data['dates']])
params=np.array([data['dates'][0]['damage'][name]['quantized_parameters'] for name in names])
average=damage.mean(0)
order=np.argsort(-average)
denominator=average.sum()
report={'result_sha256':sha(args.result),'err_in_flip':None,
        'synthetic_only':bool(data.get('synthetic_only',False)),
        'groups':[{'name':names[i],'mean_isolated_damage':float(average[i]),'parameters':int(params[i]),
                   'damage_per_million_parameters':float(average[i]/params[i]*1e6)} for i in order],
        'top_k_share_of_summed_isolated_damage':{str(k):float(average[order[:k]].sum()/denominator) if denominator>0 else None for k in [1,3,5]},
        'interpretation':'Isolated damage is non-additive. These shares are descriptive, not fractions of full-model error. No absence-of-hotspots or uniformity conclusion is assumed.'}
args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('Descriptive control analysis written',args.output)
