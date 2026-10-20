"""Report fixed H1/H2 criteria, including unfavorable or undefined outcomes."""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from model_support import ROOT,sha

parser=argparse.ArgumentParser()
parser.add_argument('confirmation',type=Path)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--frozen',type=Path,default=ROOT/'metadata/frozen_predictor.json')
args=parser.parse_args()
if args.output.exists():
    raise FileExistsError('Existing analysis retained')
frozen=json.loads(args.frozen.read_text())
data=json.loads(args.confirmation.read_text())
if data['split']!='confirmation' or data['mode']!='W4A4' or data['config_sha256']!=frozen['config_sha256']:
    raise ValueError('Confirmation/configuration does not match the frozen January predictor')
if [r['date'] for r in data['dates']]!=[f'2024-03-{d:02d}' for d in range(1,13)]:
    raise ValueError('Confirmation dates are incomplete or changed')
names=sorted(frozen['modules'])
damage=np.array([[d['modules'][n]['mse_vs_reference'] for n in names] for d in data['dates']])
features={
    'err_in_flip':[frozen['modules'][n]['err_in_flip'] for n in names],
    'local_yerr':[frozen['modules'][n]['local_yerr_squared'] for n in names],
    'flip_rate':[frozen['modules'][n]['flip_rate'] for n in names],
    'flip_rate_times_conditional_value_gap':[frozen['modules'][n]['flip_value_gap_sum']/frozen['modules'][n]['query_count'] for n in names],
    'parameter_count':[data['dates'][0]['modules'][n]['projection_parameters'] for n in names]}
def correlation(values,target):
    if any(v is None for v in values) or len(set(values))<2 or len(set(target))<2:
        return None
    rho=float(spearmanr(values,target).statistic)
    return rho if np.isfinite(rho) else None
average=damage.mean(0)
correlations={k:correlation(v,average) for k,v in features.items()}
rank=float(rankdata(-average,method='average')[names.index('var_agg')])
rng=np.random.default_rng(20261020)
bootstrap={k:[] for k in features}
for _ in range(2000):
    # Moving, non-circular blocks of three dates; dates are the resampling units.
    starts=rng.integers(0,len(damage)-2,size=math.ceil(len(damage)/3))
    index=np.concatenate([np.arange(s,s+3) for s in starts])[:len(damage)]
    target=damage[index].mean(0)
    for name,values in features.items():
        rho=correlation(values,target)
        if rho is not None:
            bootstrap[name].append(rho)
ci={k:np.quantile(v,[0.025,0.975]).tolist() if len(v)==2000 else None for k,v in bootstrap.items()}
main=correlations['err_in_flip']
baselines=[correlations[k] for k in ['local_yerr','flip_rate','flip_rate_times_conditional_value_gap']]
report={'confirmation_sha256':sha(args.confirmation),'frozen_predictor_sha256':sha(args.frozen),
        'synthetic_only':bool(data.get('synthetic_only',False) or frozen.get('synthetic_only',False)),
        'module_order':names,'mean_damage':average.tolist(),'variable_aggregation_rank':rank,
        'H1_top_quartile_supported':rank<=math.ceil(len(names)/4),
        'spearman_rho':correlations,'rho_date_block_bootstrap_95ci':ci,
        'H2_supported':None if main is None or any(v is None for v in baselines) else main>=0.7 and all(main>v for v in baselines),
        'limitations':['Only nine attention modules; no claim of universal allocation optimality.',
                       'Bootstrap intervals condition on the frozen January predictor; they do not include predictor-set uncertainty.',
                       'Isolated-module damage is not additive and is not a causal decomposition.']}
args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('Fixed-criterion analysis written',args.output)
