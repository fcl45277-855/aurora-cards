"""ClimaX isolated-attention evaluation, gated on a complete forecast task.

No model run occurs under --check. Predictor and confirmation are separate runs.
Full-model protection-budget experiments require a separately frozen allocation.
"""
import argparse, datetime, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from model_support import ROOT,sha,prepare_compat,build_climax,load_climax
from attention_adapter import replace_attention,pooled_ratios

def gate(config):
    problems=[]
    if config.get('task')!='era5_finetuned':
        problems.append('A documented ERA5-finetuned forecast checkpoint is required')
    if not config.get('training_provenance'):
        problems.append('Checkpoint/statistics training provenance is missing')
    for name in ['checkpoint','mean','std','lat','lon']:
        item=config.get(name,{})
        path=Path(item.get('path') or '/__missing_crossarch_asset__')
        if not path.is_file():
            problems.append(name+': missing file')
        elif len(item.get('sha256') or '')!=64 or sha(path)!=item['sha256']:
            problems.append(name+': missing or mismatched SHA256')
    if not config.get('lead_hours') or config.get('lead_embedding_divisor')!=100:
        problems.append('Verified native lead and upstream hours/100 embedding are required')
    expected={'plumbing':['2024-02-02'],'predictor':['2024-01-04','2024-01-06','2024-01-08'],
              'confirmation':[f'2024-03-{d:02d}' for d in range(1,13)],
              'summer':[f'2024-07-{d:02d}' for d in range(1,9)]}
    for split,dates in expected.items():
        entries=config.get('inputs',{}).get(split,[])
        if [v.get('date') for v in entries]!=dates:
            problems.append(split+': fixed input dates missing or changed')
        for item in entries:
            path=Path(item.get('path') or '/__missing_crossarch_input__')
            if not path.is_file() or not item.get('sha256') or sha(path)!=item['sha256']:
                problems.append(split+': input file/hash missing '+item.get('date',''))
    return problems

def weighted_error(pred,reference,lat):
    weights=torch.cos(torch.deg2rad(lat)).clamp(min=0)
    weights=weights/weights.mean()
    return float(((pred-reference).double().square()*weights[None,None,:,None]).mean())

def read_input(item,config,model,device):
    with np.load(item['path'],allow_pickle=False) as data, np.load(config['mean']['path'],allow_pickle=False) as mean, np.load(config['std']['path'],allow_pickle=False) as std:
        variables=data['variables'].tolist(); outputs=data['out_variables'].tolist()
        if len(set(variables))!=len(variables) or not set(variables+outputs)<=set(model.default_vars):
            raise ValueError('Unsupported or duplicated variable names')
        lat=np.load(config['lat']['path'],allow_pickle=False)
        lon=np.load(config['lon']['path'],allow_pickle=False)
        np.testing.assert_allclose(data['lat'],lat,atol=1e-6)
        np.testing.assert_allclose(data['lon'],lon,atol=1e-6)
        start=np.datetime64(str(data['init_time'].item()).removesuffix('Z'))
        end=np.datetime64(str(data['valid_time'].item()).removesuffix('Z'))
        if (end-start)/np.timedelta64(1,'h')!=config['lead_hours'] or str(start)[:10]!=item['date'] or 'T00:00' not in str(start):
            raise ValueError('Input/target timestamps disagree with the frozen task')
        def statistics(names,store):
            values=np.array([np.asarray(store[name]).item() for name in names],dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Non-finite training statistics')
            return values[:,None,None]
        xi=statistics(variables,std); yo=statistics(outputs,std)
        if not (xi>0).all() or not (yo>0).all():
            raise ValueError('Training standard deviations must be positive')
        x=(data['x']-statistics(variables,mean))/xi
        y=(data['y']-statistics(outputs,mean))/yo
        if x.shape!=(len(variables),32,64) or y.shape!=(len(outputs),32,64) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid input/target shape or values')
        persistence=x[[variables.index(v) for v in outputs]]
        return (torch.tensor(x[None],dtype=torch.float32,device=device),
                torch.tensor(y[None],dtype=torch.float32,device=device),
                torch.tensor(persistence[None],dtype=torch.float32,device=device),
                torch.tensor(lat,dtype=torch.float32,device=device),variables,outputs)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--split',choices=['plumbing','predictor','confirmation','summer'],default='plumbing')
    parser.add_argument('--mode',choices=['W4A4','W4A32'],default='W4A4')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    problems=gate(config)
    if args.check:
        print(json.dumps({'ready_assets':not problems,'problems':problems,'gpu_run_started':False},indent=2))
        return 2 if problems else 0
    if problems:
        raise RuntimeError('Asset gate failed: '+'; '.join(problems))
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not enabled; no scientific evaluation was started')
    if args.split in ['confirmation','summer']:
        frozen_path=ROOT/'metadata/frozen_predictor.json'
        if not frozen_path.exists():
            raise RuntimeError('January predictor records must be frozen before confirmation')
        frozen=json.loads(frozen_path.read_text())
        if frozen.get('synthetic_only',False):
            raise RuntimeError('Synthetic checks cannot authorize scientific confirmation')
        if frozen['config_sha256']!=sha(args.config) or sha(frozen['predictor_path'])!=frozen['predictor_sha256']:
            raise RuntimeError('Frozen predictor/configuration changed')
    prepare_compat()
    import crossarch_paths  # noqa: F401  (puts the repository root, with cards/, on sys.path)
    from cards.nvfp4 import qdq_blocks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model=build_climax()
    load_climax(model,config['checkpoint']['path'],config['checkpoint']['sha256'])
    model=model.cuda().eval()
    modules=[('var_agg',model.var_agg,'mha')]+[(f'blocks.{i}.attn',b.attn,'timm') for i,b in enumerate(model.blocks)]
    result={'config_sha256':sha(args.config),'split':args.split,'mode':args.mode,'protocol_revision':'1.2',
            'precision':'FP32 reference/accumulation; NVFP4 RTN weights; '+('NVFP4 activations' if args.mode=='W4A4' else 'FP32 activations'),
            'script_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'dates':[]}
    with torch.inference_mode():
        for item in config['inputs'][args.split]:
            x,y,persistence,lat,variables,outputs=read_input(item,config,model,'cuda')
            lead=torch.tensor([config['lead_hours']/100],device='cuda')
            def forward():
                return model(x,None,lead,variables,outputs,None,None)[1]
            reference=forward()
            record={'date':item['date'],'reference_mse_vs_truth':weighted_error(reference,y,lat),
                    'persistence_mse_vs_truth':weighted_error(persistence,y,lat),'modules':{}}
            for name,module,kind in modules:
                # Real-input baseline agreement is a mandatory per-module gate.
                with replace_attention(module,kind,lambda a:a,False):
                    identity=forward()
                # Revision 1.2: the protocol's FP32 relative-error criterion (< 1e-5, Frobenius) plus an elementwise
                # bound loosened from atol 1e-6 to 1e-5; the unquantised replay differs only by FP32 reordering.
                diff=(identity-reference).double()
                equivalence={'global_relative':float(diff.norm()/reference.double().norm()),'max_abs':float(diff.abs().max())}
                if not (equivalence['global_relative']<1e-5 and torch.allclose(reference,identity,atol=1e-5,rtol=1e-5)):
                    raise RuntimeError('Real-input attention equivalence failed: '+name+' '+json.dumps(equivalence))
                replay=[]
                with replace_attention(module,kind,qdq_blocks,args.mode=='W4A4',replay):
                    predicted=forward()
                pooled={str(alpha):pooled_ratios([{k:v for k,v in row.items() if k!='alpha'} for row in replay if row['alpha']==alpha]) for alpha in [1.0,0.5,0.25]}
                record['modules'][name]={'mse_vs_reference':weighted_error(predicted,reference,lat),
                                         'mse_vs_truth':weighted_error(predicted,y,lat),
                                         'projection_parameters':sum(p.numel() for k,p in module.named_parameters() if k.endswith('weight')),
                                         'keys_per_query':len(variables) if kind=='mha' else model.num_patches,
                                         'head_count':module.num_heads,
                                         'identity_replay_equivalence':equivalence,
                                         'key_replay':pooled}
            result['dates'].append(record)
    target=ROOT/'results'/f'climax_{args.split}_{args.mode}.json'
    if target.exists():
        raise FileExistsError('Existing scientific result retained: '+str(target))
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Result written',target)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
