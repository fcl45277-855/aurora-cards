"""Load complete verified checkpoints, without forecast or mechanism evaluation."""
import argparse, gc, json, sys
from pathlib import Path
import torch
from model_support import ROOT,sha,prepare_compat,build_climax,load_climax
from attention_adapter import mha_forward
from run_fourcastnet import load_fcn,groups,asset_path

parser=argparse.ArgumentParser()
parser.add_argument('--model',choices=['climax','fourcastnet'],required=True)
args=parser.parse_args()
torch.set_num_threads(2)
prepare_compat()
with torch.inference_mode():
    if args.model=='climax':
        path=ROOT/'weights/climax_5.625deg.ckpt'
        expected='6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43'
        with torch.device('meta'):
            model=build_climax()
        load_climax(model,path,expected,assign=True)
        q=model.var_query.expand(3,-1,-1)
        kv=torch.randn(3,5,1024,generator=torch.Generator().manual_seed(20261020))
        reference,_=model.var_agg(q,kv,kv)
        replay,_=mha_forward(model.var_agg,q,kv,kv)
        assert torch.allclose(reference,replay,atol=1e-6,rtol=1e-5)
        names=['var_agg']+[f'blocks.{i}.attn' for i in range(8)]
        inventory={name:[{'name':k,'shape':list(v.shape),'input_axis':-1,'elements':v.numel()} for k,v in model.get_submodule(name).named_parameters() if k.endswith('weight')] for name in names}
    else:
        path=asset_path()
        model=load_fcn(path)
        inventory={name:[{'name':key,'shape':list(p.shape),'input_axis':-2 if kind=='afno' else 1 if kind=='conv' else -1,'elements':p.numel()} for key,p,kind in parameters] for name,parameters in groups(model)}
    for key,value in model.state_dict().items():
        assert bool(torch.isfinite(value).all()),key
    report={'model':args.model,'checkpoint_sha256':sha(path),'strict_loading':True,
            'tensor_count':len(model.state_dict()),'all_weights_finite':True,
            'synthetic_only':True,'gpu_used':False,'inventory':inventory}
    (ROOT/f'metadata/{args.model}_checkpoint_verified_cpu.json').write_text(json.dumps(report,indent=2))
    print('Complete checkpoint loaded strictly; finite tensors:',report['tensor_count'],'candidate groups:',len(inventory),flush=True)
