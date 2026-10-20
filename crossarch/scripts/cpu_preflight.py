"""CPU plumbing checks. Outputs are not mechanism or forecast results."""
import json, math, sys
from pathlib import Path
import torch
from attention_adapter import (mha_forward,timm_forward,diagnostic_sums,pooled_ratios,
                               replace_attention,quantize_afno_matrix)
from model_support import ROOT,prepare_compat,build_climax,build_fcn,canonical_climax_state

torch.set_num_threads(2)
torch.manual_seed(20261020)
prepare_compat()
import crossarch_paths  # noqa: F401  (puts the repository root, with cards/, on sys.path)
from cards.nvfp4 import qdq_blocks
report={'synthetic_only':True,'gpu_used':False,'checks':{}}
def check(name,condition,detail=None):
    if not condition:
        raise AssertionError(name+' '+str(detail))
    report['checks'][name]=detail if detail is not None else True
    print('PASS',name,detail or '',flush=True)

with torch.inference_mode():
    m=build_climax(tiny=True)
    q=torch.randn(7,1,64); k=torch.randn(7,3,64); v=torch.randn(7,3,64)
    y,p=m.var_agg(q,k,v)
    z,pz=mha_forward(m.var_agg,q,k,v)
    check('packed MHA output and probability equivalence',
          torch.allclose(y,z,atol=1e-6,rtol=1e-5) and torch.allclose(p,pz,atol=1e-6,rtol=1e-5),float((y-z).abs().max()))
    a=m.blocks[0].attn
    x=torch.randn(3,32,64)
    reference=a(x)
    replay=timm_forward(a,x)
    check('timm fused-attention versus manual baseline',torch.allclose(reference,replay,atol=1e-6,rtol=1e-5),float((reference-replay).abs().max()))
    inputs=torch.randn(1,3,8,16)
    def forward():
        return m(inputs,None,torch.tensor([0.06]),m.default_vars,m.default_vars,None,None)[1]
    reference=forward()
    with replace_attention(m.var_agg,'mha',lambda t:t,False):
        replay=forward()
    check('end-to-end variable aggregation replacement is identity',torch.allclose(reference,replay,atol=1e-6,rtol=1e-5),float((reference-replay).abs().max()))
    records=[]
    with replace_attention(m.var_agg,'mha',qdq_blocks,True,records):
        quantized=forward()
    check('W4A4 packed projections execute and restore',bool(torch.isfinite(quantized).all()) and torch.equal(reference,forward()),len(records))
    noerror=diagnostic_sums(torch.randn(2,4,1,16),torch.randn(2,4,3,16),torch.zeros(2,4,3,16),torch.randn(2,4,3,16))
    check('zero replay error remains undefined',pooled_ratios([noerror])['err_in_flip'] is None)
    r1={'error_sum':2.0,'flip_error_sum':1.0,'query_count':2,'flip_count':1}
    r2={'error_sum':8.0,'flip_error_sum':0.0,'query_count':8,'flip_count':1}
    check('date pooling uses error sums, not mean ratios',pooled_ratios([r1,r2])['err_in_flip']==0.1)
    w=torch.randn(2,8,16,32)
    seen=[]
    def recorder(x):
        seen.append(tuple(x.shape)); return x
    w2=quantize_afno_matrix(w,recorder)
    check('AFNO groups along the actual input axis',seen==[(2,8,32,16)] and torch.equal(w,w2),seen)
    f=build_fcn(tiny=True)
    output=f(torch.randn(1,20,16,32))
    check('AFNO upstream forward on synthetic input',list(output.shape)==[1,20,16,32] and bool(torch.isfinite(output).all()),list(output.shape))
    from run_fourcastnet import groups,quantized
    inventory=groups(f)
    selected={name for _,parameters in inventory for name,_,_ in parameters}
    expected={name for name,p in f.named_parameters() if p.ndim in (2,4)}
    check('AFNO matrix inventory includes patch/head/spectral/MLP',selected==expected,len(selected))
    original={name:p.clone() for name,p in f.named_parameters()}
    for _,parameters in inventory:
        with quantized(parameters,qdq_blocks):
            check('AFNO quantized group finite '+parameters[0][0],all(bool(torch.isfinite(p).all()) for _,p,_ in parameters))
    check('AFNO group perturbations restore exact weights',all(torch.equal(p,original[name]) for name,p in f.named_parameters()))
    from run_climax import gate
    check('missing forecast assets fail closed',len(gate({}))>=10)
    import numpy as np
    from prepare_fcn_inputs import pressure_levels
    probe=np.linspace(200000,0,13)
    check('both stored pressure orders map to the same physical levels',
          pressure_levels(probe).index(500)==7 and pressure_levels(probe[::-1]).index(500)==5 and
          pressure_levels(probe).index(850)==10 and pressure_levels(probe[::-1]).index(850)==2)

    metadata=json.loads((ROOT/'metadata/checkpoint_metadata_local.json').read_text())
    with torch.device('meta'):
        architecture_models=[('ClimaX',build_climax()),('FourCastNet',build_fcn())]
    for name,model in architecture_models:
        shapes=(canonical_climax_state(metadata['climax_5.625deg.ckpt']['tensor_shapes']) if name=='ClimaX' else
                {k.removeprefix('module.'):v for k,v in metadata['fourcastnet_backbone.ckpt']['tensor_shapes'].items()})
        actual={k:list(v.shape) for k,v in model.state_dict().items()}
        check(name+' complete checkpoint key/shape agreement',actual==shapes,{'model_tensors':len(actual),'checkpoint_tensors':len(shapes)})
        if name=='FourCastNet':
            spectral=[k for k,v in model.named_parameters() if k.endswith(('.filter.w1','.filter.w2'))]
            check('AFNO spectral weights explicitly inventoried',len(spectral)==24,len(spectral))

(ROOT/'metadata/cpu_preflight.json').write_text(json.dumps(report,indent=2))
print('CPU PREPARATION CHECKS PASSED. No weather/quantization mechanism result was generated.',flush=True)
