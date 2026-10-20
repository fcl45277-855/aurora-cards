"""Eval-only ClimaX attention replay, with explicit packed-QKV coverage.

Synthetic tests establish implementation equivalence only. Every replay returns
raw sums so dates/batches are pooled before calculating err_in_flip.
"""
import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

def diagnostic_sums(q, k, dk, v, alpha=1.0):
    q,k,dk,v=(x.float() for x in (q,k,dk,v))
    logits=q @ k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    delta=alpha*q @ dk.transpose(-1,-2)/math.sqrt(q.shape[-1])
    p=logits.softmax(-1)
    error=((logits+delta).softmax(-1)-p)@v
    linear=(p*(delta-(p*delta).sum(-1,keepdim=True)))@v
    energy=error.square().sum(-1)
    flip=logits.argmax(-1)!=(logits+delta).argmax(-1)
    winner0=logits.argmax(-1)
    winner1=(logits+delta).argmax(-1)
    vv=v.expand(*q.shape[:-2],v.shape[-2],v.shape[-1])
    gap=(vv.gather(-2,winner1.unsqueeze(-1).expand(*winner1.shape,v.shape[-1]))-
         vv.gather(-2,winner0.unsqueeze(-1).expand(*winner0.shape,v.shape[-1]))).square().sum(-1)
    return {'error_sum':float(energy.double().sum()),
            'flip_error_sum':float((energy*flip).double().sum()),
            'linear_error_sum':float(linear.double().square().sum()),
            'flip_count':int(flip.sum()),'query_count':flip.numel(),
            'flip_value_gap_sum':float((gap*flip).double().sum()),
            'sharp_count':int((p.max(-1).values>0.8).sum()),
            'key_error_sum':float((alpha*dk).double().square().sum()),
            'key_reference_sum':float(k.double().square().sum()),
            'centered_delta_sum':float((delta-delta.mean(-1,keepdim=True)).double().square().sum()),
            'centered_logit_sum':float((logits-logits.mean(-1,keepdim=True)).double().square().sum())}

def pooled_ratios(records):
    totals={k:sum(r[k] for r in records) for k in records[0]}
    totals['err_in_flip']=(totals['flip_error_sum']/totals['error_sum']
                           if totals['error_sum']>0 else None)
    totals['flip_rate']=totals['flip_count']/totals['query_count']
    def ratio(num,den):
        return num/den if den>0 else None
    totals['relative_key_error_squared']=ratio(totals.get('key_error_sum',0),totals.get('key_reference_sum',0))
    totals['G_squared']=ratio(totals['error_sum'],totals.get('linear_error_sum',0))
    logit_ratio=ratio(totals.get('centered_delta_sum',0),totals.get('centered_logit_sum',0))
    totals['A_squared']=(logit_ratio/totals['relative_key_error_squared'] if logit_ratio is not None and totals['relative_key_error_squared'] else None)
    if 'local_error_sum' in totals:
        totals['local_yerr_squared']=ratio(totals['local_error_sum'],totals['local_reference_sum'])
    return totals

def heads(x, n):
    return x.reshape(x.shape[0],x.shape[1],n,x.shape[-1]//n).transpose(1,2)

def mha_forward(module, query, key, value, quantize=lambda x:x, act=False,
                records=None, need_weights=True, average_attn_weights=True,
                key_padding_mask=None, attn_mask=None, is_causal=False):
    if module.training or not module.batch_first or not module._qkv_same_embed_dim:
        raise ValueError('Only eval, batch-first, equal-width MHA is supported')
    if key_padding_mask is not None or attn_mask is not None or is_causal or module.bias_k is not None or module.add_zero_attn:
        raise ValueError('Masked/augmented MHA requires a separately verified adapter')
    w=module.in_proj_weight
    bias=module.in_proj_bias
    d=module.embed_dim
    q0,k0,v0=[heads(F.linear(x,w[i*d:(i+1)*d],None if bias is None else bias[i*d:(i+1)*d]),module.num_heads)
              for i,x in enumerate((query,key,value))]
    qw=quantize(w)
    qa=quantize if act else lambda x:x
    q,k,v=[heads(F.linear(qa(x),qw[i*d:(i+1)*d],None if bias is None else bias[i*d:(i+1)*d]),module.num_heads)
           for i,x in enumerate((query,key,value))]
    p=(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
    output=(p@v).transpose(1,2).reshape(query.shape[0],query.shape[1],d)
    output=F.linear(qa(output),quantize(module.out_proj.weight),module.out_proj.bias)
    if records is not None:
        p0=(q0@k0.transpose(-1,-2)/math.sqrt(q0.shape[-1])).softmax(-1)
        output0=F.linear((p0@v0).transpose(1,2).reshape(query.shape[0],query.shape[1],d),module.out_proj.weight,module.out_proj.bias)
        local={'local_error_sum':float((output-output0).double().square().sum()),'local_reference_sum':float(output0.double().square().sum())}
        for alpha in (1.0,0.5,0.25):
            records.append({'alpha':alpha,**local,**diagnostic_sums(q0,k0,k-k0,v0,alpha)})
    return output, (p.mean(1) if average_attn_weights else p) if need_weights else None

def timm_forward(module, x, quantize=lambda x:x, act=False, records=None):
    if module.training:
        raise ValueError('Attention replay requires eval mode')
    # ClimaX uses q_norm/k_norm=Identity. Refuse unsupported configurations.
    if not isinstance(module.q_norm,nn.Identity) or not isinstance(module.k_norm,nn.Identity):
        raise ValueError('Non-identity Q/K normalization requires separate validation')
    if getattr(module,'norm',None) is not None and not isinstance(module.norm,nn.Identity):
        raise ValueError('Non-identity post-attention normalization is unsupported')
    qa=quantize if act else lambda x:x
    def project(inp,weight):
        packed=F.linear(inp,weight,module.qkv.bias)
        return packed.reshape(x.shape[0],x.shape[1],3,module.num_heads,-1).permute(2,0,3,1,4).unbind(0)
    q0,k0,v0=project(x,module.qkv.weight)
    q,k,v=project(qa(x),quantize(module.qkv.weight))
    p=((q*module.scale)@k.transpose(-1,-2)).softmax(-1)
    out=(p@v).transpose(1,2).reshape_as(x)
    out=F.linear(qa(out),quantize(module.proj.weight),module.proj.bias)
    if records is not None:
        p0=((q0*module.scale)@k0.transpose(-1,-2)).softmax(-1)
        out0=F.linear((p0@v0).transpose(1,2).reshape_as(x),module.proj.weight,module.proj.bias)
        local={'local_error_sum':float((out-out0).double().square().sum()),'local_reference_sum':float(out0.double().square().sum())}
        for alpha in (1.0,0.5,0.25):
            records.append({'alpha':alpha,**local,**diagnostic_sums(q0,k0,k-k0,v0,alpha)})
    return out

@contextmanager
def replace_attention(module, kind, quantize, act, records=None):
    # Restore the exact bound method, including after failed inference.
    previous=module.forward
    if kind=='mha':
        def forward(query,key,value,**kwargs):
            return mha_forward(module,query,key,value,quantize,act,records,**kwargs)
    elif kind=='timm':
        # timm 1.x Block passes attn_mask=None and is_causal=False explicitly (revision 1.2); anything else is refused.
        def forward(x,attn_mask=None,is_causal=False,**kwargs):
            if kwargs or attn_mask is not None or is_causal:
                raise ValueError('Unexpected attention arguments')
            return timm_forward(module,x,quantize,act,records)
    else:
        raise ValueError(kind)
    module.forward=forward
    try:
        yield
    finally:
        module.forward=previous

def quantize_afno_matrix(weight,quantize):
    """AFNO stores [..., input, output]; block the input axis, not output."""
    if weight.ndim!=4 or weight.shape[0]!=2:
        raise ValueError('Expected AFNO real/imag matrix pair')
    return quantize(weight.transpose(-1,-2).contiguous()).transpose(-1,-2).contiguous()
