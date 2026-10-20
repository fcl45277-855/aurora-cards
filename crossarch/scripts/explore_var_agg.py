"""EXPLORATORY (post hoc, after the fixed H1/H2 analysis): where does ClimaX var_agg's W4A4 damage come from?

Same model, config, March confirmation dates and NVFP4 RTN quantiser as run_climax.py, var_agg only. Variants quantise
a subset of var_agg: weights only / activations only, and the Q/K/V input projection / output projection separately.
Damage = latitude-weighted standardised MSE against the FP32 reference. Not part of the pre-registered tests.

    python explore_var_agg.py ../configs/climax_config.json
"""
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from model_support import ROOT, sha, prepare_compat, build_climax, load_climax
import run_climax
from attention_adapter import heads

VARIANTS = {   # (quantise in_proj weight, out_proj weight, Q/K/V-input activations, out_proj-input activation)
    'W4A4 all (= run_climax)': (1, 1, 1, 1),
    'weights only (W4A32)': (1, 1, 0, 0),
    'activations only (W32A4)': (0, 0, 1, 1),
    'Q/K/V in_proj only (W4A4)': (1, 0, 1, 0),
    'out_proj only (W4A4)': (0, 1, 0, 1),
    'K/V token activations only': (0, 0, 'kv', 0),
    'out_proj input activation only': (0, 0, 0, 1),
}


def mha(module, query, key, value, qdq, cfg):
    wi, wo, ai, ao = cfg
    d, n = module.embed_dim, module.num_heads
    w = qdq(module.in_proj_weight) if wi else module.in_proj_weight
    b = module.in_proj_bias
    acts = []
    for i, x in enumerate((query, key, value)):
        quant = ai == 1 or (ai == 'kv' and i > 0)
        acts.append(heads(F.linear(qdq(x) if quant else x, w[i * d:(i + 1) * d], b[i * d:(i + 1) * d]), n))
    q, k, v = acts
    p = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
    o = (p @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], d)
    ow = qdq(module.out_proj.weight) if wo else module.out_proj.weight
    return F.linear(qdq(o) if ao else o, ow, module.out_proj.bias), None


@contextmanager
def patched(module, qdq, cfg):
    prev = module.forward
    module.forward = lambda query, key, value, **kw: mha(module, query, key, value, qdq, cfg)
    try:
        yield
    finally:
        module.forward = prev


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    prepare_compat()
    sys.path.insert(0, str(ROOT / 'sources/release_cpu_check'))
    from cards.nvfp4 import qdq_blocks
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_climax()
    load_climax(model, config['checkpoint']['path'], config['checkpoint']['sha256'])
    model = model.cuda().eval()
    damage = {k: [] for k in VARIANTS}
    stats = []
    with torch.inference_mode():
        for item in config['inputs']['confirmation']:
            x, y, p, lat, variables, outputs = run_climax.read_input(item, config, model, 'cuda')
            lead = torch.tensor([config['lead_hours'] / 100], device='cuda')
            f = lambda: model(x, None, lead, variables, outputs, None, None)[1]
            ref = f()
            for name, cfg in VARIANTS.items():
                with patched(model.var_agg, qdq_blocks, cfg):
                    damage[name].append(run_climax.weighted_error(f(), ref, lat))
            # scale of the variable tokens entering var_agg (per-variable RMS over patches and channels)
            got = []
            h = model.var_agg.register_forward_pre_hook(lambda m, i: got.append(i[1]))
            f(); h.remove()
            tok = got[0]                                  # [B*L, V, D]
            rms = tok.double().pow(2).mean((0, 2)).sqrt()
            stats.append({'token_rms_max_over_min': float(rms.max() / rms.min()),
                          'token_absmax': float(tok.abs().max()), 'token_rms_median': float(rms.median())})
    out = {'exploratory': True, 'config_sha256': sha(sys.argv[1]),
           'mean_damage': {k: sum(v) / len(v) for k, v in damage.items()}, 'token_scale_by_date': stats}
    target = ROOT / 'metadata/explore_var_agg.json'
    target.write_text(json.dumps(out, indent=2))
    for k, v in out['mean_damage'].items():
        print(f'{k:34s} {v:.3e}')
    print(json.dumps(stats[0]))


if __name__ == '__main__':
    main()
