"""Diagnose run_climax.py's real-input equivalence gate: how far the identity-replaced attention (no quantisation)
is from the stock module, per module, on the plumbing date, on the GPU. Reports max absolute, elementwise allclose
and global relative (Frobenius) differences of the model output, and of the module output itself.

    python diag_equivalence.py configs/climax_config.json
"""
import json
import sys
from pathlib import Path

import torch

from crossarch_paths import ROOT
import run_climax  # noqa: E402
import attention_adapter  # noqa: E402
from contextlib import contextmanager


@contextmanager
def replace_attention(module, kind, quantize, act, records=None):
    # attention_adapter.replace_attention, but the timm wrapper accepts timm 1.x's attn_mask=None, is_causal=False
    if kind == 'mha':
        with attention_adapter.replace_attention(module, kind, quantize, act, records):
            yield
        return
    previous = module.forward

    def forward(x, attn_mask=None, is_causal=False, **kwargs):
        if kwargs or attn_mask is not None or is_causal:
            raise ValueError('Unexpected attention arguments')
        return attention_adapter.timm_forward(module, x, quantize, act, records)
    module.forward = forward
    try:
        yield
    finally:
        module.forward = previous
from model_support import prepare_compat, build_climax, load_climax  # noqa: E402


def stats(a, b):
    d = (a - b).double()
    return {'max_abs': float(d.abs().max()), 'max_ref': float(b.abs().max()),
            'global_rel': float(d.norm() / b.double().norm()),
            'allclose_1e-6_1e-5': bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    prepare_compat()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_climax()
    load_climax(model, config['checkpoint']['path'], config['checkpoint']['sha256'])
    model = model.cuda().eval()
    item = config['inputs']['plumbing'][0]
    out = {}
    with torch.inference_mode():
        x, y, p, lat, variables, outputs = run_climax.read_input(item, config, model, 'cuda')
        lead = torch.tensor([config['lead_hours'] / 100], device='cuda')
        f = lambda: model(x, None, lead, variables, outputs, None, None)[1]
        ref, ref2 = f(), f()
        out['stock_repeat'] = stats(ref2, ref)
        modules = [('var_agg', model.var_agg, 'mha')] + [(f'blocks.{i}.attn', b.attn, 'timm') for i, b in enumerate(model.blocks)]
        for name, module, kind in modules:
            captured = {}
            h = module.register_forward_hook(lambda m, i, o: captured.__setitem__('o', o[0] if isinstance(o, tuple) else o))
            f(); stock_mod = captured['o'].clone()
            with replace_attention(module, kind, lambda a: a, False):
                ident = f()
            h.remove()
            # the hook does not fire on the replaced forward; recompute the module output directly
            out[name] = {'model_output': stats(ident, ref)}
        # module-level comparison for var_agg on its real input
        q = model.var_query
        tok = []
        def grab(m, i):
            tok.append(i)
        h = model.var_agg.register_forward_pre_hook(grab)
        f(); h.remove()
        query, key, value = tok[0][:3]
        stock = model.var_agg(query, key, value)[0]
        with replace_attention(model.var_agg, 'mha', lambda a: a, False):
            mine = model.var_agg(query, key, value)[0]
        out['var_agg_module_output'] = stats(mine, stock)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
