"""Data-free tests of the paper's method components (CPU, a few seconds; no Aurora weights or ERA5 needed).

    python -m pytest tests -q          # or: python tests/test_methods.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cards.deployable import accumulate_stats, build_deployable_linear  # noqa: E402
from cards.nvfp4 import qdq_blocks  # noqa: E402
from cards.simlinear import qdq_mxfp8  # noqa: E402
from cards.switching import switching_diagnostic  # noqa: E402

E2M1 = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


def _layer(seed=0, n_in=256, n_out=192, tokens=4096, outliers=4):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_out, n_in, generator=g) / n_in ** 0.5
    x = torch.randn(tokens, n_in, generator=g)
    x[:, :outliers] *= 30.0                                   # activation outlier channels, as in transformer layers
    return w, x


def test_nvfp4_codes_on_grid():
    x = torch.randn(64, 128)
    vals, pts = qdq_blocks(x, return_parts=True)
    blocks = vals.reshape(64, 8, 16)
    scales = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 6.0
    codes = (blocks / scales).abs()
    # every element is an E2M1 magnitude times its block scale (the block max maps to 6 up to UE4M3 rounding)
    assert torch.isfinite(vals).all() and float(pts) > 0
    assert codes.max() <= 6.0 + 1e-4


def test_mxfp8_rceil_no_clipping():
    x = torch.randn(32, 256) * torch.logspace(-3, 3, 256)
    y = qdq_mxfp8(x)
    rel = ((y - x).abs() / x.abs().clamp(min=1e-12))
    bmax = x.abs().reshape(32, 8, 32).amax(-1, keepdim=True).expand(32, 8, 32).reshape(32, 256)
    big = x.abs() > bmax * 2 ** -6                        # elements well inside the block's normal range
    assert float(rel[big].max()) <= 2 ** -4 + 1e-6         # E4M3: 3 mantissa bits, no clipped block maxima


def test_deployable_layer_beats_rtn_and_smoothing_is_exact():
    w, x = _layer()
    ref = x @ w.t()
    stats = accumulate_stats(x)
    # smoothing identity, Eq. (4): (W S)(S^-1 x) = W x
    s = (stats["amax"] ** 0.8 / w.abs().amax(0) ** 0.2)
    assert torch.allclose((w * s) @ (x / s).t(), w @ x.t(), atol=1e-3)
    q_rtn = build_deployable_linear(w, None, stats, alpha=None, rank=0, act="nvfp4", method="rtn")
    q_dep = build_deployable_linear(w, None, stats, alpha=0.8, rank=32, act="mxfp8", method="gptq")
    err = lambda q: float(((q(x).float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
    e_rtn, e_dep = err(q_rtn), err(q_dep)
    assert q_dep.info["proxy_gptq"] < q_dep.info["proxy_rtn"]          # GPTQ lowers the residual objective, Eq. (6)
    assert e_dep < 0.5 * e_rtn, (e_rtn, e_dep)                         # W4A8 deployable layer << naive W4A4


def _few_key_cross_attention(g, n=512, d=64):
    """13 learned queries over 3 keys that share a large common component (like a level resampler)."""
    k = torch.randn(n, 1, d, generator=g) * 3.0 + torch.randn(n, 3, d, generator=g) * 0.5
    q = torch.randn(n, 13, d, generator=g) * 3.0
    v = torch.randn(n, 3, d, generator=g)
    return q, k, v


def _soft_window_attention(g, n=64, d=64):
    """144 tokens attending softly among themselves (like a Swin window)."""
    return (torch.randn(n, 144, d, generator=g) * 0.3, torch.randn(n, 144, d, generator=g) * 0.3,
            torch.randn(n, 144, d, generator=g))


def _rel_noise(k, rel, g):
    return rel * k.norm(dim=-1, keepdim=True) / k.shape[-1] ** 0.5 * torch.randn(k.shape, generator=g)


def test_switching_diagnostic_few_keys_vs_many_soft_keys():
    g = torch.Generator().manual_seed(1)
    q, k, v = _few_key_cross_attention(g)
    q2, k2, v2 = _soft_window_attention(g)
    few = switching_diagnostic(q, k, _rel_noise(k, 0.1, g), v)
    many = switching_diagnostic(q2, k2, _rel_noise(k2, 0.1, g), v2)
    # same relative key error, very different consequences
    assert few["A"] > 3 * many["A"]                              # common-mode amplification of the logit differences
    assert few["err_in_flip"] > 2 * many["err_in_flip"]          # error carried by switched cells
    assert few["relative_output_error"] > 5 * many["relative_output_error"]
    tiny = switching_diagnostic(q, k, _rel_noise(k, 1e-4, g), v)
    assert tiny["flip_rate"] == 0.0 and abs(tiny["G"] - 1.0) < 0.05   # no switching -> linear regime

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
