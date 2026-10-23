"""Data-free tests of the number formats, the statistics helpers and the robustness guards (CPU, seconds).

    python -m pytest tests -q
"""
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cards import nvfp4 as NV  # noqa: E402
from cards.gptq import gptq_nvfp4, transform_hessian  # noqa: E402
from cards.nvfp4 import e2m1_round, qdq_blocks, qdq_blocks_search  # noqa: E402
from cards.rotation import BlockHadamard  # noqa: E402
from cards.simlinear import SimNVFP4Linear, qdq_mxfp8, rtn_weight  # noqa: E402


def _x(seed=0, rows=32, cols=256, lo=-3, hi=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g) * torch.logspace(lo, hi, cols)


def test_e2m1_round_to_nearest_even():
    g = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 6.0, -0.75, -5.0])
    assert e2m1_round(g).tolist() == [0, 0, 0.5, 1, 1, 2, 2, 4, 4, 6, 6, -1, -4]


def test_scale_search_never_worse_and_reduces_to_rtn():
    x = _x(1, cols=128, lo=-1, hi=1)
    err = lambda q: ((q - x) ** 2).reshape(32, 8, 16).sum(-1)
    plain, srch = qdq_blocks(x), qdq_blocks_search(x, mults=(1.0, 1.5))
    assert torch.all(err(srch) <= err(plain) + 1e-12)                  # per block, the search keeps the better scale
    assert float(err(srch).sum()) < float(err(plain).sum())            # and it helps somewhere on Gaussian data
    assert torch.equal(qdq_blocks_search(x, mults=(1.0,)), plain)      # one multiplier = plain RTN


def test_mxfp8_rceil_and_floor_scales():
    x = _x(2)
    xb = x.reshape(32, 8, 32)
    amax = xb.abs().amax(-1)
    for rounding, expect in (("rceil", torch.ceil(torch.log2(amax / 448.0))), ("floor", torch.floor(torch.log2(amax)) - 8)):
        y = qdq_mxfp8(x, rounding=rounding).reshape(32, 8, 32)
        scale = torch.exp2(expect)
        q = y / scale.unsqueeze(-1)
        assert torch.equal(q, q.to(torch.float8_e4m3fn).float())        # elements are E4M3 values of a 2^e scale
        assert float(q.abs().max()) <= 448.0
    # RCEIL never clips a block maximum; floor saturates block maxima whose scaled value exceeds 448
    yr = qdq_mxfp8(x, rounding="rceil").reshape(32, 8, 32).abs().amax(-1)
    assert torch.all(yr >= amax * (1 - 2 ** -4))
    m = torch.tensor([[480.0] + [1.0] * 31])                           # scaled block max in (448, 512)
    assert float(qdq_mxfp8(m, rounding="floor").max()) == 448.0
    assert float(qdq_mxfp8(m, rounding="rceil").max()) == 480.0


def test_mxfp8_warns_when_not_a_multiple_of_32():
    with pytest.warns(RuntimeWarning, match="not a multiple of 32"):
        qdq_mxfp8(torch.randn(4, 48))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qdq_mxfp8(torch.randn(4, 64))                                  # no warning on an MX layout


def test_gptq_zero_weight_gives_zero_not_nan():
    parts, pts, _ = gptq_nvfp4(torch.zeros(8, 32), torch.eye(32))
    assert torch.isfinite(parts).all() and torch.count_nonzero(parts * pts) == 0


def test_hessian_rotated_in_fp32_for_a_bf16_rotation():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(512, 256, generator=g)
    c = x.t() @ x
    r16 = BlockHadamard(256, 256, seed=5, compute_dtype=torch.bfloat16)
    r32 = BlockHadamard(256, 256, seed=5, compute_dtype=torch.float32)
    h16, h32 = transform_hessian(c, r16), transform_hessian(c, r32)
    assert torch.allclose(h16, h32, rtol=1e-6, atol=1e-6 * float(h32.abs().max()))
    assert r16.compute_dtype == torch.bfloat16                         # the layer's own rotation is untouched


def test_scale_min_is_per_layer():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(4, 64, generator=g)
    x[1:] *= 1e-5          # rows whose blocks sit far below the chunk's amax: block scale between 2^-9 and 2^-6
    w = torch.randn(16, 64, generator=g)
    p, s = rtn_weight(w)
    tiny = SimNVFP4Linear(p, s, None, rotation=None, act_bits="nvfp4")
    sub = SimNVFP4Linear(p, s, None, rotation=None, act_bits="nvfp4")
    tiny.scale_min, sub.scale_min = NV.E4M3_TINY, NV.E4M3_SUBNORMAL_MIN
    before = NV.SCALE_MIN
    y_tiny, y_sub = tiny(x), sub(x)
    assert NV.SCALE_MIN == before                                      # the module default is restored
    assert not torch.equal(y_tiny, y_sub)                              # the two floors give different activations
    NV.SCALE_MIN = NV.E4M3_SUBNORMAL_MIN                               # a later build changing the default ...
    try:
        assert torch.equal(tiny(x), y_tiny)                            # ... does not change an existing layer
    finally:
        NV.SCALE_MIN = before


def test_spearman_average_ranks_and_permutation_directions():
    np = pytest.importorskip("numpy")
    stats = pytest.importorskip("scipy.stats")
    import extended_analyses as E
    x = np.array([1, 2, 2, 3, 5, 5, 7, 8], float)
    y = np.array([8, 7, 6, 6, 4, 3, 2, 1], float)
    assert abs(E.spearman(x, y) - stats.spearmanr(x, y).statistic) < 1e-12
    _, p_less = E.perm_p(x, y, alternative="less")
    _, p_greater = E.perm_p(x, y, alternative="greater")
    _, p_two = E.perm_p(x, y, alternative="two-sided")
    assert p_less < 0.01 and p_greater > 0.99 and p_less <= p_two <= 2 * p_less + 1e-12


def test_bit_exact_vs_torchao():
    pytest.importorskip("torchao", reason="TorchAO is not installed; the bit-exact NVFP4 comparison needs it")
    r = NV.check_against_torchao()
    assert r["exact"], r
