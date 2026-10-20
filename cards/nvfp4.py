"""Simulated NVFP4 (E2M1 values, UE4M3 scales per 16-element block, FP32 per-tensor scale).

Numerics follow TorchAO 0.18 `nvfp4_quantize` (two-level scaling branch), which is what the native
`NVFP4DynamicActivationNVFP4WeightConfig` path uses with a dynamic per-tensor scale:

    per_tensor_scale = amax(x) / (448 * 6)
    block_scale      = e4m3( clamp(amax(block) / 6 / per_tensor_scale, max=448) )
    q                = e2m1_rne( clamp(x / (block_scale * per_tensor_scale), -6, 6) )

`check_against_torchao` compares this file bit-for-bit with TorchAO on random tensors.

Separating the two scales lets a linear layer compute  y = (s_x * s_w) * (Xb @ Wb^T)  where Xb, Wb are
block-scaled E2M1 values (e2m1 * e4m3 has <= 5 significant bits, exact in BF16), so a BF16 GEMM with FP32
accumulation reproduces an NVFP4 GEMM up to accumulation order.
"""
from __future__ import annotations

import torch

E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_TINY = 2.0 ** -6          # torch.finfo(torch.float8_e4m3fn).tiny
E4M3_SUBNORMAL_MIN = 2.0 ** -9
# Floor of the UE4M3 block scale. TorchAO clamps to the smallest normal (default); recipe "scale_min": "subnormal"
# sets it to 2^-9 so blocks far below the tensor amax keep a matching scale (E4M3 subnormals are valid scales).
SCALE_MIN = E4M3_TINY
BLOCK = 16
# positive E2M1 grid and the decision thresholds between neighbours (round-to-nearest, ties to even mantissa)
_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# tie at a midpoint goes to the neighbour with even mantissa: 0 | 1.0 | 1.0 | 2.0 | 2.0 | 4.0 | 4.0
_TIE_UP = torch.tensor([False, True, False, True, False, True, False])


def e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """Round |v| <= 6 (any float tensor) to the E2M1 grid with RNE semantics; keeps the sign."""
    a = v.abs().float()
    mid = _MID.to(a.device)
    idx = torch.bucketize(a, mid)                      # number of midpoints strictly below a (right=False)
    tie = torch.isin(a, mid)
    if tie.any():                                      # exact ties: bucketize put them below; fix per table
        tidx = torch.bucketize(a, mid, right=False)
        up = _TIE_UP.to(a.device)[tidx.clamp(max=len(mid) - 1)]
        idx = torch.where(tie & up, tidx + 1, idx)
    out = _GRID.to(a.device)[idx.clamp(max=len(_GRID) - 1)]
    return torch.copysign(out, v.float())


def e2m1_round_stochastic(v: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding to the E2M1 grid: |v| <= 6 goes to the upper neighbour with probability
    (|v| - lo) / (hi - lo), else to the lower one (unbiased: E[q] = v). Keeps the sign. Candidate 11."""
    a = v.abs().float()
    grid = _GRID.to(a.device)
    idx = torch.bucketize(a, grid, right=True).clamp(1, len(grid) - 1)     # grid[idx-1] <= a < grid[idx]
    lo, hi = grid[idx - 1], grid[idx]
    p_up = ((a - lo) / (hi - lo)).clamp(0.0, 1.0)
    out = torch.where(torch.rand_like(a) < p_up, hi, lo)
    return torch.copysign(out, v.float())


def per_tensor_scale(x: torch.Tensor) -> torch.Tensor:
    return x.detach().abs().amax().float() / (E4M3_MAX * E2M1_MAX)


def block_scales(xb: torch.Tensor, pts: torch.Tensor, scale_mult: torch.Tensor | float = 1.0) -> torch.Tensor:
    """xb: [..., nblocks, 16] float32 -> UE4M3 block scales as float32 (already rounded to e4m3).
    scale_mult (broadcastable to [..., nblocks]) lets callers search non-amax scales (e.g. MSE-optimal)."""
    s = xb.abs().amax(-1) / E2M1_MAX / pts * scale_mult
    s = torch.clamp(s, min=SCALE_MIN, max=E4M3_MAX)          # TorchAO clamps to the smallest normal E4M3
    return s.to(torch.float8_e4m3fn).float()


def qdq_blocks(x: torch.Tensor, pts: torch.Tensor | None = None, scale_mult=1.0, return_parts: bool = False,
               stochastic: bool = False):
    """Quantize-dequantize along the last dim in blocks of 16. Returns float32 dequantized tensor, or
    (block-scaled e2m1 values [same shape], per-tensor scale) if return_parts (value = parts * pts).
    stochastic=True: unbiased stochastic rounding of the E2M1 codes instead of RNE (scales unchanged)."""
    shape = x.shape
    if shape[-1] % BLOCK:
        raise ValueError(f"last dim {shape[-1]} not divisible by {BLOCK}")
    xf = x.float().reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    pts = (per_tensor_scale(x) if pts is None else pts.float()).clamp(min=1e-30)   # all-zero input stays 0, no NaN
    bs = block_scales(xf, pts, scale_mult)
    # same arithmetic as TorchAO (multiply by the reciprocal of the combined scale), so ties round identically
    recip = ((1.0 / pts) / bs).unsqueeze(-1)
    rnd = e2m1_round_stochastic if stochastic else e2m1_round
    q = rnd(torch.clamp(xf * recip, -E2M1_MAX, E2M1_MAX))
    parts = (q * bs.unsqueeze(-1)).reshape(shape)
    if return_parts:
        return parts, pts
    return parts * pts


def qdq_blocks_search(x: torch.Tensor, pts: torch.Tensor | None = None, mults=(1.0, 1.5),
                      return_parts: bool = False):
    """Per-block choice among block-scale multipliers by lowest squared error.
    mults=(1.0, 1.5) is "Four Over Six" (block max mapped to 6 or to 4; Cook et al., arXiv 2512.02010);
    a wider list (e.g. 0.75..1.5) is a ScaleSweep-style search (arXiv 2606.07618)."""
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    pts = (per_tensor_scale(x) if pts is None else pts.float()).clamp(min=1e-30)   # all-zero input stays 0, no NaN
    best, best_err = None, None
    for m in mults:
        bs = block_scales(xf, pts, m)
        q = e2m1_round(torch.clamp(xf * ((1.0 / pts) / bs).unsqueeze(-1), -E2M1_MAX, E2M1_MAX))
        parts = q * bs.unsqueeze(-1)
        err = ((parts * pts - xf) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = parts, err
        else:
            better = err < best_err
            best = torch.where(better, parts, best)
            best_err = torch.where(better, err, best_err)
    best = best.reshape(shape)
    return (best, pts) if return_parts else best * pts


def scale_underflow_share(x: torch.Tensor) -> float:
    """Share of non-zero blocks whose UE4M3 block scale rounds to 0 or to the subnormal range (< 2^-6),
    a known failure of NVFP4 on narrow distributions (arXiv 2601.19026)."""
    xf = x.float().reshape(-1, BLOCK)
    pts = per_tensor_scale(x)
    s = xf.abs().amax(-1) / E2M1_MAX / pts
    nz = xf.abs().amax(-1) > 0
    bad = (s.to(torch.float8_e4m3fn).float() < 2 ** -6) & nz
    return float(bad.sum() / nz.sum().clamp(min=1))


def check_against_torchao(n: int = 4, device: str = "cpu") -> dict:
    """Bit-exact comparison with TorchAO's reference quantizer (run on the server)."""
    """Compares the quantized codes (block-scaled E2M1 values = e2m1 * e4m3 scale) with TorchAO's nvfp4_quantize."""
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize, per_tensor_amax_to_scale
    worst, n_diff = 0.0, 0
    for seed in range(n):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(64, 256, generator=g) * torch.logspace(-3, 2, 256)).to(device=device, dtype=torch.bfloat16)
        x[0, :16] = 0
        pts = per_tensor_amax_to_scale(x.abs().amax())
        sc, lp = nvfp4_quantize(x, 16, pts)
        q = f4_unpacked_to_f32(unpack_uint4(lp.view(torch.uint8))).view(64, 16, 16)
        ref = (q * sc.float().view(64, 16, 1)).view(64, 256)
        ours, _ = qdq_blocks(x, pts, return_parts=True)
        d = (ref - ours).abs()
        worst, n_diff = max(worst, d.max().item()), n_diff + int((d > 0).sum())
    return {"max_abs_diff": worst, "n_diff": n_diff, "exact": worst == 0.0}
