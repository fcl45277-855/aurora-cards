"""The deployable quantised layer of the paper (Sect. 3.4, Eqs. 3-10, Algorithm 1) as a stand-alone function.

    y = Q_w(R) Q_a(S^-1 x) + (x A^T) B^T + b

with SmoothQuant scaling S = diag(s), s_j = amax(x_j)^alpha / max_i |W_ij|^(1-alpha); a rank-r truncated SVD
L = U_r Sigma_r V_r^T of the smoothed weight W S kept in BF16 (A = Sigma_r V_r^T S^-1, B = U_r); the residual
R = W S - L quantised to NVFP4 by GPTQ against H = S^-1 C S^-1 (C = sum x x^T over the calibration inputs);
and dynamic MXFP8 (t7_mx) or NVFP4 (t7) activation quantisation. This is the same construction that
cards/recipe.py applies to every layer of a recipe with smooth + low_rank + GPTQ and no rotation; it is exposed here
so that the method can be applied to any torch.nn.Linear without Aurora.

Usage:
    stats = None
    for x in calibration_batches:            # x: [tokens, in_features]
        stats = accumulate_stats(x, stats)
    q = build_deployable_linear(linear.weight, linear.bias, stats, alpha=0.8, rank=32, act="mxfp8")
    y = q(x)
"""
from __future__ import annotations

import torch

from .gptq import gptq_nvfp4, proxy_error, transform_hessian
from .simlinear import SimNVFP4Linear, rtn_weight


@torch.no_grad()
def accumulate_stats(x: torch.Tensor, stats: dict | None = None) -> dict:
    """Running per-channel maxima a_j = max |x_j| and second moment C = sum x x^T (FP32) of a layer's inputs."""
    x = x.reshape(-1, x.shape[-1]).float()
    if stats is None:
        stats = {"amax": torch.zeros(x.shape[1], device=x.device), "C": torch.zeros(x.shape[1], x.shape[1], device=x.device), "n": 0}
    stats["amax"] = torch.maximum(stats["amax"], x.abs().amax(0))
    stats["C"] += x.t() @ x
    stats["n"] += x.shape[0]
    return stats


@torch.no_grad()
def build_deployable_linear(weight: torch.Tensor, bias: torch.Tensor | None, stats: dict, *, alpha: float = 0.8,
                            rank: int = 32, act: str = "mxfp8", percdamp: float = 0.01,
                            act_scale_search=(1.0, 1.5), method: str = "gptq") -> SimNVFP4Linear:
    """Return the simulated deployable layer (SimNVFP4Linear) for one linear layer. act: "mxfp8" (t7_mx), "nvfp4" (t7),
    "bf16" (weight-only). method: "gptq" or "rtn" (round-to-nearest, for comparison). The returned module carries
    .info = {"proxy_gptq", "proxy_rtn"}: the residual objective of Eq. (6) for GPTQ and for RTN."""
    w = weight.detach().float()
    ax = stats["amax"].to(w.device).clamp(min=1e-5)
    aw = w.abs().amax(0).clamp(min=1e-5)
    s = (ax ** alpha / aw ** (1 - alpha)).clamp(min=1e-5) if alpha is not None else None      # Eq. (4)
    ws = w * s[None, :] if s is not None else w
    low_rank = None
    if rank:
        U, S, Vh = torch.linalg.svd(ws, full_matrices=False)                                    # Eq. (5)
        L = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        ws = ws - L
        a_mat = S[:rank, None] * Vh[:rank]
        if s is not None:
            a_mat = a_mat / s[None, :]
        low_rank = (a_mat, U[:, :rank])
    h = transform_hessian(stats["C"].to(w.device), None, s)                                      # Eq. (6)
    p_rtn, pts_rtn = rtn_weight(ws)
    info = {"proxy_rtn": proxy_error(ws, p_rtn * pts_rtn, h)}
    if method == "gptq":
        parts, pts, gi = gptq_nvfp4(ws, h, percdamp=percdamp)                                   # Eqs. (7)-(8)
        info["proxy_gptq"] = proxy_error(ws, parts * pts, h)
    else:
        parts, pts = p_rtn, pts_rtn
    q = SimNVFP4Linear(parts, pts, bias, rotation=None, smooth=s, act_bits=act, low_rank=low_rank,
                       act_scale_search=act_scale_search if act == "nvfp4" else None)
    q.info = info
    return q
