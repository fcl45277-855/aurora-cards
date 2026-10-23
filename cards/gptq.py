"""GPTQ for NVFP4 (block size 16 along the input dim, UE4M3 block scales, fixed FP32 per-tensor scale).

Follows Frantar et al. (GPTQ, ICLR 2023): columns are quantized left to right and the rounding error of each
column is spread over the not-yet-quantized columns with the inverse-Hessian Cholesky factor. For NVFP4 the
16-column scale block of each row is computed from the CURRENT (already error-updated) weights when the
quantizer reaches the block's first column ("groupwise GPTQ" with group size 16); the per-tensor scale is
fixed from the rotated weight's amax before GPTQ starts, as in the native NVFP4 conversion.

H must be the second-moment matrix of the layer input in the SAME space as W (i.e. after smoothing and
rotation), see `transform_hessian`.
"""
from __future__ import annotations

import copy

import torch

from .nvfp4 import BLOCK, E2M1_MAX, block_scales, e2m1_round, per_tensor_scale
from .rotation import _hadamard


def transform_hessian(c: torch.Tensor, rotation=None, smooth: torch.Tensor | None = None) -> torch.Tensor:
    """C = sum x x^T in the original input space -> the matrix for x' = Q (x / s)."""
    h = c.float()
    if smooth is not None:
        inv = 1.0 / smooth.float().to(h.device)
        h = h * inv[:, None] * inv[None, :]
    if rotation is not None:
        r32 = copy.copy(rotation)          # the Hessian is always rotated in FP32 with the exact Hadamard matrices,
        r32.compute_dtype = torch.float32  # also when the layer's activation rotation runs in BF16
        r32.ha = _hadamard(rotation.fa).to(h.device)
        r32.hb = None if rotation.hb is None else _hadamard(rotation.fb).to(h.device)
        h = r32.apply(r32.apply(h).t().contiguous())     # Q C Q^T  (C symmetric)
        h = 0.5 * (h + h.t())
    return h


@torch.no_grad()
def gptq_nvfp4(w: torch.Tensor, h: torch.Tensor, *, percdamp: float = 0.01, blocksize: int = 128,
               scale_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """w: [out, in] weight in the rotated space; h: [in, in]. Returns (block-scaled values as float32, pts, info).
    The dequantized weight is values * pts."""
    if blocksize % BLOCK:
        raise ValueError("blocksize must be a multiple of 16")
    W = w.float().clone()
    n_out, K = W.shape
    pts = per_tensor_scale(W).clamp(min=1e-30)       # an all-zero weight stays 0 instead of NaN
    H = h.float().clone().to(W.device)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(K, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    parts = torch.zeros_like(W)
    loss = torch.zeros(n_out, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        P1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        bs = None
        for i in range(i2 - i1):
            if i % BLOCK == 0:
                bs = block_scales(W1[:, i:i + BLOCK].unsqueeze(1), pts, scale_mult).squeeze(1)   # [out]
                total = bs * pts
                safe = torch.where(total == 0, torch.ones_like(total), total)
            col = W1[:, i]
            q = e2m1_round(torch.clamp(col / safe, -E2M1_MAX, E2M1_MAX))
            q = torch.where(total == 0, torch.zeros_like(q), q)
            P1[:, i] = q * bs
            deq = q * total
            d = Hinv1[i, i]
            err = (col - deq) / d
            loss += (col - deq) ** 2 / d ** 2 / 2
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        parts[:, i1:i2] = P1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return parts, pts, {"gptq_loss": float(loss.sum()), "damp": float(damp)}


def proxy_error(w: torch.Tensor, deq: torch.Tensor, h: torch.Tensor) -> float:
    """tr((W - Wq) H (W - Wq)^T): the layer-output squared error GPTQ minimises (same space as h)."""
    d = (w.float() - deq.float())
    return float(torch.einsum("oi,ij,oj->", d, h.float().to(d.device), d))
