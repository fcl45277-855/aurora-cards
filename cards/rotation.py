"""Random-sign Hadamard rotations Q = H_b D applied to the input dim of a Linear: y = (W Q^T)(Q x).

block = K gives the full-width QuaRot rotation used by the existing Q0 package; block = 16/32/64/128
gives block-diagonal ("micro") rotations aligned with the NVFP4 16-element scale blocks.
H_K for K = a*b is applied as a Kronecker product H_a (x) H_b with two small matmuls, so even K = 8192
never materialises a K x K matrix. Signs are drawn exactly like QuaRot/hadamard.py (torch.Generator on CPU,
randint(0, 2) * 2 - 1), so seed-matched rotations equal the existing package's rotations.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _hadamard(n: int) -> torch.Tensor:
    if n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def _fwht_quarot(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Exact copy of QuaRot/hadamard.py right_rotate: butterfly FWHT in the input dtype, same op order, so BF16
    rounding matches the native Q0 package (a Kronecker matmul in BF16 rounds differently: 0.6% rel diff)."""
    width = x.shape[-1]
    out = (x * s.to(device=x.device, dtype=x.dtype)).reshape(-1, width)
    step = 1
    while step < width:
        out = out.reshape(-1, width // (2 * step), 2, step)
        left, right = out.unbind(dim=-2)
        out = torch.stack((left + right, left - right), dim=-2)
        step *= 2
    return out.reshape(x.shape) / math.sqrt(width)


def signs(width: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randint(0, 2, (width,), generator=g, dtype=torch.int8).mul(2).sub(1)


class BlockHadamard:
    def __init__(self, width: int, block: int | None, seed: int, device="cpu", compute_dtype=torch.float32):
        self.width, self.block = width, (block or width)
        if width % self.block:
            raise ValueError(f"block {self.block} does not divide width {width}")
        self.s = signs(width, seed).to(device)
        self.compute_dtype = compute_dtype
        b = self.block
        self.fa = min(b, 128)                       # b = fa * fb, both powers of two <= 128 (b <= 16384)
        self.fb = b // self.fa
        self.ha = _hadamard(self.fa).to(device, compute_dtype)
        self.hb = _hadamard(self.fb).to(device, compute_dtype) if self.fb > 1 else None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x @ Q^T along the last dim (Q orthogonal, symmetric blocks): sign flip then block Hadamard."""
        if self.block == self.width and self.compute_dtype == torch.bfloat16:
            return _fwht_quarot(x, self.s)                     # bit-identical to the native Q0 path
        shape, dt = x.shape, x.dtype
        v = (x.to(self.compute_dtype) * self.s.to(x.device, self.compute_dtype))
        v = v.reshape(-1, self.width // self.block, self.fb, self.fa)
        v = v @ self.ha.to(v.device)                                   # H_fa on the fast axis
        if self.hb is not None:
            v = (v.transpose(-1, -2) @ self.hb.to(v.device)).transpose(-1, -2)   # H_fb on the slow axis
        return v.reshape(shape).to(dt)

    def rotate_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.apply(w)
