"""Simulated NVFP4 W4A4 Linear: y = (s_x s_w) * (Xb @ Wb^T) + bias, with optional input rotation and
per-channel smoothing.

The weight is quantized once (by RTN or by a calibration method such as GPTQ, see cards/gptq.py) and stored as
BF16 block-scaled E2M1 values `wq` plus its FP32 per-tensor scale `w_pts`. Activations are quantized
dynamically per call; like the native QuaRotNVFP4Linear in the Q0 package, the flattened activation is
processed in chunks of `chunk_tokens`, each chunk getting its own dynamic per-tensor scale.

Options (all default to the native Q0 behaviour except `rot_dtype`):
  rotation       BlockHadamard or None
  smooth         per-input-channel divisor s (x / s, W * s) applied before the rotation, or None
  act_bits       "nvfp4" | "fp8" (W4A8 diagnostic: FP32 scale per 16 elements, optimistic) | "mxfp8" (MX, E8M0 RCEIL; "mxfp8_floor" = OCP floor
                 scale per 32, deployable) | "fp8_tensor" (dynamic per-tensor scale per chunk) | "fp8_token" (per-token) | "fp8_tile128" (FP32 per 1x128) | "bf16" (weight-only)
  rot_dtype      dtype of the activation rotation (native Q0 used BF16; FP32 removes that rounding)
  act_scale_mult optional per-input-block multiplier on activation block scales (scale search)
"""
from __future__ import annotations

import torch
from torch import nn

from .nvfp4 import qdq_blocks, qdq_blocks_search
from .rotation import BlockHadamard


class SimNVFP4Linear(nn.Module):
    def __init__(self, wq: torch.Tensor, w_pts: torch.Tensor, bias: torch.Tensor | None, *,
                 rotation: BlockHadamard | None, smooth: torch.Tensor | None = None, act_bits: str = "nvfp4",
                 chunk_tokens: int = 16384, rot_dtype: torch.dtype = torch.float32,
                 low_rank: tuple[torch.Tensor, torch.Tensor] | None = None, act_scale_search=None):
        super().__init__()
        self.act_scale_search = tuple(act_scale_search) if act_scale_search else None
        self.act_step_mults = None     # e.g. [1.0, 1.5]: activation block-scale multiplier chosen by rollout step (09 #8)
        # N1 scale-separated activation quantization: set lowpass_pool (e.g. 4) and, at run time, grid = (C, H, W)
        # (filled by a forward pre-hook on the enclosing Swin block, see cards/recipe.py)
        self.lowpass_pool = None
        self.grid = None
        # ablations of N1 against prior work: "mean" (N1, k x k block mean), "global" (HadaNorm-style: one mean per
        # level/window), "haar" (STaMP-style: orthonormal 2-level Haar, low band kept, detail bands NVFP4)
        self.lowpass_mode = "mean"
        self.coarse_bits = "bf16"          # precision of the kept coarse/low band: "bf16" | "fp8"
        self.out_features, self.in_features = wq.shape
        self.register_buffer("wq", wq.to(torch.bfloat16), persistent=False)
        self.register_buffer("w_pts", w_pts.float().reshape(()), persistent=False)
        self.register_buffer("bias", None if bias is None else bias.detach().clone(), persistent=False)
        self.register_buffer("smooth", None if smooth is None else smooth.float(), persistent=False)
        self.rotation, self.act_bits, self.chunk_tokens = rotation, act_bits, chunk_tokens
        if rotation is not None:
            rotation.compute_dtype = rot_dtype
            rotation.ha = rotation.ha.to(rot_dtype)
            if rotation.hb is not None:
                rotation.hb = rotation.hb.to(rot_dtype)
        # optional low-rank residual in the ORIGINAL (unrotated) input space: y += (x @ A^T) @ B^T, BF16
        if low_rank is not None:
            self.register_buffer("lr_a", low_rank[0].to(torch.bfloat16), persistent=False)
            self.register_buffer("lr_b", low_rank[1].to(torch.bfloat16), persistent=False)
        else:
            self.lr_a = self.lr_b = None
        # to_kv mechanism probes (recipe "kv_split" / "k_centre"): only output rows [a, b) carry err_alpha x the
        # quantization error, the rest is exact (FP32 weight w_exact, unquantized input); k_centre = (a, b): rows
        # computed from x - mean over the context tokens (dim -2), exact for attention keys (softmax shift invariance)
        self.w_exact = None
        self.quant_rows = None
        self.err_alpha = 1.0
        self.k_centre = None

    def extra_repr(self) -> str:
        rot = "none" if self.rotation is None else f"hadamard{self.rotation.block}"
        return f"{self.in_features}->{self.out_features}, rot={rot}, act={self.act_bits}, smooth={self.smooth is not None}"

    def _transform_input(self, x: torch.Tensor, pre_smoothed: bool = False) -> torch.Tensor:
        if self.smooth is not None and not pre_smoothed:
            x = x / self.smooth.to(x.device)
        if self.rotation is not None:
            x = self.rotation.apply(x)
        return x

    def _lowpass_split(self, flat: torch.Tensor):
        """flat [C*H*W, K] -> (coarse [C, h, w, K] block means, residual [C*H*W, K]); flat = up(coarse) + residual."""
        C, H, W = self.grid
        k = self.lowpass_pool
        v = flat.reshape(C, H, W, -1).permute(0, 3, 1, 2)                              # C, K, H, W (activation dtype)
        coarse = torch.nn.functional.avg_pool2d(v, k, ceil_mode=True, count_include_pad=False)
        resid = v - coarse.repeat_interleave(k, 2).repeat_interleave(k, 3)[:, :, :H, :W]
        resid = resid.permute(0, 2, 3, 1).reshape(-1, flat.shape[1])
        return coarse.permute(0, 2, 3, 1), resid

    def _quant_tokens(self, t: torch.Tensor) -> torch.Tensor:
        """NVFP4 quantize-dequantize rows of t [N, K] in chunks (dynamic per-tensor scale per chunk), as the
        activation path does; returns t's dtype."""
        out = torch.empty_like(t)
        for i in range(0, t.shape[0], self.chunk_tokens):
            c = t[i:i + self.chunk_tokens]
            if self.act_scale_search:
                q = qdq_blocks_search(c, mults=self.act_scale_search)
            else:
                q = qdq_blocks(c)
            out[i:i + c.shape[0]] = q.to(t.dtype)
        return out

    def _keep(self, t: torch.Tensor) -> torch.Tensor:
        """Precision of the kept coarse band (rows [N, K])."""
        return qdq_fp8_blocks(t).to(t.dtype) if self.coarse_bits == "fp8" else t

    def _haar_emulate(self, flat: torch.Tensor) -> torch.Tensor:
        """STaMP-style: 2-level orthonormal 2-D Haar over (H, W) per level/window; the LL2 band (1/16 tokens) kept
        at coarse_bits, all detail bands NVFP4; returns the reconstructed activation (to be used unquantized)."""
        C, H, W = self.grid
        K = flat.shape[1]
        v = flat.reshape(C, H, W, K).permute(0, 3, 1, 2)
        Hp, Wp = -(-H // 4) * 4, -(-W // 4) * 4
        if (Hp, Wp) != (H, W):
            v = torch.nn.functional.pad(v.float(), (0, Wp - W, 0, Hp - H), mode="replicate").to(flat.dtype)

        def fwd(z):
            a, b, c, d = z[..., 0::2, 0::2], z[..., 0::2, 1::2], z[..., 1::2, 0::2], z[..., 1::2, 1::2]
            return (a + b + c + d) / 2, ((a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2)

        def inv(ll, det):
            lh, hl, hh = det
            out = torch.empty(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2, dtype=ll.dtype, device=ll.device)
            out[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2
            out[..., 0::2, 1::2] = (ll - lh + hl - hh) / 2
            out[..., 1::2, 0::2] = (ll + lh - hl - hh) / 2
            out[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2
            return out

        def q(band, keep=False):                       # band [C, K, h, w] -> quantized along K per token
            rows = band.permute(0, 2, 3, 1).reshape(-1, K)
            rows = self._keep(rows) if keep else self._quant_tokens(rows)
            return rows.reshape(band.shape[0], band.shape[2], band.shape[3], K).permute(0, 3, 1, 2)
        ll1, d1 = fwd(v)
        ll2, d2 = fwd(ll1)
        rec = inv(inv(q(ll2, keep=True), tuple(q(t) for t in d2)), tuple(q(t) for t in d1))
        return rec[:, :, :H, :W].permute(0, 2, 3, 1).reshape(-1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k_centre is not None and x.dim() >= 3:
            a, b = self.k_centre
            y = self._forward(x)
            y[..., a:b] = self._forward(x - x.mean(-2, keepdim=True))[..., a:b]
            return y
        y = self._forward(x)
        if self.quant_rows is not None:
            a, b = self.quant_rows
            ye = (x.float() @ self.w_exact.t()).to(y.dtype)
            if self.bias is not None:
                ye = ye + self.bias.to(y.dtype)
            out = ye.clone()
            out[..., a:b] = ye[..., a:b] + self.err_alpha * (y[..., a:b] - ye[..., a:b])
            return out
        return y

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        orig = flat
        coarse_y = None
        pre_smoothed = False
        act_now = self.act_bits
        if self.lowpass_pool and self.grid is not None and self.grid[0] == -1 and flat.shape[0] % (self.grid[1] * self.grid[2]) == 0:
            self.grid = (flat.shape[0] // (self.grid[1] * self.grid[2]), self.grid[1], self.grid[2])
            fixed_window = True
        else:
            fixed_window = False
        if self.lowpass_pool and self.grid is not None and self.grid[0] > 0                 and flat.shape[0] == self.grid[0] * self.grid[1] * self.grid[2]                 and self.rotation is None:
            if self.smooth is not None:                       # smoothing is a per-channel scale: split x / s
                flat = flat / self.smooth.to(flat.device, flat.dtype)
                pre_smoothed = True
            H, W = self.grid[1], self.grid[2]
            if self.lowpass_mode == "haar":
                flat = self._haar_emulate(flat)
                act_now = "bf16"                          # already quantized band-wise
            elif self.lowpass_mode == "global":
                C = self.grid[0]
                v = flat.reshape(C, H * W, -1)
                mean = v.float().mean(1, keepdim=True).to(flat.dtype)          # [C, 1, K]
                cy = (self._keep(mean.reshape(C, -1)).to(torch.bfloat16) @ self.wq.t()).float() * self.w_pts
                coarse_y = cy[:, None, :].expand(C, H * W, -1).reshape(-1, self.out_features)
                flat = (v - mean).reshape(-1, flat.shape[1])
            else:
                coarse, resid = self._lowpass_split(flat)
                C, h, w, _ = coarse.shape
                cin = self._keep(coarse.reshape(-1, self.in_features))
                cy = (cin.to(torch.bfloat16) @ self.wq.t()).float() * self.w_pts
                cy = cy.reshape(C, h, w, -1).permute(0, 3, 1, 2)
                k = self.lowpass_pool
                coarse_y = cy.repeat_interleave(k, 2).repeat_interleave(k, 3)[:, :, :H, :W].permute(0, 2, 3, 1).reshape(-1, self.out_features)
                flat = resid
        if fixed_window:
            self.grid = (-1, self.grid[1], self.grid[2])          # window grid: C is re-inferred on every call
        out = torch.empty(flat.shape[0], self.out_features, device=x.device, dtype=x.dtype)
        for i in range(0, flat.shape[0], self.chunk_tokens):
            blk = self._transform_input(flat[i:i + self.chunk_tokens], pre_smoothed)
            if act_now == "fp8":
                xq = qdq_fp8_blocks(blk)
                y = (xq.to(torch.bfloat16) @ self.wq.t()).float() * self.w_pts
            elif act_now in ("mxfp8", "mxfp8_floor"):     # MXFP8 E4M3: 32-blocks, E8M0 (power-of-two) scale
                xq = qdq_mxfp8(blk, rounding="floor" if act_now == "mxfp8_floor" else "rceil")   # exact in BF16
                y = (xq.to(torch.bfloat16) @ self.wq.t()).float() * self.w_pts
            elif act_now == "fp8_token":                  # dynamic per-token (row) E4M3 scale (vLLM-style)
                q8, s8 = fp8_token_parts(blk)
                y = (q8.to(torch.bfloat16) @ self.wq.t()).float() * (s8 * self.w_pts)
            elif act_now == "fp8_tile128":                # FP32 scale per 1x128 tile (DeepSeek-V3-style)
                xq = qdq_fp8_blocks(blk, block=128)
                y = (xq.to(torch.bfloat16) @ self.wq.t()).float() * self.w_pts
            elif act_now == "fp8_tensor":                 # dynamic per-tensor E4M3 scale per chunk
                q8, s8 = fp8_tensor_parts(blk)
                y = (q8.to(torch.bfloat16) @ self.wq.t()).float() * (s8 * self.w_pts)
            elif act_now == "nvfp4_sr":                   # NVFP4 with stochastic rounding of the codes (candidate 11)
                xb, x_pts = qdq_blocks(blk, return_parts=True, stochastic=True)
                y = (xb.to(torch.bfloat16) @ self.wq.t()).float() * (x_pts * self.w_pts)
            elif act_now == "nvfp4":
                if self.act_step_mults:
                    from . import harness as _H
                    xb, x_pts = qdq_blocks(blk, return_parts=True, scale_mult=self.act_step_mults[_H.STEP_IDX % len(self.act_step_mults)])
                elif self.act_scale_search:
                    xb, x_pts = qdq_blocks_search(blk, mults=self.act_scale_search, return_parts=True)
                else:
                    xb, x_pts = qdq_blocks(blk, return_parts=True)
                y = (xb.to(torch.bfloat16) @ self.wq.t()).float() * (x_pts * self.w_pts)
            else:
                y = (blk.to(torch.bfloat16) @ self.wq.t()).float() * self.w_pts
            if self.lr_a is not None:
                y = y + ((orig[i:i + self.chunk_tokens].to(torch.bfloat16) @ self.lr_a.t()) @ self.lr_b.t()).float()
            if self.bias is not None:
                y = y + self.bias.float()
            if coarse_y is not None:
                y = y + coarse_y[i:i + blk.shape[0]]
            out[i:i + blk.shape[0]] = y.to(x.dtype)
        return out.reshape(*shape[:-1], self.out_features)


def qdq_fp8_blocks(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """FP8 E4M3 activations with an FP32 scale per 16-element block (diagnostic W4A8 upper bound; the values
    are not exactly BF16-representable, so the matmul is an approximation of a true FP8 x FP4 GEMM)."""
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // block, block)
    s = (xf.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-30)
    return ((xf / s).to(torch.float8_e4m3fn).float() * s).reshape(shape)


E4M3_MAX = 448.0
E4M3_EMAX = 8                                        # largest normal E4M3 = 1.75 * 2^8


def qdq_mxfp8(x: torch.Tensor, block: int = 32, rounding: str = "rceil") -> torch.Tensor:
    """MX FP8 (E4M3 elements, shared E8M0 power-of-two scale per 32 elements), the activation format of a W4A8 GEMM
    on Blackwell (kind::f8f6f4). rounding="rceil" (default, what NVIDIA TransformerEngine / TorchAO RCEIL use):
    scale = 2^ceil(log2(amax / 448)), no element clips. rounding="floor" (OCP v1.0 reference): scale =
    2^(floor(log2 amax) - 8), block maxima in (448, 512) saturate at 448 (clips ~18% of blocks when channels have
    outliers; 2026-10-18 d06_mx with floor: qerr 119.5)."""
    shape = x.shape
    if shape[-1] % block:
        block = 16
    xf = x.float().reshape(*shape[:-1], shape[-1] // block, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    if rounding == "floor":
        e = torch.floor(torch.log2(amax)) - E4M3_EMAX
    else:
        e = torch.ceil(torch.log2(amax / E4M3_MAX))
    s = torch.exp2(e.clamp(-127, 127))
    q = torch.clamp(xf / s, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape)


def fp8_token_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token FP8 E4M3 (one FP32 scale = row amax / 448 per token): (E4M3 values, scale [N, 1])."""
    xf = x.float()
    s = (xf.abs().amax(-1, keepdim=True) / E4M3_MAX).clamp(min=1e-30)
    return torch.clamp(xf / s, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float(), s


def fp8_tensor_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 E4M3 (one FP32 scale = amax / 448 for the whole chunk): (E4M3 values, scale)."""
    xf = x.float()
    s = (xf.abs().amax() / E4M3_MAX).clamp(min=1e-30)
    return torch.clamp(xf / s, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float(), s


def rtn_weight(w_rot: torch.Tensor, scale_mult=1.0, scale_search=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest NVFP4 of an (already rotated/smoothed) weight [out, in]: (block-scaled values, pts).
    scale_search: list of block-scale multipliers chosen per block by squared error (4/6 = [1.0, 1.5])."""
    if scale_search:
        return qdq_blocks_search(w_rot, mults=tuple(scale_search), return_parts=True)
    return qdq_blocks(w_rot, return_parts=True, scale_mult=scale_mult)
