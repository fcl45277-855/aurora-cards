"""Unit check of N1 (scale-separated activation quantization) and its prior-work ablations on synthetic
spatially smooth activations: rel. error of the layer output vs the exact-activation output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from cards.simlinear import SimNVFP4Linear, rtn_weight

torch.manual_seed(0)
C, H, W, K, O = 2, 45, 90, 256, 128
yy, xx = torch.meshgrid(torch.linspace(0, 3.14, H), torch.linspace(0, 6.28, W), indexing="ij")
base = torch.stack([torch.sin(yy * (1 + i % 3)) * torch.cos(xx * (1 + i % 5)) * (1 + i % 7) for i in range(K)], -1)
x = (base.unsqueeze(0).repeat(C, 1, 1, 1) + 0.1 * torch.randn(C, H, W, K)).reshape(-1, K)
w = torch.randn(O, K) * 0.05
p, s = rtn_weight(w)
ref = x @ (p * s).t()


def run(act="nvfp4", mode=None, bits="bf16"):
    m = SimNVFP4Linear(p, s, None, rotation=None, act_bits=act)
    if mode:
        m.lowpass_pool, m.grid, m.lowpass_mode, m.coarse_bits = 4, (C, H, W), mode, bits
    return float((m(x).float() - ref).norm() / ref.norm())


print(f"bf16 act, plain            {run('bf16'):.4f}")
print(f"bf16 act, N1 mean          {run('bf16', 'mean'):.4f}  (decomposition exact)")
print(f"nvfp4, plain               {run():.4f}")
for mode, bits in (("mean", "bf16"), ("mean", "fp8"), ("global", "bf16"), ("haar", "bf16"), ("haar", "fp8")):
    print(f"nvfp4, {mode:6s} keep={bits:4s}    {run('nvfp4', mode, bits):.4f}")
