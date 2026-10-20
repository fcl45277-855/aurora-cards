"""CPU checks for the to_kv probes in SimNVFP4Linear (kv_split, k_centre).

  CUDA_VISIBLE_DEVICES= python scripts/test_kv_probe.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cards.simlinear import SimNVFP4Linear, rtn_weight    # noqa: E402

torch.manual_seed(0)
D, P, C = 64, 50, 3
w = torch.randn(2 * D, D) / D ** 0.5
x = torch.randn(P, C, D) + 3.0 * torch.randn(1, 1, D)       # large common component, like a latent with an offset
q = torch.randn(P, 5, D)
parts, pts = rtn_weight(w)


def lin():
    return SimNVFP4Linear(parts, pts, None, rotation=None)


def attn(kv):
    k, v = kv.float().chunk(2, -1)
    return torch.softmax(q @ k.transpose(-1, -2) / D ** 0.5, -1) @ v


exact = x @ w.t()
# 1. kv_split with an empty error range = exact output
m = lin(); m.w_exact = w; m.quant_rows = (0, 0)
e1 = (m(x).float() - exact).abs().max().item()
# 2. kv_split rows [0, D) only: V half exact, K half equals the plain sim output
m2 = lin(); m2.w_exact = w; m2.quant_rows = (0, D)
plain = lin()(x).float()
y2 = m2(x).float()
e2v = (y2[..., D:] - exact[..., D:]).abs().max().item()
e2k = (y2[..., :D] - plain[..., :D]).abs().max().item()
# 3. alpha scales the injected error
m3 = lin(); m3.w_exact = w; m3.quant_rows = (0, D); m3.err_alpha = 0.5
e3 = ((m3(x).float()[..., :D] - exact[..., :D]) - 0.5 * (plain[..., :D] - exact[..., :D])).abs().max().item()
# 4. k_centre: centring the keys over the context tokens leaves attention unchanged in exact arithmetic, and reduces
#    the attention error of the quantized layer when the input has a large common component
xc = x - x.mean(-2, keepdim=True)
kv_c = torch.cat([xc @ w[:D].t(), x @ w[D:].t()], -1)
e4 = (attn(kv_c) - attn(exact)).abs().max().item()
m4 = lin(); m4.k_centre = (0, D)
err_plain = (attn(plain) - attn(exact)).pow(2).mean().sqrt().item()
err_kc = (attn(m4(x)) - attn(exact)).pow(2).mean().sqrt().item()
print(f"1 empty split vs exact      max |diff| {e1:.2e}  (expect ~BF16 cast, < 1e-1 of |y| ~ {exact.abs().max():.1f})")
print(f"2 V half exact               max |diff| {e2v:.2e}; K half = plain sim: {e2k:.2e}")
print(f"3 alpha 0.5 scaling          max |diff| {e3:.2e}")
print(f"4 centred keys, exact math   max |attn diff| {e4:.2e} (expect ~0)")
print(f"  attention RMSE: plain sim {err_plain:.4f}, k_centre {err_kc:.4f}")
ok = e2k < 1e-6 and e3 < 1e-2 and e4 < 1e-4 and err_kc < err_plain
print("PASS" if ok else "FAIL")
