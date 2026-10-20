"""Example 1 (CPU, seconds, no data): quantise one linear layer with the paper's deployable recipe.

A random layer whose inputs have a few outlier channels, as in transformer activations, is quantised four ways and
the relative output error is printed:
  naive W4A4      NVFP4 weights (round-to-nearest) and NVFP4 activations      (like q0 without rotation)
  W4A16 GPTQ      NVFP4 weights by GPTQ, BF16 activations                       (like c08)
  t7              SmoothQuant 0.8 + rank-32 branch + GPTQ, NVFP4 activations    (paper recipe t7)
  t7_mx           the same with MXFP8 activations                               (paper recipe t7_mx)

    python examples/01_deployable_layer.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cards.deployable import accumulate_stats, build_deployable_linear  # noqa: E402

torch.manual_seed(0)
n_in, n_out = 512, 512
w = torch.randn(n_out, n_in) / n_in ** 0.5
calib = torch.randn(8192, n_in); calib[:, :8] *= 40.0         # calibration inputs with outlier channels
test = torch.randn(4096, n_in); test[:, :8] *= 40.0           # held-out inputs
stats = accumulate_stats(calib)
ref = test @ w.t()

cases = {
    "naive W4A4 (RTN, NVFP4 act.)": dict(alpha=None, rank=0, act="nvfp4", method="rtn"),
    "W4A16 GPTQ (BF16 act.)": dict(alpha=None, rank=0, act="bf16", method="gptq"),
    "t7     (smooth+LR+GPTQ, NVFP4 act.)": dict(alpha=0.8, rank=32, act="nvfp4", method="gptq"),
    "t7_mx  (smooth+LR+GPTQ, MXFP8 act.)": dict(alpha=0.8, rank=32, act="mxfp8", method="gptq"),
}
print(f"{'recipe':40s} {'relative output error':>22s}")
for name, kw in cases.items():
    q = build_deployable_linear(w, None, stats, **kw)
    err = float(((q(test).float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
    print(f"{name:40s} {err:22.4f}")
