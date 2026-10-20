"""Example 2 (CPU, seconds, no data): why few-key cross-attention is fragile under key quantisation.

The same relative key perturbation is applied to (a) a learned-query cross-attention with 13 queries over 3 keys and
sharp attention, like Aurora's decoder resampler, and (b) a soft self-attention over 144 tokens, like a Swin window.
The switching diagnostic (cards/switching.py) reports how many cells switch their argmax key, the share of the output
error those cells carry (err_in_flip, the paper's sensitivity predictor), the common-mode amplification A and the
nonlinearity G, as the noise amplitude is reduced. The
keys are synthetic; the paper's values come from the replay of Aurora itself (scripts/replay_attn.py).

    python examples/02_switching_diagnostic.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cards.switching import switching_diagnostic  # noqa: E402

g = torch.Generator().manual_seed(0)
d = 64
# (a) few-key cross-attention: 13 learned queries over 3 keys that share a large common component
k = torch.randn(1024, 1, d, generator=g) * 3.0 + torch.randn(1024, 3, d, generator=g) * 0.5
q = torch.randn(1024, 13, d, generator=g) * 3.0
v = torch.randn(1024, 3, d, generator=g)
# (b) soft window attention among 144 tokens
q2 = torch.randn(128, 144, d, generator=g) * 0.3
k2 = torch.randn(128, 144, d, generator=g) * 0.3
v2 = torch.randn(128, 144, d, generator=g)
rel = lambda kk: 0.1 * kk.norm(dim=-1, keepdim=True) / d ** 0.5 * torch.randn(kk.shape, generator=g)   # 10 % key error

print("same 10 % relative key error, scaled by alpha:\n")
print(f"{'module':34s} {'alpha':>5s} {'sharp':>6s} {'flip':>7s} {'err_in_flip':>11s} {'A':>6s} {'G':>6s} {'rel.out.err':>11s}")
for name, (qq, kk, vv) in (("few-key cross-attention (3 keys)", (q, k, v)), ("soft window attention (144 keys)", (q2, k2, v2))):
    dk = rel(kk)
    for alpha in (1.0, 0.5, 0.25):
        r = switching_diagnostic(qq, kk, dk, vv, alpha=alpha)
        print(f"{name:34s} {alpha:5.2f} {r['share_sharp']:6.2f} {r['flip_rate']:7.4f} {r['err_in_flip']:11.3f} "
              f"{r['A']:6.2f} {r['G']:6.2f} {r['relative_output_error']:11.4f}")
print("\nThe synthetic keys only illustrate the quantities. The values reported in the paper (A = 19, G = 11, err_in_flip >= 0.94)"
      "\ncome from replaying Aurora's resamplers with their measured quantisation error: scripts/replay_attn.py.")
