"""How many weight values each native construction of a recipe changes, beyond FP32 rounding (no rollouts; Table S14j).

  AQ_MODEL=pretrained python scripts/native_weight_modes.py [--recipe t7_a16] > results/_native_weight_modes_t7_a16.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards.recipe import build  # noqa: E402
from cards.simlinear import SimNVFP4Linear  # noqa: E402
from scripts.native_t7 import NativeT7Linear, codes_and_scales  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="t7_a16")
    a = ap.parse_args()
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    cfg = NVFP4WeightOnlyConfig()
    model, _ = build(json.loads((ROOT / "recipes" / f"{a.recipe}.json").read_text(encoding="utf-8")))
    sims = [(n, m) for n, m in model.named_modules() if isinstance(m, SimNVFP4Linear)]
    out = {"recipe": a.recipe, "n_layers": len(sims), "tolerance": "1e-6 x layer max |W|", "modes": {}}
    total = sum(m.wq.numel() for _, m in sims)
    below6 = 0
    for _, m in sims:
        codes, _ = codes_and_scales(m.wq)
        below6 += int((codes.reshape(codes.shape[0], -1, 16).abs().amax(-1) < 6).sum())
    out["blocks_with_max_code_below_6"] = below6 / (total // 16)
    for mode in ("torchao", "fp32-tensor-scale", "from-codes"):
        changed = ulp = 0.0
        worst = 0.0
        for _, m in sims:
            nat = NativeT7Linear(m, cfg, "w4a16", weights=mode)
            changed += nat.mismatch_frac * m.wq.numel()
            ulp += nat.ulp_diff_frac * m.wq.numel()
            worst = max(worst, nat.mismatch_max_rel)
            del nat
            torch.cuda.empty_cache()
        out["modes"][mode] = {"changed_frac": changed / total, "any_bit_diff_frac": ulp / total,
                              "max_change_rel_to_layer_max": worst}
        print(mode, out["modes"][mode], file=sys.stderr, flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
