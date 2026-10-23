"""Composition of the Q0 and t7 layer sets (paper Table 2): layer counts and weights per category, from the layer
manifest and recipe definitions only (no checkpoint or GPU needed).

    python scripts/layer_sets.py
"""
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
# t7 removes these name fragments from Q0 (recipes/t7_mx.json: "q0:exc=...")
T7_EXCLUDE = json.loads((RECIPES / "t7_mx.json").read_text())["layers"].split("exc=", 1)[1].split(",")


def category(name):
    if "level_agg" in name:
        return "Encoder resampler (level_agg)"
    if "level_decoder" in name or "decoder.atmos_levels_embed" in name:
        return "Decoder resampler (level_decoder and its level embedding)"
    if "ln_modulation" in name or "time_mlp" in name or "_embed" in name:
        return "Other state-independent layers (embeddings, time MLP, LayerNorm modulation)"
    if name.startswith("encoder."):
        return "Encoder surface MLP"
    if ".attn." in name:
        return "Swin attention (qkv, proj)"
    if ".mlp." in name:
        return "Swin MLP (fc1, fc2)"
    return "Patch merging and splitting"


def main():
    manifest = json.loads((RECIPES / "manifest_314.json").read_text())
    dims, names = manifest["dimensions"], manifest["selected_layers"]
    bf16 = {l.strip() for l in (RECIPES / "q0_bf16_layers.txt").read_text().splitlines() if l.strip()}
    q0 = [n for n in names if n not in bf16]
    t7 = [n for n in q0 if not any(e in n for e in T7_EXCLUDE)]
    weights = lambda n: dims[n]["in_features"] * dims[n]["out_features"]
    rows = OrderedDict()
    for n in names:
        r = rows.setdefault(category(n), [0, 0, 0, 0, 0, 0])
        for i, member in enumerate((True, n in q0, n in t7)):
            if member:
                r[2 * i] += 1
                r[2 * i + 1] += weights(n)
    print(f"{'Layer category':78s} {'All (n / M)':>16s} {'Q0 (n / M)':>16s} {'t7 (n / M)':>16s}")
    total = [0] * 6
    for c, r in rows.items():
        total = [a + b for a, b in zip(total, r)]
        print(f"{c:78s} " + " ".join(f"{r[2 * i]:>6d} / {r[2 * i + 1] / 1e6:7.1f}" for i in range(3)))
    print(f"{'Total':78s} " + " ".join(f"{total[2 * i]:>6d} / {total[2 * i + 1] / 1e6:7.1f}" for i in range(3)))


if __name__ == "__main__":
    main()
