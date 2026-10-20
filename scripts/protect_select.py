"""B deep-dive, same-budget protection: pick attention modules to keep in BF16 under a parameter budget by
three rankings and write the corresponding Q0-numerics W4A4 recipes (everything in Q0's 195 layers quantized except the
protected modules' attention Linear layers).

Modules (8): Perceiver encoder / decoder (to_q, to_kv, to_out) and backbone stages enc0..2 / dec0..2 (attn.qkv + attn.proj
of all blocks). Rankings: B score = flip_rate(alpha=1) x value gap (replay files), yerr = mean layer-local activation
error of the module's layers (results/_diag_layers*.json), params = smallest modules first (an "obvious" baseline).

  python scripts/protect_select.py --budget 25e6
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402

RES = ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")


def module_layers():
    """Units: each Perceiver's attention (to_q/to_kv/to_out) and each Swin block's attention (qkv + proj)."""
    man = H.manifest_layers()
    mods = {"perc_enc": [n for n in man if n.startswith("encoder.level_agg") and ".0.0.to_" in n],
            "perc_dec": [n for n in man if n.startswith("decoder.level_decoder") and ".0.0.to_" in n]}
    for n in man:
        if ".attn." in n and n.startswith("backbone."):
            mods.setdefault(n.rsplit(".attn.", 1)[0] + ".attn", []).append(n)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=25e6)
    ap.add_argument("--diag", default=str(RES / "_diag_layers_all314.json"))
    a = ap.parse_args()
    dims = json.loads(H.MANIFEST.read_text())["dimensions"] if H.MODEL != "small" else None
    mods = module_layers()

    def params(n):
        if dims is not None:
            return int(dims[n]["in_features"]) * int(dims[n]["out_features"])
        raise SystemExit("param counts for small: not implemented")

    swin = json.load(open(RES / "_replay_swin.json"))
    d0 = next(iter(swin["blocks"]))
    B = {n: s["B_score"] for n, s in swin["blocks"][d0].items()}            # per Swin block (…blocks.k.attn)
    for k, f in (("perc_dec", "_replay_attn_decoder.json"), ("perc_enc", "_replay_attn_encoder.json")):
        r = json.load(open(RES / f)); s = r["dates"][next(iter(r["dates"]))]
        B[k] = s["alpha_1.0"]["flip_rate"] * s["vgap_rel_flipped"]
    diag = json.load(open(a.diag))
    dd = next(iter(next(iter(diag.values())).keys()))
    yerr = {m: sum(diag[n][dd]["yerr_tiny"] for n in ls if n in diag) / max(sum(1 for n in ls if n in diag), 1) for m, ls in mods.items()}
    psize = {m: sum(params(n) for n in ls) for m, ls in mods.items()}
    out = {"budget": a.budget, "modules": {m: {"params": psize[m], "B": B.get(m), "yerr": yerr[m], "n_layers": len(ls)} for m, ls in mods.items()}}
    q0 = H.layer_set("q0")
    missing = [m for m in mods if m not in B]
    if missing:
        raise SystemExit(f"no B score for {missing[:3]}... (replay keys mismatch)")
    rank = {"B": sorted(mods, key=lambda m: -B[m]), "yerr": sorted(mods, key=lambda m: -yerr[m]),
            "param": sorted(mods, key=lambda m: psize[m])}
    for name, order in rank.items():
        chosen, used = [], 0
        for m in order:
            if used + psize[m] <= a.budget:
                chosen.append(m); used += psize[m]
        prot = {n for m in chosen for n in mods[m]}
        layers = [n for n in q0 if n not in prot]
        (ROOT / "recipes" / f"layers_prot_{name}.txt").write_text("\n".join(layers) + "\n")
        rec = {"name": f"prot_{name}", "layers": f"recipes/layers_prot_{name}.txt",
               "rotation": {"block": "full", "dtype": "bf16"}, "weight": {"method": "rtn", "source": "bf16"}}
        (ROOT / "recipes" / f"prot_{name}.json").write_text(json.dumps(rec, indent=1))
        out[name] = {"protected_modules": chosen, "protected_params": used, "quantized_layers": len(layers)}
        print(f"{name:6s} protect {chosen} ({used / 1e6:.1f}M params) -> {len(layers)} layers quantized")
    (RES / "_protect_select.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["modules"], indent=1))


if __name__ == "__main__":
    main()
