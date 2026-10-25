"""Write the exact layer lists of the W4A16 protection ablation to recipes/sets/ (Earth Science Informatics revision).

  python scripts/make_w16_sets.py          (CPU only, reads the layer manifest)

Each list is the 138-layer t7 set plus one protected group of t7_a16 added back:
  w16_t7_dec     + decoder resampler (decoder.level_decoder, 5 layers, and decoder.atmos_levels_embed) = 144
  w16_t7_enc     + encoder resampler (encoder.level_agg, 5 layers)                                     = 143
  w16_t7_res2    + both resamplers                                                                      = 149
  w16_t7_static  + the 51 state-independent layers of Q0 (ln_modulation, time_mlp, embeddings)          = 189
The released recipes/sets/ already contain these lists; the script regenerates them.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H  # noqa: E402

T7 = "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed"

dims = json.loads(H.MANIFEST.read_text(encoding="utf-8"))["dimensions"]
names = H.manifest_layers()
t7 = H.layer_set(T7)
dec = [n for n in names if n.startswith("decoder.level_decoder.") or n == "decoder.atmos_levels_embed"]
enc = [n for n in names if n.startswith("encoder.level_agg.")]
static = [n for n in H.layer_set("q0:exc=decoder.level_decoder,decoder.atmos_levels_embed") if n not in set(t7)]
assert (len(t7), len(dec), len(enc), len(static)) == (138, 6, 5, 51), (len(t7), len(dec), len(enc), len(static))
assert not set(dec + enc + static) & set(t7)


def params(ls):
    return sum(dims[n]["in_features"] * dims[n]["out_features"] for n in ls) / 1e6


sets = {"w16_t7_dec": t7 + dec, "w16_t7_enc": t7 + enc, "w16_t7_res2": t7 + dec + enc, "w16_t7_static": t7 + static}
out = ROOT / "recipes" / "sets"
out.mkdir(parents=True, exist_ok=True)
print(f"t7 {len(t7)} layers {params(t7):.1f} M | dec {params(dec):.2f} M | enc {params(enc):.2f} M | static {params(static):.1f} M")
for k, ls in sets.items():
    keep = set(ls)
    ordered = [n for n in names if n in keep]          # manifest order, as layer_set returns it
    (out / f"{k}.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    print(f"{k}: {len(ordered)} layers, {params(ordered):.1f} M -> {out / (k + '.txt')}")
