"""Group the 62 layers that the all-layer W4A16 configuration quantises beyond t7 (60 inherited Swin and patch layers and
the encoder surface MLP) and write one layer list per group: t7 plus that group (Table 5, lower block; Table S17b).

  python scripts/make_w16_groups.py [--dry]          (CPU only)
The released recipes/sets/ already contain these lists; the script regenerates them.
"""
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H  # noqa: E402

T7 = "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed"
ALL = "all314:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed,encoder.level_agg"

dims = json.loads(H.MANIFEST.read_text(encoding="utf-8"))["dimensions"]
names = H.manifest_layers()
t7 = H.layer_set(T7)
extra = [n for n in H.layer_set(ALL) if n not in set(t7)]
assert len(t7) == 138 and len(extra) == 62, (len(t7), len(extra))


def group(n: str) -> str:
    if n.startswith("encoder.surf_mlp"):
        return "surf_mlp"
    m = re.match(r"backbone\.(encoder|decoder)_layers\.(\d+)\.(blocks|downsample|upsample)", n)
    if m:
        side, stage, part = m.group(1), m.group(2), m.group(3)
        return ("patch" if part != "blocks" else f"{side[:3]}{stage}")
    return "other"


groups = OrderedDict()
for n in extra:
    groups.setdefault(group(n), []).append(n)
out = ROOT / "recipes" / "sets"
for g, ls in groups.items():
    p = sum(dims[n]["in_features"] * dims[n]["out_features"] for n in ls) / 1e6
    print(f"{g}: {len(ls)} layers, {p:.1f} M  e.g. {ls[0]}")
    if "--dry" not in sys.argv:
        keep = set(t7) | set(ls)
        (out / f"w16_t7_inh_{g}.txt").write_text("\n".join(n for n in names if n in keep) + "\n", encoding="utf-8")
