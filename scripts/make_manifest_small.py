"""Write recipes/manifest_small.txt for Aurora-small: every nn.Linear except the 9 output heads, in module order
(the rule that produces the 314-layer manifest of the 1.3B model). Run on the server (needs the aurora package).

  CUDA_VISIBLE_DEVICES= python scripts/make_manifest_small.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402

H._paths()
from aurora import AuroraSmallPretrained  # noqa: E402

m = AuroraSmallPretrained()
names = [n for n, mod in m.named_modules()
         if isinstance(mod, torch.nn.Linear) and not n.startswith(("decoder.atmos_heads", "decoder.surf_heads"))]
H.MANIFEST_SMALL.write_text("\n".join(names) + "\n", encoding="utf-8")
print(len(names), "->", H.MANIFEST_SMALL)
