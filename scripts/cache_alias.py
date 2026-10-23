"""Re-link build-cache files written under the first (whole-recipe) cache key to the weight-only key.

  AQ_MODEL=pretrained python scripts/cache_alias.py t7mx_all314_noenc [more recipe names]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import recipe as R      # noqa: E402

for name in sys.argv[1:]:
    r = R.merged(json.loads((ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8")))
    old, new = R._legacy_cache_path(r), R._cache_path(r)
    if new.exists():
        print(f"{name}: {new.name} already present")
    elif old.exists():
        os.link(old, new)
        print(f"{name}: linked {old.name} -> {new.name}")
    else:
        print(f"{name}: no cache file ({old.name})")
