"""Check the software environment, the Aurora package and the data/checkpoint locations before a run.

    python scripts/check_env.py

Reports package versions against the ones used for the paper, verifies that the installed Aurora package source is
identical to the version used (docs/aurora_package_md5.txt; a different Aurora version may change results), and lists
what is present under AQ_CKPT_DIR and AQ_ERA5_DIR.
"""
import hashlib
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = {"torch": "2.12.1", "torchao": "0.18.0", "numpy": "2.5.0", "scipy": "1.18.0", "xarray": "2026.4.0", "pyshtools": None, "mslk": "1.2.0"}


def version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def main():
    print(f"python {sys.version.split()[0]} (paper: 3.12)")
    for mod, want in PAPER.items():
        have = version(mod)
        note = "" if have is None or want is None or have.startswith(want) else f"   (paper used {want})"
        print(f"{mod:10s} {have or 'NOT INSTALLED'}{note}")
    try:
        import torch
        print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "",
              torch.cuda.get_device_capability() if torch.cuda.is_available() else "")
    except Exception as e:
        print("torch error:", e)
    # Aurora package identity
    try:
        import aurora
        root = Path(aurora.__file__).resolve().parent.parent
        ref = [l.split() for l in (ROOT / "docs" / "aurora_package_md5.txt").read_text().splitlines() if l.strip()]
        diff = [p for h, p in ref if not (root / p).exists() or hashlib.md5((root / p).read_bytes()).hexdigest() != h]
        print(f"aurora package at {root}: {len(ref) - len(diff)}/{len(ref)} source files identical to the paper's version"
              + ("" if not diff else f"; differing: {diff[:6]}{' ...' if len(diff) > 6 else ''}"))
    except Exception as e:
        print("aurora package: NOT IMPORTABLE", e)
    ck = Path(os.environ.get("AQ_CKPT_DIR", ROOT / "checkpoints"))
    print(f"AQ_CKPT_DIR {ck}: {sorted(p.name for p in ck.glob('*.ckpt')) if ck.exists() else 'missing'}")
    era = Path(os.environ.get("AQ_ERA5_DIR", ROOT / "data" / "era5"))
    days = sorted({p.stem.split('_', 1)[1] for p in era.glob('surface_*.npz')}) if era.exists() else []
    print(f"AQ_ERA5_DIR {era}: static.nc {'present' if (era / 'static.nc').exists() else 'MISSING'}; {len(days)} days"
          + (f" ({days[0]} .. {days[-1]})" if days else ""))


if __name__ == "__main__":
    main()
