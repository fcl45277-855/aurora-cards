"""Aurora quantization "card" harness: build a quantized Aurora from a recipe, roll it out, score it.

Locations (environment variables, defaults relative to the repository root):
  AQ_CKPT_DIR   directory with the Aurora checkpoints (aurora-0.25-pretrained.ckpt, -small-pretrained, -finetuned)
  AQ_ERA5_DIR   directory with static.nc and the per-day ERA5 files surface_<YYYY-MM-DD>.npz / air_<YYYY-MM-DD>.npz
                (written by scripts/download_era5_day.py; see docs/DATA.md)
  AQ_MODEL      pretrained | finetuned | small
The Aurora package itself is imported from the Python environment (pip install microsoft-aurora; see README).

Scoring (per date, rollout step, variable), all on the 720 x 1440 cropped grid, latitude-weighted (cos lat):
  rmse_truth  vs ERA5 (surface 2t/10u/10v/msl; z500, t850, u850, v850, q700)
  rmse_ref    vs the FP32 Aurora prediction for the same date/step (quantization error only, label-free)
Headline numbers (see summarize): mean over variables of rmse_truth / rmse_truth(FP32) - 1  ("excess %"),
and the share of Q0's excess that a card removes ("gap closed").
"""
from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AQ_ERA5_DIR", str(REPO / "data" / "era5")))
CKPT_DIR = Path(os.environ.get("AQ_CKPT_DIR", str(REPO / "checkpoints")))
# Which Aurora checkpoint (env AQ_MODEL; cross-model replication 2026-10-19). "pretrained" is the model of all earlier
# cards and is built exactly as before. "finetuned" has the same architecture (+ additive per-step LoRA modules that
# stay in full precision), so the 314-layer manifest and the Q0 lists apply. "small" (113M) uses its own manifest of
# all nn.Linear layers except the 9 output heads, which is the rule that produced the 314-layer manifest.
MODEL = os.environ.get("AQ_MODEL", "pretrained")
MODELS = {"pretrained": "aurora-0.25-pretrained.ckpt", "small": "aurora-0.25-small-pretrained.ckpt",
          "finetuned": "aurora-0.25-finetuned.ckpt"}
CKPT = CKPT_DIR / MODELS[MODEL]
MANIFEST = REPO / "recipes" / "manifest_314.json"   # the 314 non-head linear layers of Aurora 0.25 (pretrained/finetuned)
MANIFEST_SMALL = Path(__file__).resolve().parents[1] / "recipes" / "manifest_small.txt"
Q0_BF16_LIST = REPO / "recipes" / "q0_bf16_layers.txt"   # the 119 layers kept in BF16 by the Q0 mixed-precision set
STEP_IDX = 0                                      # current rollout step (0-based), set by rollout()
SEED = 20260809                                   # Q0 builder seed (+ manifest index, 1-based)
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURF = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
ATMOS = (("z", 500), ("t", 850), ("u", 850), ("v", 850), ("q", 700))
VARS = list(SURF) + [f"{v}{l}" for v, l in ATMOS]
FP32_SOURCES = ("2t", "10u", "10v", "msl")

SPLITS = {
    "calib": [f"2024-01-{d:02d}" for d in range(4, 11)],          # 7 dates, used only to collect statistics
    "calib2": [f"2024-01-{d:02d}" for d in range(18, 25)],        # disjoint calibration set (robustness check)
    "dev": [f"2024-02-{d:02d}" for d in range(2, 14)],             # 12 dates, card screening
    "test": [f"2024-03-{d:02d}" for d in range(1, 31)],            # 30 dates (03-01..03-30; npz ends 03-30), held out
    "calib1": ["2024-01-04"],                                       # 1 date: examples only
    "summer": [f"2024-07-{d:02d}" for d in range(1, 9)],           # 8 dates, July 2024 (CDS, scripts/download_era5_day.py):
    "y2023": [f"2023-03-{d:02d}" for d in range(1, 5)],            # 4 dates, March 2023 (CDS, 2026-10-20): cross-year check
    "oct": [f"2024-10-{d:02d}" for d in range(1, 15)],            # 14 dates, October 2024: frozen confirmation of t7_a16
}                                                                   # seasonal robustness of winter-calibrated recipes


def _paths():
    """Kept for call-site compatibility: Aurora is imported from the installed package."""
    return None


def manifest_layers() -> list[str]:
    if MODEL == "small":
        names = [l.strip() for l in MANIFEST_SMALL.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(names) == 146, len(names)
        return names
    d = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = list(d["dimensions"].keys())
    assert len(names) == 314
    return names


def layer_set(spec: str) -> list[str]:
    """'q0' | 'all314' | path, optionally followed by ':inc=a,b' (keep names containing any substring) or
    ':exc=a,b' (drop them), e.g. 'q0:exc=ln_modulation,time_mlp'. A pattern 'a&b' matches names containing both."""
    if ":" in spec and not Path(spec).exists():
        base, rule = spec.split(":", 1)
        mode, pats = rule.split("=", 1)
        pats = [p for p in pats.split(",") if p]
        names = layer_set(base)
        hit = [n for n in names if any(all(q in n for q in p.split("&")) for p in pats)]   # 'a&b' = both substrings
        return hit if mode == "inc" else [n for n in names if n not in set(hit)]
    names = manifest_layers()
    if spec == "all314":
        return names
    if spec == "q0":
        if MODEL == "small":
            raise ValueError("no Q0 layer list for Aurora-small: use rule-based 'all314:...' specs")
        keep = {l.strip() for l in Q0_BF16_LIST.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
        out = [n for n in names if n not in keep]
        assert len(out) == 195, len(out)
        return out
    p = Path(spec)
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---- model -----------------------------------------------------------------------------------------

def load_fp32(device="cuda"):
    _paths()
    from aurora import Aurora, Batch, Metadata
    from cards.compat import preserve_coordinate_precision
    preserve_coordinate_precision(Batch, Metadata)
    if not CKPT.exists():
        raise FileNotFoundError(f"Aurora checkpoint not found: {CKPT} (set AQ_CKPT_DIR; see README)")
    if MODEL == "small":
        from aurora import AuroraSmallPretrained
        m = AuroraSmallPretrained()
    elif MODEL == "finetuned":
        m = Aurora()                      # default use_lora=True, lora_mode "single", as the checkpoint was trained
    else:
        m = Aurora(use_lora=False)
    m.load_checkpoint_local(str(CKPT))
    chunk_perceiver(int(os.environ.get("AQ_PERCEIVER_CHUNK", "0")))
    return m.eval().to(device)


def chunk_perceiver(patches: int) -> None:
    """Memory-light mode (for GPUs shared with other jobs): run Aurora's Perceiver resamplers (encoder level_agg,
    decoder level_decoder) over at most `patches` patches at a time. Each patch is independent (attention, MLP and
    LayerNorm act within a patch), and with patches a multiple of 16384 every chunk boundary of SimNVFP4Linear's
    16384-token dynamic activation scale stays where it was (C' or C tokens per patch), so results match the
    unchunked run up to GEMM-shape round-off. 0 = off (default)."""
    from aurora.model.perceiver import PerceiverResampler
    orig = PerceiverResampler.__dict__.get("_aq_orig_forward", PerceiverResampler.forward)
    PerceiverResampler._aq_orig_forward = orig
    if not patches:
        PerceiverResampler.forward = orig
        return
    if patches % 16384:
        raise ValueError("AQ_PERCEIVER_CHUNK must be a multiple of 16384 (keeps activation-scale chunks aligned)")

    def forward(self, latents, x):
        n = latents.shape[0]
        if n <= patches:
            return orig(self, latents, x)
        return torch.cat([orig(self, latents[i:i + patches], x[i:i + patches]) for i in range(0, n, patches)])
    PerceiverResampler.forward = forward


def parent_of(model, name):
    p = model
    for part in name.split(".")[:-1]:
        p = getattr(p, part)
    return p, name.rsplit(".", 1)[-1]


# ---- data ------------------------------------------------------------------------------------------

def _npz_member(path: Path, member: str) -> np.ndarray:
    """Memory-map an uncompressed npz member (falls back to a full load)."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return np.load(path)[member]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        n, e = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
        f.seek(info.header_offset + 30 + n + e)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0(f))
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


_HOUR_INDEX: dict = {}
_DESCENDING: dict = {}


def levels_descending(day: str) -> bool:
    """True if air_<day>.npz stores pressure levels 1000 -> 50 hPa (the 2024-01..03 files do), False if 50 -> 1000
    (files written by quark_pipeline/download_era5_day.py). Decided from the data: geopotential grows with height.
    Aurora's Metadata.atmos_levels is LEVELS (50 -> 1000), so descending files must be flipped on the level axis.
    (Bug found 2026-10-18: the original raw_batch fed these files without flipping, i.e. vertically inverted.)"""
    if day not in _DESCENDING:
        z = _npz_member(DATA / f"air_{day}.npz", "z")
        lo, hi = float(np.asarray(z[0, 0, ::40, ::40]).mean()), float(np.asarray(z[0, -1, ::40, ::40]).mean())
        _DESCENDING[day] = lo < hi
    return _DESCENDING[day]


def level_index(day: str, lev: int) -> int:
    i = LEVELS.index(lev)
    return len(LEVELS) - 1 - i if levels_descending(day) else i


def hour_index(day: str, hour: int) -> int:
    """Position of `hour` (UTC) in the day's files: full hourly files (24 slots) or 4-slot subsets written by
    scripts/cut_stream.py; both carry valid_time in surface_<day>.npz, and air_<day>.npz uses the same order."""
    if day not in _HOUR_INDEX:
        vt = np.asarray(_npz_member(DATA / f"surface_{day}.npz", "valid_time"))
        hrs = ((vt - vt.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int)
        _HOUR_INDEX[day] = {int(h): i for i, h in enumerate(hrs)}
    return _HOUR_INDEX[day][hour]


class Truth:
    """ERA5 truth at arbitrary valid times (hourly files, one per date)."""

    def __init__(self):
        self.cache = {}

    def get(self, when: np.datetime64, var: str) -> np.ndarray:
        day = str(when.astype("datetime64[D]"))
        hour = int((when - when.astype("datetime64[D]")) / np.timedelta64(1, "h"))
        key = (day, hour, var)
        if key not in self.cache:
            i = hour_index(day, hour)
            if var in SURF:
                a = _npz_member(DATA / f"surface_{day}.npz", SURF[var])[i]
            else:
                v, lev = var[0], int(var[1:])
                a = _npz_member(DATA / f"air_{day}.npz", v)[i, level_index(day, lev)]
            self.cache[key] = np.asarray(a, dtype=np.float32)[:720]
            if len(self.cache) > 400:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def lat_weights(model_lat: torch.Tensor) -> np.ndarray:
    lat = model_lat.float().cpu().numpy()[:720]
    w = np.cos(np.deg2rad(lat))
    return (w / w.mean())[:, None]


def fields(pred) -> dict[str, np.ndarray]:
    out = {k: pred.surf_vars[k][0, 0].float().cpu().numpy() for k in SURF}
    for v, lev in ATMOS:
        out[f"{v}{lev}"] = pred.atmos_vars[v][0, 0, LEVELS.index(lev)].float().cpu().numpy()
    return out


# ---- rollout ---------------------------------------------------------------------------------------

def raw_batch_fast(date: str):
    """Same Batch as run_aurora_quarot_nvfp4_90d.raw_batch (history 00 and 06 UTC), via memory maps."""
    _paths()
    import xarray as xr
    from aurora import Batch, Metadata
    hist = [hour_index(date, 0), hour_index(date, 6)]
    static = xr.open_dataset(DATA / "static.nc", engine="netcdf4")
    surf = DATA / f"surface_{date}.npz"
    air = DATA / f"air_{date}.npz"
    vt = _npz_member(surf, "valid_time")
    batch = Batch(
        static_vars={k: torch.from_numpy(static[k].values[0]) for k in ("z", "slt", "lsm")},
        surf_vars={k: torch.from_numpy(np.ascontiguousarray(_npz_member(surf, src)[hist])[None]) for k, src in SURF.items()},
        atmos_vars={k: torch.from_numpy(np.ascontiguousarray(
            _npz_member(air, k)[hist][:, ::-1] if levels_descending(date) else _npz_member(air, k)[hist])[None])
            for k in ("t", "u", "v", "q", "z")},
        metadata=Metadata(lat=torch.from_numpy(static.latitude.values), lon=torch.from_numpy(static.longitude.values),
                          time=(np.asarray(vt[hist[-1]]).astype("datetime64[s]").tolist(),), atmos_levels=LEVELS))
    static.close()
    return batch


def rollout(model, date: str, steps: int, surface_fp32: bool) -> list:
    """Initialise from ERA5 (00 and 06 UTC of `date`) and roll forward `steps` 6-hour steps."""
    _paths()
    from cards.compat import prepare
    batch = raw_batch_fast(date)
    patch = None
    if surface_fp32:
        raise NotImplementedError("surface_fp32 was a patch for the earlier full-BF16 pipeline and is not part of the "
                                  "released harness; all paper results use Aurora's autocast regime")
    else:
        current = prepare(model, batch)
    preds = []
    try:
        with torch.inference_mode():
            for _step in range(steps):
                global STEP_IDX
                STEP_IDX = _step                    # read by SimNVFP4Linear.act_step_mults (per-step activation scale)
                p = model(current)
                preds.append(p.to("cpu"))
                current = dataclasses.replace(
                    p,
                    surf_vars={k: torch.cat([current.surf_vars[k][:, 1:], v], 1) for k, v in p.surf_vars.items()},
                    atmos_vars={k: torch.cat([current.atmos_vars[k][:, 1:], v], 1) for k, v in p.atmos_vars.items()})
    finally:
        if patch is not None:
            patch.close()
    return preds, batch.metadata.lat


def score(preds, lat, truth: Truth, ref: list[dict] | None) -> list[dict]:
    w = lat_weights(lat)
    rows = []
    for s, p in enumerate(preds, 1):
        when = np.datetime64(p.metadata.time[0], "s")
        f = fields(p)
        for var in VARS:
            a = f[var]
            t = truth.get(when, var)
            d = a - t
            row = {"step": s, "var": var, "valid": str(when), "rmse_truth": float(np.sqrt(np.mean(w * d * d))),
                   "bias_truth": float(np.mean(w * d))}
            if ref is not None:
                r = a - ref[s - 1][var]
                row["rmse_ref"] = float(np.sqrt(np.mean(w * r * r)))
            rows.append(row)
    return rows


def ref_path(cache: Path, date: str, step: int) -> Path:
    return cache / f"{date}_s{step}.npz"


def load_ref(cache: Path, date: str, steps: int) -> list[dict] | None:
    out = []
    for s in range(1, steps + 1):
        p = ref_path(cache, date, s)
        if not p.exists():
            return None
        z = np.load(p)
        out.append({k: z[k] for k in z.files})
    return out


def save_ref(cache: Path, date: str, preds) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for s, p in enumerate(preds, 1):
        np.savez(ref_path(cache, date, s), **fields(p))


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def summarize(records: list[dict], baselines: dict[str, list[dict]] | None = None) -> dict:
    """Per variable/step means; with baselines {'fp32': recs, 'q0': recs} also excess % and gap closed,
    paired by date with a date-bootstrap CI for the headline."""
    def table(recs):
        t = {}
        for r in recs:
            t.setdefault((r["date"], r["step"], r["var"]), r)
        return t
    mine = table(records)
    out = {"n_dates": len({k[0] for k in mine}), "per_var": {}}
    for var in VARS:
        for step in sorted({k[1] for k in mine}):
            vals = [r for k, r in mine.items() if k[1] == step and k[2] == var]
            if not vals:
                continue
            e = {"rmse_truth": float(np.mean([r["rmse_truth"] for r in vals]))}
            if all("rmse_ref" in r for r in vals):
                e["rmse_ref"] = float(np.mean([r["rmse_ref"] for r in vals]))
            out["per_var"][f"{var}_s{step}"] = e
    if baselines and "fp32" in baselines:
        fp = table(baselines["fp32"])
        q0 = table(baselines["q0"]) if "q0" in baselines else None
        keys = [k for k in mine if k in fp and (q0 is None or k in q0)]
        dates = sorted({k[0] for k in keys})
        by_date = {d: [k for k in keys if k[0] == d] for d in dates}
        has_ref = all("rmse_ref" in mine[k] for k in keys) and (q0 is None or all("rmse_ref" in q0[k] for k in keys))

        def headline(ds: list[str]) -> dict:
            """ds may repeat dates (bootstrap). Primary: quantization error = rmse vs the FP32 forecast, normalised
            by FP32's own rmse vs ERA5 (per variable/step), averaged; reduction vs Q0 = 1 - mine/Q0 of that mean."""
            ks = [k for d in ds for k in by_date[d]]
            f = np.array([fp[k]["rmse_truth"] for k in ks])
            h = {"excess": float(np.mean(np.array([mine[k]["rmse_truth"] for k in ks]) / f - 1))}
            if has_ref:
                h["qerr"] = float(np.mean(np.array([mine[k]["rmse_ref"] for k in ks]) / f))
                if q0 is not None:
                    q = float(np.mean(np.array([q0[k]["rmse_ref"] for k in ks]) / f))
                    h["red"] = 1 - h["qerr"] / q if q else float("nan")
            return h
        main = headline(dates)
        rng = np.random.default_rng(0)
        boots = [headline(list(rng.choice(dates, len(dates)))) for _ in range(2000)]

        def ci(key):
            return [100 * float(np.percentile([b[key] for b in boots], p)) for p in (2.5, 97.5)]
        out["excess_pct"], out["excess_pct_ci95"] = 100 * main["excess"], ci("excess")
        if "qerr" in main:
            out["qerr_pct"], out["qerr_pct_ci95"] = 100 * main["qerr"], ci("qerr")
        if "red" in main:
            out["qerr_reduction_vs_q0_pct"], out["qerr_reduction_vs_q0_pct_ci95"] = 100 * main["red"], ci("red")
        out["n_paired"] = len(keys)
        out["ci_method"] = ("i.i.d. date bootstrap (2000 resamples) for the *_ci95 fields; the paper's intervals use "
                            "3-day moving blocks (scripts/block_ci.py, scripts/extended_analyses.py)")
    return out


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
