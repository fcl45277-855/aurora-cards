"""Regression tests for the two pipeline errors fixed on 2026-10-18 and for truth/time alignment (CPU, no Aurora
weights or ERA5 needed; small synthetic files in the harness's npz layout).

  1. Pressure-level order: the 2024-01..03 files store 1000 -> 50 hPa, files from scripts/download_era5_day.py store
     50 -> 1000 hPa. The harness must read the same physical level from both (the original reader did not flip).
  2. Hour index: 4-slot (00/06/12/18 UTC) and 24-slot files must give the slot of the requested valid hour.
  3. Scoring aligns each forecast step with the truth at its own valid time (a 6-h shift must show up as error).
  4. Inputs follow the model's parameter dtype (FP32 for the autocast regime of every recipe); the original pipeline
     cast the unnormalised physical inputs to BF16. No recipe may select the legacy full-BF16 path.
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H  # noqa: E402
from cards.compat import prepare  # noqa: E402
from cards.recipe import merged  # noqa: E402

NY, NX = 5, 8


def _value(var, lev, hour):
    """Synthetic field value: unique per variable, level and hour; geopotential decreases with pressure."""
    base = {"z": 1.0e5, "t": 300.0, "u": 10.0, "v": -5.0, "q": 1.0e-3}[var]
    sign = -1.0 if var == "z" else 1.0
    return base + sign * 37.0 * lev + 0.25 * hour


def _write_day(path, day, descending, hours=(0, 6, 12, 18)):
    vt = np.array([np.datetime64(f"{day}T{h:02d}:00") for h in hours], dtype="datetime64[ns]")
    levs = list(H.LEVELS)[::-1] if descending else list(H.LEVELS)
    air = {}
    for var in ("z", "t", "u", "v", "q"):
        a = np.zeros((len(hours), 13, NY, NX), np.float32)
        for i, h in enumerate(hours):
            for j, lev in enumerate(levs):
                a[i, j] = _value(var, lev, h)
        air[var] = a
    surf = {src: np.stack([np.full((NY, NX), 1000.0 * (k + 1) + 0.25 * h, np.float32) for h in hours])
            for k, src in enumerate(H.SURF.values())}
    np.savez(path / f"air_{day}.npz", **air, valid_time=vt)
    np.savez(path / f"surface_{day}.npz", **surf, valid_time=vt)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "DATA", tmp_path)
    H._HOUR_INDEX.clear()
    H._DESCENDING.clear()
    yield tmp_path
    H._HOUR_INDEX.clear()
    H._DESCENDING.clear()


def test_level_order_is_inferred_from_geopotential(data):
    _write_day(data, "2024-03-01", descending=True)                # 1000 -> 50 hPa, as the 2024-01..03 files
    _write_day(data, "2024-07-01", descending=False)               # 50 -> 1000 hPa, as download_era5_day.py writes
    assert H.levels_descending("2024-03-01") and not H.levels_descending("2024-07-01")
    for day in ("2024-03-01", "2024-07-01"):
        truth = H.Truth()
        for var in ("z500", "t850", "u850", "v850", "q700"):
            v, lev = var[0], int(var[1:])
            got = truth.get(np.datetime64(f"{day}T12:00"), var)
            assert np.allclose(got, _value(v, lev, 12)), (day, var, float(got.mean()))


def test_hour_index_four_and_24_slot_files(data):
    _write_day(data, "2024-03-01", descending=True, hours=(0, 6, 12, 18))
    _write_day(data, "2024-03-02", descending=True, hours=tuple(range(24)))
    assert H.hour_index("2024-03-01", 12) == 2 and H.hour_index("2024-03-02", 12) == 12
    truth = H.Truth()
    for day in ("2024-03-01", "2024-03-02"):
        for h in (0, 6, 12, 18):
            assert np.allclose(truth.get(np.datetime64(f"{day}T{h:02d}:00"), "t850"), _value("t", 850, h))


@dataclasses.dataclass
class _Meta:
    time: tuple
    lat: torch.Tensor = None
    lon: torch.Tensor = None


@dataclasses.dataclass
class _Pred:
    surf_vars: dict
    atmos_vars: dict
    metadata: _Meta


def _pred_at(day, hour):
    surf = {k: torch.full((1, 1, NY, NX), 1000.0 * (i + 1) + 0.25 * hour) for i, k in enumerate(H.SURF)}
    atmos = {v: torch.stack([torch.full((NY, NX), _value(v, lev, hour)) for lev in H.LEVELS])[None, None]
             for v in ("z", "t", "u", "v", "q")}
    return _Pred(surf, atmos, _Meta(time=(np.datetime64(f"{day}T{hour:02d}:00").astype("datetime64[s]").tolist(),)))


def test_scoring_uses_each_steps_valid_time(data):
    _write_day(data, "2024-03-01", descending=True)
    lat = torch.linspace(60, -60, NY)
    preds = [_pred_at("2024-03-01", 12), _pred_at("2024-03-01", 18)]         # steps 1 and 2 from the 06 UTC state
    rows = H.score(preds, lat, H.Truth(), None)
    assert {r["step"] for r in rows} == {1, 2} and max(r["rmse_truth"] for r in rows) < 1e-3
    shifted = [_pred_at("2024-03-01", 6), _pred_at("2024-03-01", 12)]
    shifted[0].metadata = _Meta(time=preds[0].metadata.time)                  # values of 06 UTC labelled 12 UTC
    rows = H.score(shifted, lat, H.Truth(), None)
    assert min(r["rmse_truth"] for r in rows if r["step"] == 1) > 1.0


@dataclasses.dataclass
class _Batch:
    static_vars: dict
    surf_vars: dict
    atmos_vars: dict
    metadata: _Meta

    def crop(self, patch_size):
        return self


class _Model(torch.nn.Module):
    patch_size = 4

    def __init__(self, dtype):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2, dtype=dtype))

    def batch_transform_hook(self, batch):
        return batch


def test_inputs_follow_model_dtype_and_recipes_use_autocast():
    b = _Batch({"z": torch.rand(NY, NX, dtype=torch.float64)}, {"2t": torch.rand(1, 2, NY, NX) * 300},
               {"t": torch.rand(1, 2, 13, NY, NX) * 300}, _Meta(time=(None,), lat=torch.linspace(90, -90, NY),
                                                                 lon=torch.linspace(0, 359.75, NX)))
    p = prepare(_Model(torch.float32), b)
    assert p.surf_vars["2t"].dtype == torch.float32 and p.atmos_vars["t"].dtype == torch.float32
    assert p.metadata.lat.dtype == torch.float32 and p.metadata.lon.dtype == torch.float32
    for f in sorted((ROOT / "recipes").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(r, dict) or "name" not in r:
            continue
        m = merged(r)
        assert m["precision"] == "autocast", f.name                 # no recipe selects the full-BF16 cast
        assert not m.get("surface_fp32"), f.name
