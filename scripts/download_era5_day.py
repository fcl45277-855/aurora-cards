"""Download one ERA5 day from the (new) CDS and write it in the harness's npz layout (same as scripts/cut_stream.py):
surface_<day>.npz: t2m, u10, v10, msl [4, 721, 1440] float32 + valid_time [4] (00, 06, 12, 18 UTC)
air_<day>.npz:     z, q, t, u, v   [4, 13, 721, 1440] float32, levels 50 -> 1000 hPa (harness.LEVELS order) + valid_time

  python scripts/download_era5_day.py 2024-07-01 [2024-07-02 ...] [--out $AQ_ERA5_DIR]
                                       [--tmp ./era5_tmp] [--keep-nc]

Needs ~/.cdsapirc and the two dataset licences accepted. One request per (day, dataset); CDS queueing dominates the
wall time, so run it under nohup and watch the log. Existing npz files are not overwritten (skipped with a message).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
HOURS = ["00:00", "06:00", "12:00", "18:00"]
SURF = {"t2m": "2m_temperature", "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
AIR = {"z": "geopotential", "q": "specific_humidity", "t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind"}


def retrieve(c, dataset: str, req: dict, target: Path, log) -> None:
    if target.exists() and target.stat().st_size > 1_000_000:
        log(f"  have {target.name}")
        return
    t0 = time.time()
    c.retrieve(dataset, req, str(target))
    log(f"  got {target.name}: {target.stat().st_size / 2**20:.0f} MiB in {time.time() - t0:.0f}s")


def to_npz(day: str, sfc: Path, pl: Path, out: Path, log) -> None:
    import xarray as xr
    ds = xr.open_dataset(sfc)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds.sortby(tdim)
    lat = ds["latitude"].values
    assert lat[0] > lat[-1] and len(lat) == 721 and ds.sizes["longitude"] == 1440, (lat[:2], dict(ds.sizes))
    vt = ds[tdim].values.astype("datetime64[ns]")
    assert len(vt) == 4, vt
    surf = {k: np.ascontiguousarray(ds[k].values, dtype=np.float32) for k in SURF}
    ds.close()
    da = xr.open_dataset(pl)
    tdim2 = "valid_time" if "valid_time" in da.dims else "time"
    ldim = "pressure_level" if "pressure_level" in da.dims else "level"
    da = da.sortby(tdim2).sortby(ldim)                                    # 50 -> 1000, ascending pressure
    assert [int(x) for x in da[ldim].values] == list(LEVELS), da[ldim].values
    assert (da[tdim2].values.astype("datetime64[ns]") == vt).all()
    assert da["latitude"].values[0] > da["latitude"].values[-1]
    air = {k: np.ascontiguousarray(da[k].values, dtype=np.float32) for k in AIR}
    da.close()
    for k, v in surf.items():
        assert v.shape == (4, 721, 1440) and np.isfinite(v).all(), (k, v.shape)
    for k, v in air.items():
        assert v.shape == (4, 13, 721, 1440) and np.isfinite(v).all(), (k, v.shape)
    zm = air["z"][0, :, ::120, ::360].mean((1, 2))
    assert zm[0] > zm[-1], "geopotential must decrease towards 1000 hPa"
    np.savez(out / f"surface_{day}.npz", **surf, valid_time=vt)
    np.savez(out / f"air_{day}.npz", **air, valid_time=vt)
    log(f"  wrote surface_{day}.npz / air_{day}.npz  (t2m mean {surf['t2m'].mean():.1f} K, msl mean {surf['msl'].mean():.0f} Pa, "
        f"z50 mean {zm[0]:.0f}, z1000 mean {zm[-1]:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+")
    ap.add_argument("--out", default=os.environ.get("AQ_ERA5_DIR", "data/era5"))
    ap.add_argument("--tmp", default="era5_tmp")
    ap.add_argument("--keep-nc", action="store_true")
    a = ap.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    import cdsapi
    c = cdsapi.Client(quiet=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    for day in a.days:
        if (out / f"surface_{day}.npz").exists() and (out / f"air_{day}.npz").exists():
            log(f"{day}: npz exist, skip")
            continue
        y, m, d = day.split("-")
        log(f"{day}: requesting")
        sfc, pl = tmp / f"{day}_sfc.nc", tmp / f"{day}_pl.nc"
        common = {"product_type": ["reanalysis"], "year": [y], "month": [m], "day": [d], "time": HOURS,
                  "data_format": "netcdf", "download_format": "unarchived"}
        retrieve(c, "reanalysis-era5-single-levels", {**common, "variable": list(SURF.values())}, sfc, log)
        retrieve(c, "reanalysis-era5-pressure-levels", {**common, "variable": list(AIR.values()),
                                                        "pressure_level": [str(l) for l in LEVELS]}, pl, log)
        to_npz(day, sfc, pl, out, log)
        if not a.keep_nc:
            sfc.unlink(missing_ok=True)
            pl.unlink(missing_ok=True)
    log("done")


if __name__ == "__main__":
    main()
