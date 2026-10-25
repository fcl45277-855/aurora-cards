"""ERA5 1991-2020 climatology for the anomaly correlation coefficient (Table 9, Table S17c).

Monthly means by hour of day (00, 06, 12, 18 UTC) for March and October, 1991-2020, on the 1-degree grid
(CDS interpolates the 0.25-degree data; the 1-degree points coincide with every fourth 0.25-degree point), averaged
over the 30 years. Output: data_clim/era5_clim_1991_2020_1deg.npz with arrays <var>_m<MM>_h<HH> of shape (181, 360),
latitude 90..-90, longitude 0..359.

  python scripts/clim_download.py          (needs ~/.cdsapirc and the CDS licences accepted)
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_clim"
OUT.mkdir(exist_ok=True)
YEARS = [str(y) for y in range(1991, 2021)]
MONTHS = ["03", "10"]
HOURS = ["00:00", "06:00", "12:00", "18:00"]
REQ = {  # our name: (dataset, CDS variable, pressure level or None, netcdf short name)
    "z500": ("reanalysis-era5-pressure-levels-monthly-means", "geopotential", "500", "z"),
    "t850": ("reanalysis-era5-pressure-levels-monthly-means", "temperature", "850", "t"),
    "2t": ("reanalysis-era5-single-levels-monthly-means", "2m_temperature", None, "t2m"),
    "msl": ("reanalysis-era5-single-levels-monthly-means", "mean_sea_level_pressure", None, "msl"),
}


def main():
    import cdsapi
    c = cdsapi.Client()
    arrays = {}
    for var, (ds, cvar, lev, short) in REQ.items():
        target = OUT / f"raw_{var}.nc"
        if not target.exists():
            req = {"product_type": ["monthly_averaged_reanalysis_by_hour_of_day"], "variable": [cvar],
                   "year": YEARS, "month": MONTHS, "time": HOURS, "grid": ["1.0", "1.0"],
                   "data_format": "netcdf", "download_format": "unarchived"}
            if lev:
                req["pressure_level"] = [lev]
            print("requesting", var, flush=True)
            c.retrieve(ds, req, str(target))
        d = xr.open_dataset(target)
        da = d[short]
        if "pressure_level" in da.dims:
            da = da.isel(pressure_level=0)
        tdim = "valid_time" if "valid_time" in da.dims else "time"
        t = da[tdim].values
        months = (t.astype("datetime64[M]").astype(int) % 12) + 1
        hours = ((t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int)
        lat = da["latitude"].values
        assert lat[0] > lat[-1] and da.shape[-2:] == (181, 360), (lat[:3], da.shape)
        for m in (3, 10):
            for h in (0, 6, 12, 18):
                sel = (months == m) & (hours == h)
                assert sel.sum() == 30, (var, m, h, sel.sum())
                arrays[f"{var}_m{m:02d}_h{h:02d}"] = da.values[sel].mean(0).astype(np.float32)
        d.close()
        print("done", var, flush=True)
    np.savez(OUT / "era5_clim_1991_2020_1deg.npz", **arrays)
    print("wrote", OUT / "era5_clim_1991_2020_1deg.npz", len(arrays), "arrays")


if __name__ == "__main__":
    sys.exit(main())
