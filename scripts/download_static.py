"""Download the ERA5 static fields Aurora needs (geopotential z, land-sea mask lsm, soil type slt at 0.25 deg) from the
Copernicus Climate Data Store and write <out>/static.nc, the file the harness reads.

    python scripts/download_static.py [--out $AQ_ERA5_DIR]

Needs a CDS account, ~/.cdsapirc and the licence of reanalysis-era5-single-levels accepted (see docs/DATA.md).
"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("AQ_ERA5_DIR", "data/era5"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    target = out / "static.nc"
    if target.exists():
        print("static.nc exists:", target); return
    import cdsapi
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": ["geopotential", "land_sea_mask", "soil_type"],
        "year": ["2023"], "month": ["01"], "day": ["01"], "time": ["00:00"],
        "data_format": "netcdf", "download_format": "unarchived"}, str(target))
    import xarray as xr
    s = xr.open_dataset(target)
    assert {"z", "lsm", "slt"} <= set(s.data_vars) and s.sizes["latitude"] == 721 and s.sizes["longitude"] == 1440, s
    print("wrote", target)


if __name__ == "__main__":
    main()
