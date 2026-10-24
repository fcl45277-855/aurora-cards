# Data

## Aurora checkpoints

| Model | File | Size | Source |
| --- | --- | --- | --- |
| Aurora 0.25° pretrained (all main results) | `aurora-0.25-pretrained.ckpt` | ≈ 5 GB | huggingface.co/microsoft/aurora |
| Aurora 0.25° finetuned (cross-model replication) | `aurora-0.25-finetuned.ckpt` | ≈ 5 GB | same |
| Aurora 0.25° small (cross-model replication) | `aurora-0.25-small-pretrained.ckpt` | ≈ 0.45 GB | same |

```bash
huggingface-cli download microsoft/aurora aurora-0.25-pretrained.ckpt --local-dir checkpoints
export AQ_CKPT_DIR=$PWD/checkpoints
```

## ERA5

The harness reads one pair of files per day from `AQ_ERA5_DIR`:

- `surface_YYYY-MM-DD.npz`: `t2m`, `u10`, `v10`, `msl` with shape [4, 721, 1440] at 00, 06, 12 and 18 UTC, plus `valid_time`.
- `air_YYYY-MM-DD.npz`: `z`, `q`, `t`, `u`, `v` with shape [4, 13, 721, 1440] on the levels 50 to 1000 hPa, plus `valid_time`.

It also needs `static.nc`, holding `z`, `lsm` and `slt` at 0.25°.

```bash
# one-off: create a CDS account, put the API key in ~/.cdsapirc, accept the licences of
# "reanalysis-era5-single-levels" and "reanalysis-era5-pressure-levels" on the CDS website
export AQ_ERA5_DIR=$PWD/data/era5
python scripts/download_static.py
python scripts/download_era5_day.py 2024-03-01 2024-03-02 ...
```

Each day takes about 1.15 GB on disk and a few minutes to download, depending on the CDS queue. The reader accepts pressure levels stored in either order; it detects the order from the geopotential.

### Dates used in the paper (`cards/harness.py`, `SPLITS`)

| Split | Dates | Days | Disk | Used for |
| --- | --- | --- | --- | --- |
| `calib` | 2024-01-04 .. 01-10 | 7 | 8 GB | calibration statistics (GPTQ Hessians, smoothing maxima) |
| `calib2` | 2024-01-18 .. 01-24 | 7 | 8 GB | calibration-week robustness |
| `dev` | 2024-02-02 .. 02-13 | 12 | 14 GB | all recipe choices (Table 3, mechanism, protection order) |
| `test` | 2024-03-01 .. 03-30 | 30 | 35 GB | held-out evaluation (Tables 4, 6 and 7; Figs. 2b and 3) |
| `summer` | 2024-07-01 .. 07-08 | 8 | 9 GB | season check, Hurricane Beryl case |
| `y2023` | 2023-03-01 .. 03-04 (+ 03-05, 03-06 for the 20-step truth) | 4 | 7 GB | cross-year check |
| `oct` | 2024-10-01 .. 10-14 (`scripts/download_era5_day.py`) | 14 | 16 GB | frozen confirmation of t7_a16 (criteria logged before download; Table S7b) |
| `calib1` | 2024-01-04 | 1 | 1.2 GB | examples only |

Each forecast starts from the 00 and 06 UTC states of a date. Step 1 is verified at 12 UTC and step 2 at 18 UTC of the same date; 20-step rollouts need the following five days as well. Days are scored only when their verifying ERA5 fields exist.

Licence: ERA5 is © ECMWF, distributed by the Copernicus Climate Change Service under CC-BY 4.0 (Hersbach et al., 2020).

## Result archive

The planned Zenodo archive (data DOI pending deposition) mirrors the `results/` layout written by the scripts. It holds per-card `records.jsonl` files with per-date, per-step and per-variable RMSE, the `_diag_fields_*` and `_physmetrics_*` files with the physical diagnostics, error maps (`*_maps.npz`), the replay, profiling and Beryl outputs, and the same `summary.json` files as `results_summary/`. Set `AQ_RESULTS` to the unpacked archive to redraw the figures with `scripts/paper_figures.py`.
