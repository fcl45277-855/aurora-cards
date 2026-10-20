# aurora-cards: 4-bit inference for the Aurora weather foundation model

Code for the paper **"Four-bit inference for a weather foundation model: where NVFP4 quantisation of Aurora fails, why, and how to deploy it safely"** (Y. Guan, J. Li, L. Lu, Q. Hou; submitted to *Computers & Geosciences*).

The repository contains a quantisation harness for [Microsoft Aurora](https://github.com/microsoft/aurora) 0.25° (1.3 B parameters) and every script behind the paper's tables and figures:

- **Simulated block floating-point quantisation**: NVFP4 (E2M1 codes, UE4M3 scale per 16 elements, FP32 per-tensor scale) and MXFP8 (E4M3, power-of-two scale per 32 elements). The NVFP4 simulation is bit-exact against TorchAO's `NVFP4Tensor`.
- **The deployable recipe t7_mx**: SmoothQuant scaling, a rank-32 low-rank branch and GPTQ on 138 of Aurora's 314 linear layers, with NVFP4 weights and MXFP8 activations. It is available as a stand-alone function for any `torch.nn.Linear` in `cards/deployable.py`.
- **The switching-dominance diagnostic** in `cards/switching.py`. It replays an attention module under its quantisation-induced key perturbation and reports which share of the output error is carried by cells whose argmax key switches. This share predicts the sensitivity of attention modules across three Aurora variants.
- **Documentation**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md) (inputs, outputs, options, expected behaviour), [docs/TUTORIAL.md](docs/TUTORIAL.md) (three typical use cases), [docs/REPRODUCE.md](docs/REPRODUCE.md) (every table and figure), [docs/DATA.md](docs/DATA.md).
- **Evaluation**: latitude-weighted RMSE against ERA5 and against full-precision Aurora, block-bootstrap confidence intervals, lead-time loss, physical-consistency diagnostics and the PhysMetrics.Weather metrics.
- **On-device builds** with TorchAO/mslk NVFP4 kernels on Blackwell GPUs, plus step timing and profiling.

## Install

Python 3.12. Model runs need an NVIDIA GPU with at least 24 GB (see *Computational requirements*); the tests, examples 1–2 and the table check run on a CPU.

```bash
git clone <this repository> && cd aurora-cards
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # CPU or CUDA build of torch, see requirements.txt
python scripts/check_env.py
```

The native NVFP4 runs (`scripts/native_*.py`) also need `mslk` and a GPU of compute capability ≥ 12.0. Install mslk with `pip install mslk==1.2.0 --no-deps --index-url https://download.pytorch.org/whl/cu130`.

## Quick start

Each level needs more than the one before it:

| Level | What it shows | Needs | Time |
| --- | --- | --- | --- |
| 1 | `python -m pytest tests -q` (quantisers, GPTQ, deployable layer, switching diagnostic), `python tests/selftest_check.py` (bit-exactness against TorchAO when installed) and `python tests/tutorial_check.py` (runs the tutorial code) | CPU | < 1 min |
| 2 | `python examples/01_deployable_layer.py` and `python examples/02_switching_diagnostic.py` | CPU | seconds |
| 3 | `python scripts/paper_tables.py`: Tables 2, 3, S10 and Sect. 4.5 from the released result summaries, checked against the paper | CPU | seconds |
| 4 | `bash examples/03_one_date.sh`: download checkpoint and data, run FP32, naive and deployable quantisation on one date | GPU ≥ 24 GB, CDS account | 30–40 min |
| 5 | Full re-run of any table or figure, see [docs/REPRODUCE.md](docs/REPRODUCE.md) | GPU, ERA5 subsets | 0.5–15 GPU h per table |

## Computational requirements

| Resource | Tests, examples 1–2, tables from summaries | One card (model run) | Full reproduction |
| --- | --- | --- | --- |
| GPU | none | NVIDIA, ≥ 24 GB (peak 13–15 GiB with `AQ_PERCEIVER_CHUNK=16384`, 24–27 GiB without) | same; compute capability ≥ 12.0 (Blackwell) only for the native NVFP4 kernels |
| Host memory | 4 GB | 16 GB (the FP32 checkpoint is loaded in CPU memory first) | about 32 GB for the 20-step diagnostics, which hold one date's FP32 and quantised fields (estimated from the field sizes) |
| Disk | 20 MB | 5 GB checkpoint, about 1.15 GB per ERA5 day | about 85 GB ERA5 (all splits), 15 GB checkpoints, about 12 GB FP32 reference caches and results |
| Time | seconds to 1 min | 3–15 min per 30-date card plus 30 s (RTN) or 4–6 min (GPTQ) to build | about 40 GPU hours (estimate) for every table and figure on one RTX 5090 |
| Software | Python 3.12, PyTorch ≥ 2.11 (CPU build is enough) | PyTorch 2.12 with CUDA, TorchAO 0.18, the Aurora package | plus pyshtools (spectra) and mslk 1.2 (native kernels) |

GPU memory and times were measured on a cloud instance with one RTX 5090 (32 GB); host memory, disk and the total time are estimates from the data sizes and per-card times. The full list of versions is in `requirements.txt`. `scripts/check_env.py` checks the installation.

## Data and checkpoints

- **Aurora checkpoints**: `aurora-0.25-pretrained.ckpt`, `aurora-0.25-finetuned.ckpt` and `aurora-0.25-small-pretrained.ckpt` from [huggingface.co/microsoft/aurora](https://huggingface.co/microsoft/aurora). Set `AQ_CKPT_DIR` to their directory.
- **ERA5** (Copernicus Climate Data Store, CC-BY 4.0): `scripts/download_static.py` and `scripts/download_era5_day.py` write the files the harness reads into `AQ_ERA5_DIR`. [docs/DATA.md](docs/DATA.md) lists the dates of every split and the disk space they need.
- **Result archive** (Zenodo, DOI: *to be added*): full per-date records, diagnostic fields and maps behind every figure. The small summaries are already in `results_summary/`.

## Repository layout

```
cards/            harness: simulated quantisers (nvfp4.py, simlinear.py), GPTQ (gptq.py), rotation, recipe builder
                  (recipe.py), rollout and scoring (harness.py), deployable layer (deployable.py),
                  switching diagnostic (switching.py)
recipes/          one JSON file per experiment ("card"); manifest_314.json and q0_bf16_layers.txt define the layer sets
scripts/          experiments, diagnostics, on-device builds, figures, tables, data download
tests/            data-free unit tests (pytest) and script-style checks (*_check.py)
examples/         three runnable examples (see above)
results_summary/  summary.json / recipe.json / build_info.json of every card in the paper, plus aggregate
                  diagnostics, per model (pretrained, finetuned, small)
docs/             USER_GUIDE.md, TUTORIAL.md, REPRODUCE.md, DATA.md, aurora_package_md5.txt
```

## Notes

- All results use Aurora's own reduced-precision mode (`autocast`: BF16 backbone, FP32 encoder/decoder/IO). The quantised layers replace `torch.nn.Linear` modules of that model.
- The Aurora package version matters. `scripts/check_env.py` compares the installed package source with the version used for the paper (`docs/aurora_package_md5.txt`).
- Simulated and native results agree to within 0.2 % on average (paper Sect. 4.6). The MXFP8-activation path of t7_mx has no NVFP4 × MXFP8 kernel in TorchAO 0.18 and is simulated.

## Citation

See `CITATION.cff`. Please also cite Aurora (Bodnar et al., 2025, *Nature* 641, 1180–1187) and ERA5 (Hersbach et al., 2020, *Q. J. R. Meteorol. Soc.* 146, 1999–2049).

## Licence

Code: MIT (see `LICENSE`). Aurora weights and code are distributed by Microsoft under their own licence; ERA5 under the Copernicus licence (CC-BY 4.0).

Contact: Yue Guan, yue3.24@intl.zju.edu.cn
