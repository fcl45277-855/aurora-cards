# aurora-cards: NVFP4 quantisation of Aurora

Research code for studying how low-precision post-training quantisation (NVFP4, MXFP8 and FP8) affects [Microsoft Aurora](https://github.com/microsoft/aurora): forecast accuracy, attention-module sensitivity, physical consistency and inference cost.

This repository accompanies **NVFP4 quantisation of Aurora: attention-module sensitivity, forecast consistency and hardware evaluation**, by Y. Guan, J. Li, L. Lu, G. Pan, J. Song and Q. Hou, prepared for submission to *Computers & Geosciences*. Version 1.1.0 matches manuscript version v16/v17.

## What the repository provides

- **Quantisation and evaluation.** Block-format NVFP4 and MXFP8 simulators, GPTQ, SmoothQuant scaling, low-rank branches, experiment recipes and rollout scoring. Evaluation separates error relative to full-precision Aurora from forecast error against ERA5, and includes block-bootstrap confidence intervals, lead-time loss and physical-consistency diagnostics.
- **The recommended `t7_a16` recipe.** NVFP4 weights (SmoothQuant 0.8, a rank-32 BF16 branch and GPTQ) with BF16 activations on 138 of Aurora's 314 linear layers; the resamplers, the state-independent layers and 60 inherited Swin and patch layers stay in high precision. It runs with TorchAO's native W4A16 kernel. `t7_mx` uses the same weights with MXFP8 activations and serves as a W4A8 feasibility example. FP8 baselines (uniform and with the resamplers protected) are in `recipes/u8_*`. [cards/deployable.py](cards/deployable.py) exposes the layer construction for use with other PyTorch models; its return value is a simulated quantised layer.
- **An attention diagnostic.** [cards/switching.py](cards/switching.py) replays attention under a quantisation-induced key perturbation and measures the share of output error associated with a change in the highest-logit key. The paper evaluates its ability to rank module sensitivity across three Aurora variants. In an independent architecture, ClimaX, the analogous variable-aggregation module is again the most sensitive, but its attention is diffuse and the diagnostic does not rank the modules ([crossarch/](crossarch/)); treat it as specific to few-key, sharply attended cross-attention.
- **Cross-architecture check.** [crossarch/](crossarch/) quantises ClimaX (fine-tuned by us on ERA5 at 5.625°) and FourCastNet module by module under a pre-run protocol, with code, protocol and result summaries.
- **Hardware experiments.** Native NVFP4 and FP8 builds with TorchAO/MSLK on Blackwell GPUs, timing against FP32- and BF16-resident references, and profiling scripts. The MXFP8 activation path of `t7_mx` has no native kernel and stays simulated.

**Simulation and hardware coverage:** NVFP4 uses E2M1 values, a UE4M3 scale per 16 elements and an FP32 tensor scale; MXFP8 uses E4M3 values and a power-of-two scale per 32 elements. For RTN weights the simulator agrees element by element with TorchAO's `NVFP4Tensor`; GPTQ weights are re-quantised by TorchAO with slightly different block scales (Sect. 4.7 of the paper). In the native comparisons of Sect. 4.7, simulated and native forecasts agree within 0.5 % on average. Native execution did not accelerate whole-model inference, and the unfused native `t7_a16` build is slower than the BF16 reference.

## Installation

The paper environment uses Python 3.12 and PyTorch 2.12.1; [requirements.txt](requirements.txt) records the package versions. A CPU build of PyTorch is sufficient for the data-free tests, the first two examples and reading the released result tables. Aurora model runs require a CUDA GPU; the measured memory use is listed below.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/fcl45277-855/aurora-cards.git
cd aurora-cards
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install the PyTorch build appropriate for your machine, following the comments in `requirements.txt`, then run:

```bash
python -m pip install -r requirements.txt
python scripts/check_env.py
```

The environment check reports package versions, compares the installed Aurora source with [the paper's source checksums](docs/aurora_package_md5.txt), and checks the configured checkpoint and data directories. Missing model assets are expected if you only intend to run the CPU examples.

Native NVFP4 scripts (`scripts/native_*.py`) additionally require a GPU with compute capability ≥ 12.0 and the paper's mslk build:

```bash
python -m pip install mslk==1.2.0 --no-deps --index-url https://download.pytorch.org/whl/cu130
```

## Quick start

### Check the implementation on a CPU

These commands require neither Aurora checkpoints nor ERA5 data:

```bash
python -m pytest tests -q
python tests/selftest_check.py
python tests/tutorial_check.py
python examples/01_deployable_layer.py
python examples/02_switching_diagnostic.py
```

The bit-exact comparison with TorchAO runs only when TorchAO is installed; otherwise pytest reports it as skipped (`-rs` shows the reason), not as passed. The same tests run on every push in the CPU workflow [.github/workflows/cpu-tests.yml](.github/workflows/cpu-tests.yml), which also checks the paper tables with `scripts/paper_tables.py`. The tutorial check executes the examples in [the tutorial](docs/TUTORIAL.md). The two standalone examples demonstrate layer construction and the attention diagnostic on small synthetic inputs; they do not establish forecast accuracy for another model.

To reconstruct Tables 2, 3, 4, 6 and 8 and Table S10 and the Sect. 4.6 comparisons from the released summaries:

```bash
python scripts/paper_tables.py
```

This reads existing results; it does not rerun the forecasts.

### Run one forecast experiment on a GPU

After configuring a [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) account, run:

```bash
bash examples/03_one_date.sh
```

The script downloads the checkpoint and required ERA5 days, then compares the reference, naive quantisation and a `t7_mx` example on one test date (replace the recipe by `t7_a16` for the recommended configuration). It calibrates on one day, rather than the paper's full calibration split, so its output need not match the paper's averages. Allow roughly 30–40 minutes including downloads; transfer and CDS queue times vary. The script uses Bash and needs a GPU with at least 24 GB of memory under the chunked configuration below.

For complete experiments, follow [the reproduction guide](docs/REPRODUCE.md), which maps each table and figure to its scripts, recipes and data splits.

## Data and checkpoints

Set `AQ_CKPT_DIR` to the checkpoint directory and `AQ_ERA5_DIR` to the prepared ERA5 directory before running model experiments.

| Asset | Source and preparation |
| --- | --- |
| Aurora checkpoints | Download `aurora-0.25-pretrained.ckpt`, `aurora-0.25-finetuned.ckpt` and `aurora-0.25-small-pretrained.ckpt` as needed from [Microsoft's Aurora model repository](https://huggingface.co/microsoft/aurora). |
| ERA5 inputs | Use `scripts/download_static.py` and `scripts/download_era5_day.py`. [docs/DATA.md](docs/DATA.md) specifies the variables, dates, split definitions and storage requirements. |
| Released summaries | [results_summary/](results_summary/) contains per-card summaries (`summary.json`, `recipe.json`, `build_info.json`), timing files and aggregate diagnostics (gzip-compressed under `_aggregates/diag/`). From these, Tables 2–8 and S10, Figs. 1, 2, 4 and 5 and the timing values can be rebuilt offline. Intervals printed in `summary.json` are i.i.d. date-bootstrap intervals; the paper's 3-day block-bootstrap intervals, paired differences and lead-time loss need the per-record files of the full archive. |
| Full result archive | Per-record results (`records.jsonl`), the 2° error maps of Fig. 3 and the reference caches are being prepared for Zenodo deposition; the data DOI is pending. |

The reference follows Aurora's own precision policy: a BF16 backbone under autocast, with FP32 encoder, decoder and input/output operations. Quantised layers replace selected `torch.nn.Linear` modules within that reference. Consult [the user guide](docs/USER_GUIDE.md) for the exact metrics, layer selections, inputs and outputs.

## Computational requirements

| Resource | CPU tests, examples 1–2 and summary tables | Aurora model experiments |
| --- | --- | --- |
| GPU | None | At least 24 GB recommended. Measured peak: 13–15 GiB with `AQ_PERCEIVER_CHUNK=16384`, versus 24–27 GiB without chunking. Native NVFP4 kernels additionally require Blackwell compute capability ≥ 12.0. |
| Host memory | About 4 GB | About 16 GB for a single card; about 32 GB for the 20-step diagnostics that retain reference and quantised fields. |
| Working disk space | About 20 MB for repository files, excluding the Python environment | About 5 GB per full-size checkpoint and 1.15 GB per ERA5 day. Full reproduction: about 85 GB ERA5, 15 GB checkpoints and 12 GB reference caches/results. |
| Runtime | Seconds to about one minute | About 3–15 minutes per 30-date card, plus 30 seconds for RTN or 4–6 minutes for GPTQ construction. Full reproduction is estimated at about 40 GPU hours; individual tables take roughly 0.5–15 GPU hours. |

GPU memory use and per-card times were measured on an RTX 5090 with 32 GB of memory. Host memory, disk use and total reproduction time are estimates. These figures exclude installation overhead and do not guarantee the same performance on other systems. Use the versions in `requirements.txt`; spectral diagnostics also require pyshtools, and native kernels require mslk.

## Repository guide

| Path | Contents |
| --- | --- |
| [cards/](cards/) | Quantisers, GPTQ, rotation, recipe construction, rollout/scoring, reusable layer construction and attention diagnostics |
| [recipes/](recipes/) | JSON experiment configurations ("cards"), plus the layer manifests |
| [scripts/](scripts/) | Data downloads, experiments, diagnostics, native builds, figures and tables |
| [tests/](tests/) | Data-free unit tests and script-style checks |
| [examples/](examples/) | Two CPU examples and one GPU forecast example |
| [results_summary/](results_summary/) | Released per-card summaries and aggregate diagnostics for the pretrained, finetuned and small models |
| [crossarch/](crossarch/) | ClimaX and FourCastNet cross-architecture check: protocol, scripts, result summaries |
| [docs/](docs/) | [User guide](docs/USER_GUIDE.md), [tutorial](docs/TUTORIAL.md), [reproduction guide](docs/REPRODUCE.md) and [data specification](docs/DATA.md) |

## AI-assisted development

Claude Code (Anthropic) assisted development of this repository. The authors manually reviewed and verified the AI-assisted code and take responsibility for the final implementation.

## Citation and licence

See [CITATION.cff](CITATION.cff) for the software and manuscript metadata. Please also cite the original [Aurora paper](https://doi.org/10.1038/s41586-025-09005-y) and [ERA5 paper](https://doi.org/10.1002/qj.3803) when using those resources.

Repository code is released under the [MIT licence](LICENSE). Aurora code and weights remain subject to Microsoft's terms, and ERA5 to the applicable Copernicus licence; see [the data specification](docs/DATA.md) for the source links.

Contact: Qingchun Hou (corresponding author), qingchunhou@intl.zju.edu.cn. Developers: Yue Guan and Guyang Pan; Guyang Pan carried out the main code revisions for version 1.1.
