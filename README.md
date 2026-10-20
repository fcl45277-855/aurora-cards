# aurora-cards: NVFP4 quantisation of Aurora

Research code for studying how 4-bit quantisation affects [Microsoft Aurora](https://github.com/microsoft/aurora): forecast accuracy, attention sensitivity, physical consistency and inference cost.

This repository accompanies **NVFP4 quantisation of Aurora: attention sensitivity, forecast consistency and deployment**, by Y. Guan, J. Li, L. Lu, G. Pan, J. Song and Q. Hou, prepared for submission to *Computers & Geosciences*.

## What the repository provides

- **Quantisation and evaluation.** Block-format NVFP4 and MXFP8 simulators, GPTQ, SmoothQuant scaling, low-rank branches, experiment recipes and rollout scoring. Evaluation separates error relative to full-precision Aurora from forecast error against ERA5, and includes block-bootstrap confidence intervals, lead-time loss and physical-consistency diagnostics.
- **The `t7_mx` recipe.** NVFP4 weights and MXFP8 activations, with SmoothQuant scaling and a rank-32 branch; GPTQ is applied to 138 of Aurora's 314 linear layers. [cards/deployable.py](cards/deployable.py) exposes the layer construction for use with other PyTorch models. Its return value is a simulated quantised layer.
- **An attention diagnostic.** [cards/switching.py](cards/switching.py) replays attention under a quantisation-induced key perturbation and measures the share of output error associated with a change in the highest-logit key. The paper evaluates its ability to rank module sensitivity across three Aurora variants; independent architectures have not yet been validated here.
- **Hardware experiments.** Native NVFP4 builds with TorchAO/mslk on Blackwell GPUs, plus timing and profiling scripts. These experiments are separate from the simulated `t7_mx` activation path.

**Simulation and hardware coverage:** NVFP4 uses E2M1 values, a UE4M3 scale per 16 elements and an FP32 tensor scale. The simulator is checked for bit-exact agreement with TorchAO's `NVFP4Tensor`. MXFP8 uses E4M3 values and a power-of-two scale per 32 elements. In the native comparisons reported in Sect. 4.6, simulated and native results agree within 0.2% on average. The best `t7_mx` path uses simulated MXFP8 activations: no native NVFP4 × MXFP8 kernel was evaluated with TorchAO 0.18. It should not be interpreted as a demonstrated whole-model speedup.

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

The self-test checks agreement with TorchAO when it is installed. The tutorial check executes the examples in [the tutorial](docs/TUTORIAL.md). The two standalone examples demonstrate layer construction and the attention diagnostic on small synthetic inputs; they do not establish forecast accuracy for another model.

To reconstruct Tables 2, 3 and S10 and the Sect. 4.5 comparisons from the released summaries:

```bash
python scripts/paper_tables.py
```

This reads existing results; it does not rerun the forecasts.

### Run one forecast experiment on a GPU

After configuring a [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) account, run:

```bash
bash examples/03_one_date.sh
```

The script downloads the checkpoint and required ERA5 days, then compares the reference, naive quantisation and a `t7_mx` example on one test date. It calibrates on one day, rather than the paper's full calibration split, so its output need not match the paper's averages. Allow roughly 30–40 minutes including downloads; transfer and CDS queue times vary. The script uses Bash and needs a GPU with at least 24 GB of memory under the chunked configuration below.

For complete experiments, follow [the reproduction guide](docs/REPRODUCE.md), which maps each table and figure to its scripts, recipes and data splits.

## Data and checkpoints

Set `AQ_CKPT_DIR` to the checkpoint directory and `AQ_ERA5_DIR` to the prepared ERA5 directory before running model experiments.

| Asset | Source and preparation |
| --- | --- |
| Aurora checkpoints | Download `aurora-0.25-pretrained.ckpt`, `aurora-0.25-finetuned.ckpt` and `aurora-0.25-small-pretrained.ckpt` as needed from [Microsoft's Aurora model repository](https://huggingface.co/microsoft/aurora). |
| ERA5 inputs | Use `scripts/download_static.py` and `scripts/download_era5_day.py`. [docs/DATA.md](docs/DATA.md) specifies the variables, dates, split definitions and storage requirements. |
| Released summaries | [results_summary/](results_summary/) contains the compact records used to reconstruct the paper's tables and aggregate diagnostics. |
| Full result archive | Per-date records, diagnostic fields and figure maps are being prepared for Zenodo deposition. The data DOI is pending. |

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
| [docs/](docs/) | [User guide](docs/USER_GUIDE.md), [tutorial](docs/TUTORIAL.md), [reproduction guide](docs/REPRODUCE.md) and [data specification](docs/DATA.md) |

## AI-assisted development

Claude Code (Anthropic) assisted development of this repository. The authors manually reviewed and verified the AI-assisted code and take responsibility for the final implementation.

## Citation and licence

See [CITATION.cff](CITATION.cff) for the software and manuscript metadata. Please also cite the original [Aurora paper](https://doi.org/10.1038/s41586-025-09005-y) and [ERA5 paper](https://doi.org/10.1002/qj.3803) when using those resources.

Repository code is released under the [MIT licence](LICENSE). Aurora code and weights remain subject to Microsoft's terms, and ERA5 to the applicable Copernicus licence; see [the data specification](docs/DATA.md) for the source links.

Contact: Yue Guan, yue3.24@intl.zju.edu.cn
