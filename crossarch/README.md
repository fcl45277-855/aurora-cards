# Cross-architecture check: ClimaX and FourCastNet


Code and result summaries for Sect. 4.3 and Supplementary Sect. S13. We test whether Aurora's sensitivity patterns and switching predictor transfer to two independently developed weather models:

- **ClimaX** ([Nguyen et al., 2023](https://proceedings.mlr.press/v202/nguyen23a.html)). Variable aggregation uses one learned query to attend over variable tokens, providing a structural comparison with Aurora's Perceiver resamplers.
- **FourCastNet** ([Pathak et al., 2022](https://arxiv.org/abs/2202.11214)). An AFNO model without softmax attention, used as an architectural contrast.

[PROTOCOL.md](PROTOCOL.md) records the hypotheses, criteria, dates and two revisions, fixed before either model's quantised runs. This was an internal protocol, not a public preregistration.

## Results

| Model                                  | Most sensitive component                          | Concentration of damage                                      | Attention regime                                                    | err_in_flip ρ         |
| -------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- | --------------------- |
| Aurora (paper, three variants)         | two Perceiver resamplers                          | resamplers (1 % of weights) dominate                         | sharp: 73 % of cells; switched cells carry ≥ 94 % of the error      | +0.95 / +0.98 / +0.90 |
| ClimaX 5.625° (our 6-h ERA5 fine-tune) | variable aggregation `var_agg` (1 query, 38 keys) | rank 1 of 9 in March and July; 4.8× the next module in March | diffuse: no sharp cells, 1.4 % switched, linear response (G = 1.00) | −0.52                 |
| FourCastNet 0.25°                      | patch embedding, then output head                 | top 1 / 3 of 26 groups: 43 / 71 %                            | no softmax attention                                                | not defined           |

The 73 % sharp-cell share refers to Aurora's pretrained decoder replay. FourCastNet's shares are descriptive; isolated-module damage is non-additive.

- **H1 is supported.** The variable aggregation is the ClimaX hotspot.
- **H2 is not supported.** The switching predictor does not rank ClimaX's attention modules.
- **Exploratory decomposition.** Activation-only quantisation reproduces 95 % of `var_agg`'s W4A4 damage; quantising K/V token inputs alone reproduces about two thirds. These effects are non-additive.

Sensitivity recurs at field–latent interfaces, but the switching predictor fails in ClimaX. Cross-architecture records are in [results_summary/](results_summary/).

| Folder                             | Contents                                                                                                                                                                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results_summary/climax/`          | per-split damage and replay records (`climax_*_W4A4.json`, `climax_confirmation_W4A32.json`); the frozen January predictor; the fixed-criterion analysis (`climax_analysis.json`); the exploratory decomposition (`explore_var_agg.json`); the identity-equivalence diagnostic and the full-model check |
| `results_summary/climax_finetune/` | fine-tuning provenance, per-epoch history and log, the evaluation config, the CPU smoke-test log                                                                                                                                                                                                        |
| `results_summary/fourcastnet/`     | per-split damage records and descriptive concentration analyses                                                                                                                                                                                                                                         |

Paths inside these files are written as `$CROSSARCH_ROOT`, `$AQ_ERA5_DIR` and `<repo>`. The SHA-256 values they record refer to the original files.

## Requirements

Use the repository's [requirements.txt](../requirements.txt), plus `numcodecs` for reading WeatherBench2 Zarr chunks. Fine-tuning and GPU evaluation used one RTX 5090:

| Step                   | Time on the RTX 5090 |
| ---------------------- | -------------------- |
| Fine-tuning, 20 epochs | 3.3 min per epoch    |
| ClimaX evaluation      | under 10 s per split |
| FourCastNet evaluation | under 10 s per split |

Disk needs:

- 3.2 GB for the 5.625° training states;
- 0.4 GB for the ClimaX checkpoint;
- 0.9 GB for the original FourCastNet checkpoint;
- about 2.8 GB for the FourCastNet inputs.

The CPU steps need about 2 GB of memory.

Set two environment variables:

```bash
export CROSSARCH_ROOT=/path/to/crossarch_work   # created by the scripts
export AQ_ERA5_DIR=/path/to/aurora_era5         # Aurora's ERA5 inputs, see docs/DATA.md
```

## Reproduction

Run the scripts from `crossarch/scripts/`. Every stage verifies the SHA-256 of what it reads and refuses to overwrite existing results.

1. **Sources and checkpoints (CPU, network).**
   - `bootstrap_cpu.py` captures the ClimaX (commit 6d5d354) and FourCastNet (commit 93360c1) sources.
   - `climax_range_download.py` downloads the CMIP6-pretrained 5.625° ClimaX checkpoint.
   - `fourcastnet_range_download.py` and `assemble_fcn_inference.py` download the original NERSC FourCastNet checkpoint and write a weights-only inference file.
   - `verify_checkpoint_cpu.py --model climax` and `--model fourcastnet` load both models strictly.
   - `cpu_preflight.py` runs the implementation checks of the replay adapter.
2. **FourCastNet inputs (CPU, CDS account).**
   - `prepare_era5_extras.py --month <YYYY-MM> --fcn-only` fetches the four ERA5 fields that Aurora's inputs lack (sp, r500, r850, tcwv).
   - `prepare_fcn_inputs.py <dates>` assembles the 20-channel inputs from `$AQ_ERA5_DIR`. `prepare_remaining_data.py` runs both steps for all fixed dates.
   - `validate_inputs_cpu.py` checks channels, units, timestamps and normalisation.
3. **ClimaX training data and test inputs (CPU, network).**
   - `prepare_training.py --workers 8` freezes the task plan, prepares conservatively regridded WeatherBench2 ERA5 states (2010–2016, 64 × 32), and computes training-only normalisation statistics. Downloads are resumable.
   - `fetch_wb2_regridding.py` pins WeatherBench2's conservative-regridding code.
   - `prepare_holdout.py` regrids the 2024 test dates from `$AQ_ERA5_DIR`.
4. **CPU checks of the fine-tuning code.**
   - `train_climax_era5.py --smoke` runs a tiny model on synthetic states. It checks the loss, a checkpoint reload and exact resume.
   - `train_climax_era5.py --check` checks the frozen plan, data and weights.
   - `test_make_config.py` tests the config writer.
1. **ClimaX (GPU).** `bash run_gpu_queue.sh` runs these steps in order:
   - the full-model check;
   - fine-tuning, with the frozen persistence gate;
   - writing the evaluation config;
   - `run_climax.py` for the plumbing date and the January predictor dates;
   - `freeze_predictor.py`;
   - `run_climax.py` for the March confirmation dates and the July season dates (W4A4), and for March under W4A32;
   - `analyze_climax.py`, which reports the fixed H1/H2 criteria.
6. **FourCastNet (GPU).** Run each split, then the descriptive analysis:

   ```bash
   python run_fourcastnet.py $CROSSARCH_ROOT/data/fourcastnet/2024-03-{01..12}.npz --output $CROSSARCH_ROOT/results/fourcastnet_confirmation_W4A32.json
   python analyze_fourcastnet.py $CROSSARCH_ROOT/results/fourcastnet_confirmation_W4A32.json --output $CROSSARCH_ROOT/results/fourcastnet_confirmation_analysis.json
   ```

   Repeat for the numerical-consistency date (2024-02-02) and July dates, using separate output files for each split.
7. **Exploratory and diagnostic scripts.** These analyses fall outside the fixed H1/H2 tests:
   - `explore_var_agg.py <config>` decomposes the `var_agg` damage;
   - `diag_equivalence.py <config>` compares the unquantised replay with the original modules (protocol v1.2).

`train_climax_era5.py` loads ClimaX's loss from `climax_upstream_metrics.py`, a verbatim copy of `src/climax/utils/metrics.py` at commit 6d5d354. That file is © Microsoft, MIT licence. ClimaX and FourCastNet sources and weights are downloaded from their official locations and remain under their own licences.
