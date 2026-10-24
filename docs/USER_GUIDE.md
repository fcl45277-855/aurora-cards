# User guide

This guide describes the inputs, outputs and options of the harness and what to expect when it runs. For installation see `README.md`, for data see `docs/DATA.md`, for worked examples see `docs/TUTORIAL.md`, and for the commands behind each table see `docs/REPRODUCE.md`.

## 1. Concepts

- **Card**: one experiment. A card is a recipe run on one data split for a given number of 6-h steps. Its results go to `results/<recipe name>__<split>_h<steps>/`.
- **Recipe**: a JSON file in `recipes/`. It says which linear layers of Aurora are quantised and how.
- **Split**: a named list of initial dates, defined in `SPLITS` in `cards/harness.py`. Each forecast starts from the 00 and 06 UTC ERA5 states of a date.
- **Reference regime**: Aurora's own reduced-precision mode, with the backbone in BF16 under autocast and the encoder, decoder and input/output in FP32. Quantised layers replace `torch.nn.Linear` modules of this model.
- **FP32 reference**: the forecast of the unquantised FP32 model for the same date and step. Quantisation error is measured against it.

## 2. Inputs

### 2.1 Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `AQ_CKPT_DIR` | `checkpoints/` | Directory with the Aurora checkpoints |
| `AQ_ERA5_DIR` | `data/era5/` | Directory with `static.nc`, `surface_<date>.npz` and `air_<date>.npz` (see `docs/DATA.md`) |
| `AQ_MODEL` | `pretrained` | `pretrained`, `finetuned` or `small`. It selects the checkpoint and the default results directory (`results/`, `results_finetuned/` or `results_small/`) |
| `AQ_PERCEIVER_CHUNK` | `0` (off) | Runs the two Perceiver resamplers over chunks of this many patches. The value must be a multiple of 16384. With 16384, peak GPU memory drops from about 26 GiB to about 15 GiB, and results agree with the unchunked run up to GEMM round-off |
| `AQ_MEM_FRAC` | `0` (off) | Caps this process's share of GPU memory, for GPUs shared with other jobs |
| `AQ_RESULTS` | `results/` | Results directory that `scripts/paper_figures.py` reads; set it to the unpacked Zenodo archive to redraw the figures |
| `AQ_FIGURE_AUDIT` | unset | Optional folder of a panel-alignment audit helper used by `paper_figures.py` |

### 2.2 Recipe files

A recipe is a JSON object. Keys that are left out take the defaults below. Nested objects such as `weight` or `calib` are merged key by key with the defaults (`cards/recipe.py`, `merged`).

| Key | Default | Values and meaning |
| --- | --- | --- |
| `name` | (required) | Card name; also the name of the results directory |
| `kind` | `"sim"` | `"fp32"` builds the unquantised reference and writes the FP32 cache. `"bf16"` is the reference regime with no quantisation. `"sim"` applies simulated quantisation to `layers` |
| `layers` | `"q0"` | Layer set (Sect. 2.3) |
| `precision` | `"autocast"` | `"autocast"` is Aurora's official reduced-precision mode, used for every paper result. `"bf16_cast"` casts the whole model to BF16 and is a legacy option |
| `weight.method` | `"rtn"` | `"rtn"` is round-to-nearest NVFP4, `"gptq"` is GPTQ with NVFP4 rounding, `"none"` keeps BF16 weights so only activations are quantised |
| `weight.source` | `"bf16"` | Whether weights are quantised from their BF16 (`"bf16"`) or FP32 (`"fp32"`) values |
| `weight.percdamp` | `0.01` | GPTQ damping λ as a fraction of mean(diag H) |
| `weight.scale_mult` | `1.0` | Multiplier on the NVFP4 weight block scale |
| `weight.scale_search` | none | List of block-scale multipliers chosen per block by squared error (RTN only) |
| `weight.gptq_skip` | none | Substrings of layer names that use RTN even when `method` is `"gptq"` |
| `act` | `"nvfp4"` | Activation format: `"nvfp4"`, `"mxfp8"` (round-to-ceiling scales, deployable), `"mxfp8_floor"` (OCP floor scales), `"fp8"` (FP32 scale per 16 elements, an optimistic upper bound), `"fp8_tensor"`, `"fp8_token"`, `"fp8_tile128"`, `"nvfp4_sr"` (stochastic rounding) or `"bf16"` (weight-only) |
| `act_scale_search` | none | `[1.0, 1.5]` chooses the NVFP4 activation block scale per block by squared error. It acts only on layers whose activation format is `"nvfp4"` and is ignored for every other format; the field in `t7_mx.json` and `t7_a16.json` (MXFP8 and BF16 activations) therefore has no effect, and those recipes use no scale search |
| `act_step_mults` | none | NVFP4 activation scale multiplier per rollout step; overrides the search |
| `act_overrides` | `[]` | `[[substring, format], ...]` sets a per-layer activation format; the first match wins |
| `rotation.block` | `"full"` | Randomised Hadamard rotation of the layer input: `"full"`, a block width (16, 32, 64 or 128), or `null` for none |
| `rotation.dtype` | `"bf16"` | Precision of the rotation |
| `smooth` | none | `{"alpha": a}` applies SmoothQuant scaling s_j = amax(x_j)^a / max\|W_:,j\|^(1−a), paper Eq. (4) |
| `low_rank` | none | `{"rank": r}` adds a BF16 rank-r branch of the smoothed weight; the residual is quantised (Eq. 5) |
| `calib` | `{"split": "calib", "steps": 2, "budget_gib": 2.5}` | Dates and steps of the calibration pass. It is used only by GPTQ and smoothing. `budget_gib` bounds GPU memory for the statistics; layers are processed in groups |
| `scale_min` | `"tiny"` | Smallest NVFP4 block scale: `"tiny"` (2^-6, as TorchAO) or `"subnormal"` (2^-9). Stored per layer, so recipes built one after another in one process keep their own floor |
| `lowpass` | none | Scale-separated activation quantisation with a BF16 or FP8 coarse branch; a negative result in the paper |
| `kv_split`, `k_centre` | none | Mechanism probes for `to_kv` layers: only some output rows carry quantisation error, or keys are centred over the context tokens |

The recommended recipe of the paper, `recipes/t7_a16.json` (`t7_mx.json` is identical except `"act": "mxfp8"`):

```json
{"name": "t7_a16",
 "layers": "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed",
 "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": null},
 "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5], "act": "bf16"}
```

`act_scale_search` is kept in these files so that they match the recipes that produced the paper's results; it is inert
for BF16 and MXFP8 activations (see the table above). Built weights are cached in `results/_build_cache/`, keyed on the
weight-relevant recipe fields, the model and checkpoint, the rotation seed and a hash of the quantisation code
(`cards/recipe.py`, `CODE_FILES`); editing that code therefore triggers a rebuild.

### 2.3 Layer-set specifications

A layer set selects linear layers by name from the 314 non-head linear layers of Aurora 0.25° (`recipes/manifest_314.json`). The small model has 146 layers (`recipes/manifest_small.txt`).

| Specification | Selects |
| --- | --- |
| `all314` | All 314 layers (all 146 for the small model) |
| `q0` | The 195-layer mixed-precision set Q0: all layers except the 119 in `recipes/q0_bf16_layers.txt`. Not defined for the small model |
| `<set>:inc=a,b` | Layers of `<set>` whose name contains `a` or `b` |
| `<set>:exc=a,b` | Layers of `<set>` whose name contains neither `a` nor `b` |
| `a&b` inside a pattern | Both substrings must occur, e.g. `all314:inc=backbone.encoder_layers.1.&.attn.` |
| a file path | One layer name per line |

`python -c "from cards import harness as H; print(len(H.layer_set('<spec>')))"` prints the size of a set without loading the model.

## 3. Running

| Script | Purpose | Main options |
| --- | --- | --- |
| `scripts/run_card.py` | Build a recipe, roll it out on a split and score it | `--recipe`, `--split`, `--steps` (default 2), `--max-dates`, `--results` |
| `scripts/run_deck.sh` | Run several cards in order | `<split> <steps> <recipe> ...` |
| `scripts/block_ci.py` | 3-day moving-block bootstrap intervals and per-variable tables | `cards...`, `--split`, `--ref`, `--full` |
| `scripts/skill_hours.py` | Lead-time loss of 20-step cards | `cards...`, `--split`, `--steps` |
| `scripts/diag_fields.py` | Physical-consistency diagnostics computed online | `--recipes`, `--split`, `--steps`, `--dates`, `--tag` |
| `scripts/physmetrics_fields.py` | PhysMetrics.Weather metrics and KE(500) spectra | `--recipes`, `--split`, `--steps`, `--dates`, `--results` |
| `scripts/replay_attn.py`, `replay_swin.py` | Switching diagnostic replayed on the resamplers or the Swin windows | `--module`, `--dates` |
| `scripts/b_eval.py` | Rank predictors against per-module sensitivity | `--results`, `--diag` |
| `scripts/native_w4a16.py`, `native_t7.py`, `native_timing.py`, `profile_step.py` | On-device NVFP4 builds, timing and profile | see each file's header |
| `scripts/paper_tables.py`, `paper_figures.py` | Tables and figures of the paper from results | `--results` / `AQ_RESULTS` |
| `scripts/check_env.py` | Versions, Aurora package identity, data and checkpoint check | none |

Every script documents its usage at the top of the file; the scripts with options also accept `--help`.

**Resuming.** `run_card.py` appends one block of rows per date to `records.jsonl` and skips dates already present. An interrupted card therefore continues where it stopped. Delete the card directory to start a card again from scratch.

**Order.** The `fp32` card of a split must run before any other card on that split. It writes the FP32 reference cache `results/_ref_fp32_h<steps>/<date>_s<step>.npz`, and without it the quantised cards cannot compute `qerr`. A results directory belongs to one model: `results/_model.json` records which, and `run_card.py` refuses to mix models.

## 4. Outputs

### 4.1 Per card: `results/<name>__<split>_h<steps>/`

| File | Content |
| --- | --- |
| `recipe.json` | The recipe after merging with the defaults |
| `build_info.json` | `recipe`, `precision`, `layers` (per quantised layer: GPTQ diagnostics such as `proxy_gptq`, `proxy_rtn`, `gptq_loss`, `damp` when GPTQ is used), `proxy_total`, `build_seconds`, `perceiver_chunk`, `mem_frac` |
| `records.jsonl` | One JSON row per date, step and variable (fields below) |
| `summary.json` | Card-level means and confidence intervals (fields below) |

`records.jsonl` fields:

| Field | Meaning |
| --- | --- |
| `date`, `step`, `valid` | Initial date, step (1 = +6 h), valid time |
| `var` | `2t`, `10u`, `10v`, `msl`, `z500`, `t850`, `u850`, `v850` or `q700` |
| `rmse_truth`, `bias_truth` | Latitude-weighted RMSE and mean error against ERA5 on the 720 × 1440 grid |
| `rmse_ref` | Latitude-weighted RMSE against the FP32 forecast. This is the quantisation error |
| `seconds`, `peak_gib` | Rollout time for the date and peak GPU memory so far |

`summary.json` fields:

| Field | Meaning |
| --- | --- |
| `qerr_pct`, `qerr_pct_ci95` | Mean over dates, variables and steps of rmse_ref / rmse_truth(FP32), in per cent, with a 95 % bootstrap interval over dates (2000 resamples). This is the paper's qerr |
| `excess_pct`, `excess_pct_ci95` | Mean of rmse_truth / rmse_truth(FP32) − 1, in per cent: the ERA5 skill lost |
| `qerr_reduction_vs_q0_pct` (+ `_ci95`) | 1 − qerr / qerr(q0_sim), when the q0_sim card exists on the split |
| `per_var` | Mean `rmse_truth` and `rmse_ref` per variable and step, e.g. `z500_s2` |
| `n_dates`, `n_paired` | Dates scored, and date–variable–step cells paired with FP32 |
| `mean_seconds_per_date`, `peak_gib` | Cost |

`scripts/block_ci.py` replaces the plain bootstrap with the 3-day moving-block bootstrap reported in the paper.

### 4.2 Aggregate files: `results/_*.json`, `_*.npz`

These are written by the analysis scripts. Their names start with an underscore and their structure is documented at the top of the writing script. The two most used:

| File | Writer | Content |
| --- | --- | --- |
| `_diag_fields_<split>_h<steps>[_tag].json` | `diag_fields.py` | Per recipe, step and date: hydrostatic residual, super-adiabatic share, ageostrophic residual, wind and temperature extremes, small-scale error share and spectra, with the same quantities for FP32 and ERA5 |
| `_physmetrics_<split>_h<steps>.json` | `physmetrics_fields.py` | Per recipe, step and date: hydrostatic and geostrophic RMSE with the PhysMetrics definitions, L_eff, small-scale energy ratio, spectral divergence and residual, and the KE(500) spectra |

## 5. Expected behaviour and resources

| Quantity | Typical value (RTX 5090, `AQ_PERCEIVER_CHUNK=16384`) |
| --- | --- |
| Build of an RTN recipe | about 30 s (mostly loading the checkpoint) |
| Build of a GPTQ or smoothing recipe | 4–6 min (calibration pass over 7 days plus GPTQ of 138–195 layers) |
| One date, 2 steps | 5–30 s depending on the recipe (FP32 about 6 s; recipes with low-rank branches and many quantised layers are slower in simulation) |
| One card on the test split (30 dates) | 3–15 min plus the build |
| Peak GPU memory | 13–15 GiB with chunking, 24–27 GiB without |
| Agreement of simulated and native NVFP4 | t7_a16 with identical weights (`--weights from-codes`): qerr 9.52 against 9.46, mean paired difference +0.6 % (paper Sect. 4.7, Table S14j) |

**Determinism.** The rotation seeds are fixed, and GPTQ, smoothing and the low-rank branch are deterministic given the calibration statistics. Re-runs on the same GPU type reproduce `summary.json` to the printed digits. A different GPU or library version can change the last digit.

**Common errors.**

| Message | Cause and fix |
| --- | --- |
| `Aurora checkpoint not found` | Set `AQ_CKPT_DIR` |
| `FileNotFoundError ... surface_<date>.npz` | A day of the split or its verifying day is missing in `AQ_ERA5_DIR` |
| `no FP32 reference cache` / `qerr` missing in the summary | Run the `fp32` card on that split first |
| `results ... belongs to ...` | The results directory was created for another `AQ_MODEL`; pass `--results` |
| `manifest does not match the loaded model` | A different Aurora version or checkpoint; run `scripts/check_env.py` |
| `CUDA out of memory` | Set `AQ_PERCEIVER_CHUNK=16384`, or free GPU memory |
| `AQ_PERCEIVER_CHUNK must be a multiple of 16384` | Use 16384, 32768, ... |
