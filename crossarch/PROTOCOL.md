# Pre-run protocol of the cross-architecture check

**Plan v1 (2026-10-20).** An internal, time-stamped protocol for testing transfer, recorded after the Aurora results and before any ClimaX or FourCastNet quantisation results. This was not a public preregistration.

**Revisions.** Both were made before any quantised forward pass of the two models:
- v1.1 (2026-10-20) renamed one condition;
- v1.2 (2026-10-21) fixed two implementation faults.

See Revisions below.

## Question

Do Aurora's sensitivity patterns and switching predictor transfer to an independently developed model? In Aurora, quantisation sensitivity concentrates in learned-query cross-attention over state-dependent keys, and `err_in_flip` ranks module sensitivity. ClimaX provides a structural comparison through its learned-query variable aggregation. FourCastNet (AFNO) lacks this structure and serves as an architectural contrast; it may still have sensitive layers.

## Assets and gates

Evaluation requires the following:
- Upstream source commits, checkpoints and SHA-256 hashes are fixed.
- Checkpoints load strictly: no dropped output heads and no randomly initialised weights.
- Variables, units, levels, grid orientation, timestamps and lead time are fixed.
- Normalisation statistics use training data only.
- Missing inputs are never zero-filled.

No public ClimaX ERA5 forecast checkpoint with matching statistics was found, so we fine-tuned the CMIP6-pretrained checkpoint. `prepare_training.py` froze the task in `task_plan.json` before GPU training:
- WeatherBench2 ERA5 at 64 × 32 conservative regridding: 2010–2015 for training and 2016 for validation, 6-h lead;
- 38 input and 5 output variables;
- full FP32 fine-tuning with AdamW at learning rate 1e-4 and weight decay 0.05, batch 2 × 4, one warm-up epoch, then cosine decay to 1e-6;
- gradient clipping at 1, at most 20 epochs, early stopping with patience 4 after epoch 5.

**Forecast gate.** The selected checkpoint must have lower 2016 validation MSE than 6-h persistence, both overall and separately for z500 and t850. Otherwise, quantisation evaluation stops.

## Fixed dates

| Use | Dates |
| --- | --- |
| Plumbing | 2024-02-02 (numerical consistency only) |
| Predictor | 2024-01-04, 2024-01-06, 2024-01-08 |
| Confirmation | 2024-03-01..12 (12 dates) |
| Season check | 2024-07-01..08 (8 dates) |

Each date contributes one 00 UTC step at the model's native lead. These dates overlap Aurora's evaluation, providing paired cross-model inputs rather than independent periods. Within-date grid points, heads and queries are not independent; intervals use a moving-block bootstrap over dates (block size 3).

## Quantisation

NVFP4 round-to-nearest, without rotation, GPTQ, smoothing or low-rank branches:
- 16-element blocks along the input axis;
- UE4M3 block scales and an FP32 tensor scale;
- the packed Q/K/V matrix shares one tensor scale, and the output projection has its own.

ClimaX quantises all four projections in one attention module at a time: `var_agg` or one of eight ViT blocks. Conditions:
- W4A4: weights and projection-input activations in NVFP4 (primary condition);
- W4A32: weights only (auxiliary condition).

FourCastNet quantises the weights of one module group at a time (W4A32):
- groups: patch embedding, 12 AFNO spectral filters, 12 MLPs, output head;
- AFNO weights are blocked along their true input axis, and the real and imaginary parts share a tensor scale.

The reference and accumulation are FP32. Damage is the latitude-weighted standardised MSE against the FP32 forecast. Isolated-module damages are not additive and are not a causal decomposition.

## Hypotheses and criteria

**H1 (ClimaX hotspot).** `var_agg` ranks in the top quarter of the nine attention modules by isolated W4A4 damage on the confirmation dates. Its rank, absolute damage, parameter count and damage per parameter are reported. A miss means no support; the hotspot criterion remains fixed.

**H2 (predictor transfer).** `err_in_flip` is the fraction of attention-output squared error in query cells whose argmax key changes, measured before the output projection under K-path perturbation with Q and V fixed. Pool numerator and denominator over heads, cells and dates before division; the ratio is undefined when total error is zero.

err_in_flip is computed on the predictor dates and frozen. Its Spearman ρ with the confirmation damage must satisfy two conditions:
- ρ ≥ 0.7;
- ρ is higher than for each of three baselines: local activation error yerr, switching rate, and switching rate × value gap.

Report results at α = 1, 1/2 and 1/4 for mechanism consistency; keep the primary condition fixed.

**FourCastNet (descriptive).** Report isolated damage per module group, top-k concentration and damage per parameter. `err_in_flip` is inapplicable to AFNO. Hotspots would refute the claim that all hotspots arise from attention, without contradicting Aurora's attention findings.

## Revisions

**v1.1 (2026-10-20).** The weight-only auxiliary condition is W4A32, not W4A16, because the reference is FP32. The protection-budget experiment (H3) was deferred to a separately frozen allocation and was not run.

**v1.2 (2026-10-21).** ClimaX's initial run stopped at the identity check, which requires the unquantised adapter to reproduce the original modules. Two implementation faults were fixed:

1. **Over-strict tolerance.** Outputs from packed and split FP32 GEMMs differed by about 2e-6 near zero, failing `allclose(atol=1e-6, rtol=1e-5)`. Global relative differences across nine modules were 2.2–5.7 × 10⁻⁷ (`results_summary/climax/diag_equivalence.json`). The revised check requires a global relative difference < 1e-5 (the protocol's FP32 criterion) and `allclose(atol=1e-5, rtol=1e-5)`, with per-module values recorded.
2. **Rejected keyword arguments.** The adapter rejected timm 1.x's `attn_mask=None, is_causal=False`, blocking all backbone attention modules. It now accepts these defaults while rejecting unsupported arguments.

Criteria, dates, formats, modules and the analysis scripts are unchanged.

## Exploratory analyses

The `var_agg` decomposition in `explore_var_agg.py` was defined after the H1/H2 analysis. It is reported as exploratory.
