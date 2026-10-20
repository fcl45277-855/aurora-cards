# Pre-run protocol of the cross-architecture check

**Plan v1.** Recorded on 2026-10-20, after the Aurora results and before any quantised result of ClimaX or FourCastNet. It is an internal, time-stamped plan for a confirmatory test of transfer. It is not a public pre-registration of the study.

**Revisions.** Both were made before any quantised forward pass of the two models:
- v1.1 (2026-10-20) renamed one condition;
- v1.2 (2026-10-21) fixed two implementation faults.

Both are described at the end of this file.

## Question

Does the Aurora finding transfer to an independently developed model? In Aurora, quantisation sensitivity concentrates in learned-query cross-attention that aggregates state-dependent keys, and err_in_flip ranks module sensitivity. The test model is ClimaX, whose variable aggregation is a learned-query cross-attention. FourCastNet (AFNO) is an architectural contrast without this structure. It is not expected to be free of sensitive layers.

## Assets and gates

The following must hold before any evaluation:
- Upstream source commits, checkpoints and SHA-256 hashes are fixed.
- Checkpoints load strictly: no dropped output heads and no randomly initialised weights.
- Variables, units, levels, grid orientation, timestamps and lead time are fixed.
- Normalisation statistics come from training data, never from test dates.
- Missing inputs are never zero-filled.

A forecast-level claim needs a forecast checkpoint that beats persistence. No public ERA5 ClimaX forecast checkpoint with matching statistics was found, so the CMIP6-pretrained checkpoint was fine-tuned. The fine-tuning task was frozen before GPU training in `task_plan.json`, written by `prepare_training.py`:
- WeatherBench2 ERA5 at 64 × 32 conservative regridding: 2010–2015 for training and 2016 for validation, 6-h lead;
- 38 input and 5 output variables;
- full FP32 fine-tuning with AdamW at learning rate 1e-4 and weight decay 0.05, batch 2 × 4, one warm-up epoch, then cosine decay to 1e-6;
- gradient clipping at 1, at most 20 epochs, early stopping with patience 4 after epoch 5.

**Forecast gate.** The selected checkpoint must beat 6-h persistence on the aggregate 2016 validation MSE and individually on z500 and t850. Otherwise no quantisation results are produced.

## Fixed dates

| Use | Dates |
| --- | --- |
| Plumbing | 2024-02-02 (numerical consistency only) |
| Predictor | 2024-01-04, 2024-01-06, 2024-01-08 |
| Confirmation | 2024-03-01..12 (12 dates) |
| Season check | 2024-07-01..08 (8 dates) |

Each date contributes one 00 UTC step at the model's native lead. These dates overlap the Aurora evaluation; they are paired cross-model inputs, not new independent periods. Grid points, heads and queries within a date are not independent samples, so intervals use a moving-block bootstrap over dates (blocks of 3).

## Quantisation

NVFP4 round-to-nearest, without rotation, GPTQ, smoothing or low-rank branches:
- 16-element blocks along the input axis;
- UE4M3 block scales and an FP32 tensor scale;
- the packed Q/K/V matrix shares one tensor scale, and the output projection has its own.

ClimaX quantises one attention module at a time: `var_agg` and each of the eight ViT attention blocks, with all four projections. Conditions:
- W4A4: weights and projection-input activations in NVFP4 (primary condition);
- W4A32: weights only (auxiliary condition).

FourCastNet quantises the weights of one module group at a time (W4A32):
- groups: patch embedding, 12 AFNO spectral filters, 12 MLPs, output head;
- AFNO weights are blocked along their true input axis, and the real and imaginary parts share a tensor scale.

The reference and accumulation are FP32. Damage is the latitude-weighted standardised MSE against the FP32 forecast. Isolated-module damages are not additive and are not a causal decomposition.

## Hypotheses and criteria

**H1 (ClimaX hotspot).** `var_agg` ranks in the top quarter of the nine attention modules by isolated W4A4 damage on the confirmation dates. Its rank, absolute damage, parameter count and damage per parameter are reported. A miss counts as no support; the notion of hotspot is not redefined.

**H2 (predictor transfer).** err_in_flip is the share of attention-output squared error carried by query cells whose argmax key changes. It is computed under a K-path perturbation with Q and V fixed, before the output projection. Numerators and denominators are pooled over heads, cells and dates before dividing, and the value is undefined when there is no error.

err_in_flip is computed on the predictor dates and frozen. Its Spearman ρ with the confirmation damage must satisfy two conditions:
- ρ ≥ 0.7;
- ρ is higher than for each of three baselines: local activation error yerr, switching rate, and switching rate × value gap.

Perturbation amplitudes α = 1, 1/2 and 1/4 are reported for mechanism consistency. The primary condition is not changed after the fact.

**FourCastNet (descriptive).** The analysis reports isolated damage per module group, top-k concentration and damage per parameter. err_in_flip is undefined for AFNO and is reported as not applicable. A hotspot in FourCastNet would not contradict the attention claim; it would only rule out the stronger claim that all hotspots come from attention.

## Revisions

**v1.1 (2026-10-20).** The weight-only auxiliary condition is W4A32, not W4A16, because the reference is FP32. The protection-budget experiment (H3) was deferred to a separately frozen allocation and was not run.

**v1.2 (2026-10-21).** The ClimaX plumbing run stopped at the identity check: the adapter, with no quantisation, must reproduce the stock modules. Two faults were corrected in the evaluation scripts released here:

1. **Over-strict tolerance.** The elementwise tolerance (`allclose`, atol 1e-6, rtol 1e-5) rejected differences of about 2e-6 in near-zero outputs that arise from FP32 re-ordering of packed versus split GEMMs. The global relative difference was 2.2–5.7 × 10⁻⁷ for all nine modules (`results_summary/climax/diag_equivalence.json`). The check now requires a global relative difference below 1e-5, the protocol's FP32 criterion, together with `allclose(atol=1e-5, rtol=1e-5)`. The measured value is stored per module in the results.
2. **Rejected keyword arguments.** timm 1.x calls the attention module with `attn_mask=None, is_causal=False`, which the adapter refused, so no backbone module could run. The adapter now accepts exactly these values and still refuses any other argument.

Criteria, dates, formats, modules and the analysis scripts are unchanged.

## Exploratory analyses

The `var_agg` decomposition in `explore_var_agg.py` was defined after the H1/H2 analysis. It is reported as exploratory.
