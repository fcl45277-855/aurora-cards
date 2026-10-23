# Reproducing the paper

All commands run from the repository root with `AQ_CKPT_DIR` and `AQ_ERA5_DIR` set (see `docs/DATA.md`). Results go to `results/<card>__<split>_h<steps>/` with `records.jsonl` (per date, step and variable), `summary.json` and `build_info.json`. GPU times are for one RTX 5090 (32 GB) with `AQ_PERCEIVER_CHUNK=16384`, which keeps the peak memory near 15 GiB.

**Order matters.** On every split, run the `fp32` card first. It writes the full-precision reference that all other cards are scored against (`results/_ref_fp32_h<steps>/`). Recipes that use GPTQ or smoothing also need the 7 calibration days (`calib` split) on disk.

```bash
export AQ_PERCEIVER_CHUNK=16384
python scripts/paper_tables.py --results results     # after a re-run: compare against the paper's values
```

Table, figure and section numbers below refer to the submitted manuscript (C&G, revision v11) and its supplement (S1–S14).

## Without a GPU: from the result archive

Every number in the manuscript can be recomputed from the per-record results (`records.jsonl` of each card and the `_*.json` aggregates) without re-running a forecast. These records are in the full result archive `aurora-cards-results-v1.zip` (554 MB). Its Zenodo DOI has not been assigned yet; until deposition the archive is available from the corresponding authors. Unpack it in the repository root, which creates `results/`, `results_finetuned/` and `results_small/`; the scripts below then find their inputs at the default locations:

```bash
unzip aurora-cards-results-v1.zip -d .
```

| Paper item | Command | Per-record inputs |
| --- | --- | --- |
| Table 2 (layer sets) | `python scripts/layer_sets.py` | `recipes/manifest_314.json`, `recipes/q0_bf16_layers.txt`, `recipes/t7_mx.json` (in the repository) |
| Tables 3, 4 and 6; headline values | `python scripts/paper_tables.py --results results` | `summary.json` of the dev and test cards |
| Table 4 intervals (3-day blocks) | `python scripts/block_ci.py --split test t7_mx t7_g02_nodec_fold c08_weight_only_gptq --full` | `fp32__test_h2`, `t7_mx__test_h2`, `t7_g02_nodec_fold__test_h2`, `c08_weight_only_gptq__test_h2` |
| S14a–c (block lengths 1–7, per-date July and 2023 values, per-date rollout excess) and the distinct-resample counts | `python scripts/review_reanalysis.py` | the cards above plus `*__summer_h2`, `*__y2023_h2`, `fp32__test_h20`, `t7_mx__test_h20`, `fp32__summer_h20`, `t7_mx__summer_h20` |
| S14d (predictor without the resamplers) and Fig. 1 | `python scripts/b_eval.py` per model; `AQ_RESULTS=. python scripts/paper_figures.py --only 4` | `_b_eval.json` of the pretrained, finetuned and small models |
| Lead-time loss (Sects. 2.3, 4.4, 4.6; Table S14g) | `python scripts/skill_hours.py --split test --steps 20 t7_a16 t7_mx c08_t7set u8_w8a16_mx_prot q0_sim` | `fp32__test_h20` and the card's `__test_h20` (12 dates) |
| Table 6 intervals, paired differences (S14h), rollout intervals (S14c), FP32 reference skill (S7), 120-h spectra, predictor p-values and date robustness (S14d) | `python scripts/extended_analyses.py --results <archive>` (sections A–E) | test `__test_h2` and `__test_h20` cards, `_diag_fields_test_h20*.json`, `_replay_*.json` and `_b_eval.json` of the three models |
| Fig. 2 (a: excess by lead on 12 winter dates with the block-bootstrap band, b: storage) | `AQ_RESULTS=. python scripts/paper_figures.py --only 5` | `*__test_h20`, `*__summer_h20`; storage values from Table S14f |
| Figs. 3–5 | `AQ_RESULTS=. python scripts/paper_figures.py --only 2 3 1` | `_diag_fields_test_h2.json` and maps, `_diag_fields_dev_h2*.json`, `_diag_fields_test_h20.json` |
| Table 7 (PhysMetrics.Weather) | `python scripts/physmetrics_table.py results/_physmetrics_test_h2.json` | `_physmetrics_test_h2.json` |
| Table 8 (device) and the profile shares (Sect. 4.7) | values printed from `_native_timing.json` and `_profile_step.json` | `_native_timing.json`, `_profile_step.json` |
| Sect. 4.2 `to_kv` checks (8a–8k in S4) | `python scripts/decide_test.py` | `m1_to_kv`, `m1kv_*`, `m01_*` test cards |
| Table 5, S13 (ClimaX and FourCastNet) | see `crossarch/README.md` | `crossarch/results_summary/` (in the repository) |
| Table S14f (storage accounting) | `python scripts/memory_table.py` (needs the checkpoint, CPU only) | the checkpoint |
| Fig. 3 coastlines | `results/_lsm_2deg.npz` (2° land fraction from Aurora's `static.nc`) | — |

## Main tables (re-running the forecasts)

| Paper item | Data | Command | GPU time |
| --- | --- | --- | --- |
| Table 1 (formats) | none | definitions only; `tests/test_methods.py` checks the quantisers | — |
| Table 2 (layer sets) | none | `python scripts/layer_sets.py` | — |
| Table 3 (development path) | calib, dev | `bash scripts/run_deck.sh dev 2 fp32 q0_sim c04_no_rot c11_norot_gptq c09_smooth05_gptq g01_svdq_a05_r32 g02_svdq_a08_r32 g03_svdq_a08_r64 t7_g02_nodec_fold t7_mx c08_weight_only_gptq g02_mx` | ≈ 1.5 h |
| Table 4 (test month) | calib, test | `bash scripts/run_deck.sh test 2 fp32 bf16 q0_sim m01_q0_decoder_only m02_q0_no_decoder c08_weight_only_gptq g02_svdq_a08_r32 t7_g02_nodec_fold t7_mx`; intervals: `python scripts/block_ci.py --split test t7_mx t7_g02_nodec_fold c08_weight_only_gptq --full` | ≈ 2.5 h |
| Avg bits columns, Table S14f | none | `python scripts/memory_table.py bf16 q0_sim c08_weight_only_gptq g02_svdq_a08_r32 c08_t7set t7_g02_nodec_fold t7_mx` (needs the checkpoint) | CPU |
| Table 6 (protected configurations and 8-bit baselines) | calib, dev, test | `bash scripts/run_deck.sh test 2 t7_a16 c08_t7set t7mx_all314_noenc t7a16_all314_noenc u8_w8a16_mx u8_w8a16_ch u8_w8a8_mx u8_w8a8_ch u8_w8a16_mx_prot u8_w8a8_mx_prot` (t7 and t7_mx as for Tables 3 and 4). FP8 weights: recipe `weight.method` `mxfp8` or `fp8_channel`. Recipes with identical weights share one build through the cache in `results/_build_cache` (`AQ_BUILD_CACHE=0` disables it) | ≈ 1.5 h on an otherwise idle GPU |
| Table 7 (PhysMetrics.Weather) and S9 | test | `python scripts/diag_fields.py --recipes q0_sim m01_q0_decoder_only m02_q0_no_decoder t7_mx --split test --steps 2`; `python scripts/physmetrics_fields.py --recipes q0_sim t7_mx m01_q0_decoder_only m02_q0_no_decoder --split test --steps 2`, then `python scripts/physmetrics_table.py results/_physmetrics_test_h2.json`; 20 steps: add `--steps 20 --dates 2024-03-01 ... 2024-03-06` | ≈ 3 h + 3 h + 1.5 h |
| Table 8 (device cost) | dev (1 date) | `python scripts/native_timing.py --layers q0:inc=backbone --configs bf16 bf16_resident w4a16 w4a4`; `python scripts/native_timing.py --layers all314 --configs bf16 bf16_resident fp8_w8a16 fp8_w8a8 mxfp8_w8a8`; `python scripts/native_t7.py --recipe t7_a16 --act w4a16 --split test` (accuracy) and `--split dev --max-dates 1 --time-repeats 10` (timing); `python scripts/native_w4a16.py --act bf16`; `python scripts/native_w4a16.py --act nvfp4`; `python scripts/native_t7.py --recipe rne_t7 --act native`; `python scripts/native_t7.py --recipe t7_mx --act mxfp8_sim`; `python scripts/profile_step.py`. The reference keeps FP32 weights resident with BF16 autocast, as Aurora does by default | ≈ 1 h; needs mslk and compute capability ≥ 12.0 |
| S11 (protection order) | dev | `python scripts/alloc_greedy.py` writes 20 recipes and prints their names; run them with `run_deck.sh dev 2 ...`, then `python scripts/alloc_greedy.py --table` | ≈ 2 h |
| S12 (Hurricane Beryl) | summer | `python scripts/beryl_case.py` | ≈ 0.6 h |
| Table S10 (cross-model) | calib, dev | `AQ_MODEL=finetuned bash scripts/run_deck.sh dev 2 fp32 x_all_q0num x_dec_only m1enc x_bb_q0num ...` and the same with `AQ_MODEL=small`. The criteria and card lists are in the supplement. | ≈ 2 h per model |

## Mechanism (Sect. 4.2)

| Item | Command |
| --- | --- |
| Decoder `to_kv` cards: 88 % (`m1_to_kv` W4A4 vs the decoder-only card, squared qerr ratio), 97 % (K-half vs full activation-only error, `m1kv_K` vs `m1kv_act`), superlinearity, key centring | `bash scripts/run_deck.sh test 2 m1_to_kv m1kv_K m1kv_K_a05 m1kv_kc m1kv_act m1kv_g02 m1kv_g02_noLR`, then `python scripts/decide_test.py` |
| Replay of the resamplers (A, G, switching rate, err_in_flip, Eq. 11) | `python scripts/replay_attn.py --module decoder` and `--module encoder` |
| Replay of the Swin window attention | `python scripts/replay_swin.py` |
| Per-module sensitivity and the predictor ranking (Fig. 1) | `bash scripts/run_deck.sh dev 2 b_attn_perc_enc b_attn_perc_dec b_attn_en0 b_attn_en1 b_attn_en2 b_attn_de0 b_attn_de1 b_attn_de2`, then `python scripts/b_eval.py`; repeat both with `AQ_MODEL=finetuned` and `AQ_MODEL=small` |
| The diagnostic on any attention module | `cards/switching.py` (`switching_diagnostic(q, k, dk, v)`); demo: `examples/02_switching_diagnostic.py` |

## Robustness (Sect. 4.6)

| Item | Command |
| --- | --- |
| Second calibration week | `bash scripts/run_deck.sh test 2 t7_mx_cal2` (needs `calib2`) |
| Summer week | `bash scripts/run_deck.sh summer 2 fp32 bf16 q0_sim c08_weight_only_gptq t7_mx`; summer-calibrated recipe on winter: `run_deck.sh test 2 t7_mx_calsum` |
| Cross-year | `bash scripts/run_deck.sh y2023 2 fp32 bf16 q0_sim c08_weight_only_gptq t7_mx` |
| 20-step rollouts, lead-time loss | `python scripts/run_card.py --recipe recipes/<card>.json --split test --steps 20 --max-dates 12` for `fp32`, `q0_sim`, `t7_mx`, `t7_a16`, `c08_t7set`, `u8_w8a16_mx_prot`; then `python scripts/skill_hours.py --split test --steps 20 q0_sim t7_mx t7_a16 c08_t7set u8_w8a16_mx_prot` (rule in the script docstring) |
| 120-h spectra of the protected configurations | `python scripts/diag_fields.py --recipes t7_a16 c08_t7set u8_w8a16_mx_prot --split test --steps 20 --dates 2024-03-01 ... 2024-03-06 --tag _v14` |
| Wind extremes and balances over 20 steps | `python scripts/diag_fields.py --recipes q0_sim t7_mx --split test --steps 20 --dates 2024-03-01 ... 2024-03-12` |

## Unsuccessful interventions (Sect. 4.8, S12)

| Item | Command |
| --- | --- |
| Router (argmax / top-2) | `python scripts/router_test.py --modes softmax argmax top2 --modules enc dec both` |
| Exact key-path folding | `bash scripts/run_deck.sh dev 2 f_both_w4a4 f_both_mx m1enc_kv_K m1enc_kv_V` |
| Stochastic rounding over 20 steps | `run_card.py --recipe recipes/sr_t7.json --split test --steps 20 --max-dates 6`, and the same for `rne_t7` |
| Cross-step terms | `python scripts/crossterm.py --recipe t7_mx` |
| Input precision | `python scripts/input_bits.py --bits 16 12 10 8 6 4 --modes init state` |
| Switch-based block protection | `python scripts/diag_layers.py --layers all314 --out results/_diag_layers_all314.json`, then `python scripts/protect_select.py --budget 25e6` |

## Figures

`python scripts/paper_figures.py` draws Figs. 1–5 into `figures/`. It reads `results/_diag_fields_*`, the three `_b_eval.json` files and the `records.jsonl` of the 20-step cards below the directory `AQ_RESULTS`; with the archive unpacked in the repository root, run it with `AQ_RESULTS=.` to redraw the figures without re-running anything. The script's internal figure names follow an earlier numbering: `fig4_switching_predictor` is Fig. 1, `fig5_deployment` Fig. 2, `fig2_fingerprints_attribution` Fig. 3, `fig3_dose_response` Fig. 4 and `fig1_rollout_balances` Fig. 5 (`--only` uses the internal numbers).

## Determinism

The rotation and calibration seeds are fixed (`SEED` in `cards/harness.py`). GPTQ, smoothing and the low-rank branch are deterministic given the calibration statistics. Re-runs on the same GPU type reproduce `summary.json` to the printed digits. Other GPU types or library versions can change the last digit through floating-point accumulation order. Bootstrap intervals use a fixed seed (0).
