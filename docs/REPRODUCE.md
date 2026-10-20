# Reproducing the paper

All commands run from the repository root with `AQ_CKPT_DIR` and `AQ_ERA5_DIR` set (see `docs/DATA.md`). Results go to `results/<card>__<split>_h<steps>/` with `records.jsonl` (per date, step and variable), `summary.json` and `build_info.json`. GPU times are for one RTX 5090 (32 GB) with `AQ_PERCEIVER_CHUNK=16384`, which keeps the peak memory near 15 GiB.

**Order matters.** On every split, run the `fp32` card first. It writes the full-precision reference that all other cards are scored against (`results/_ref_fp32_h<steps>/`). Recipes that use GPTQ or smoothing also need the 7 calibration days (`calib` split) on disk.

```bash
export AQ_PERCEIVER_CHUNK=16384
python scripts/paper_tables.py --results results     # after a re-run: compare against the paper's values
```

## Main tables

| Paper item | Data | Command | GPU time |
| --- | --- | --- | --- |
| Table 1 (formats) | none | definitions only; `tests/test_methods.py` checks the quantisers | — |
| Table 2 (path, dev) | calib, dev | `bash scripts/run_deck.sh dev 2 fp32 q0_sim c04_no_rot c11_norot_gptq c09_smooth05_gptq g01_svdq_a05_r32 g02_svdq_a08_r32 g03_svdq_a08_r64 t7_g02_nodec_fold t7_mx c08_weight_only_gptq g02_mx` | ≈ 1.5 h |
| Table 3 (test month) | calib, test | `bash scripts/run_deck.sh test 2 fp32 bf16 q0_sim m01_q0_decoder_only m02_q0_no_decoder c08_weight_only_gptq g02_svdq_a08_r32 t7_g02_nodec_fold t7_mx`; confidence intervals: `python scripts/block_ci.py --split test t7_mx c08_weight_only_gptq --full` | ≈ 2.5 h |
| Avg bits column | none | `python scripts/memory_table.py bf16 q0_sim c08_weight_only_gptq g02_svdq_a08_r32 t7_g02_nodec_fold t7_mx` (needs the checkpoint) | CPU |
| Table 5 / S9 (balance diagnostics, test) | test | `python scripts/diag_fields.py --recipes q0_sim m01_q0_decoder_only m02_q0_no_decoder t7_mx --split test --steps 2` | ≈ 3 h |
| Table 6 (PhysMetrics.Weather) | test | `python scripts/physmetrics_fields.py --recipes q0_sim t7_mx m01_q0_decoder_only m02_q0_no_decoder --split test --steps 2`, then `python scripts/physmetrics_table.py results/_physmetrics_test_h2.json`; 20 steps: add `--steps 20 --dates 2024-03-01 ... 2024-03-06` | ≈ 3 h + 1.5 h |
| Table 7 (device cost) | dev (1 date) | `python scripts/native_timing.py`; `python scripts/native_w4a16.py --act bf16`; `python scripts/native_w4a16.py --act nvfp4`; `python scripts/native_t7.py --recipe rne_t7 --act native`; `python scripts/native_t7.py --recipe t7_mx --act mxfp8_sim`; `python scripts/profile_step.py` | ≈ 1 h; needs mslk and compute capability ≥ 12.0 |
| Table 8 / S11 (protection order) | dev | `python scripts/alloc_greedy.py` writes 20 recipes and prints their names; run them with `run_deck.sh dev 2 ...`, then `python scripts/alloc_greedy.py --table` | ≈ 2 h |
| Table 9 / S12 (Hurricane Beryl) | summer | `python scripts/beryl_case.py` | ≈ 0.6 h |
| Table S10 (cross-model) | calib, dev | `AQ_MODEL=finetuned bash scripts/run_deck.sh dev 2 fp32 x_all_q0num x_dec_only m1enc x_bb_q0num ...` and the same with `AQ_MODEL=small`. The criteria and card lists are in the supplement. | ≈ 2 h per model |

## Mechanism (Sect. 4.2)

| Item | Command |
| --- | --- |
| Decoder `to_kv` cards (88 % / 97 % shares, superlinearity, key centring) | `bash scripts/run_deck.sh test 2 m1_to_kv m1kv_K m1kv_K_a05 m1kv_kc m1kv_act m1kv_g02 m1kv_g02_noLR`, then `python scripts/decide_test.py` |
| Replay of the resamplers (A, G, switching rate, err_in_flip) | `python scripts/replay_attn.py --module decoder` and `--module encoder` |
| Replay of the Swin window attention | `python scripts/replay_swin.py` |
| Per-module sensitivity and the predictor ranking (Fig. 1) | `bash scripts/run_deck.sh dev 2 b_attn_perc_enc b_attn_perc_dec b_attn_en0 b_attn_en1 b_attn_en2 b_attn_de0 b_attn_de1 b_attn_de2`, then `python scripts/b_eval.py`; repeat both with `AQ_MODEL=finetuned` and `AQ_MODEL=small` |
| The diagnostic on any attention module | `cards/switching.py` (`switching_diagnostic(q, k, dk, v)`); demo: `examples/02_switching_diagnostic.py` |

## Robustness (Sect. 4.5)

| Item | Command |
| --- | --- |
| Second calibration week | `bash scripts/run_deck.sh test 2 t7_mx_cal2` (needs `calib2`) |
| Summer week | `bash scripts/run_deck.sh summer 2 fp32 bf16 q0_sim c08_weight_only_gptq t7_mx`; summer-calibrated recipe on winter: `run_deck.sh test 2 t7_mx_calsum` |
| Cross-year | `bash scripts/run_deck.sh y2023 2 fp32 bf16 q0_sim c08_weight_only_gptq t7_mx` |
| 20-step rollouts, lead-time loss | `python scripts/run_card.py --recipe recipes/<card>.json --split test --steps 20 --max-dates 12` for `fp32`, `q0_sim`, `t7_mx`; then `python scripts/skill_hours.py --split test --steps 20 q0_sim t7_mx` |
| Wind extremes and balances over 20 steps | `python scripts/diag_fields.py --recipes q0_sim t7_mx --split test --steps 20 --dates 2024-03-01 ... 2024-03-12` |

## Negative results (Sect. 4.9, S12)

| Item | Command |
| --- | --- |
| Router (argmax / top-2) | `python scripts/router_test.py --modes softmax argmax top2 --modules enc dec both` |
| Exact key-path folding | `bash scripts/run_deck.sh dev 2 f_both_w4a4 f_both_mx m1enc_kv_K m1enc_kv_V` |
| Stochastic rounding over 20 steps | `run_card.py --recipe recipes/sr_t7.json --split test --steps 20 --max-dates 6`, and the same for `rne_t7` |
| Cross-step terms | `python scripts/crossterm.py --recipe t7_mx` |
| Input precision | `python scripts/input_bits.py --bits 16 12 10 8 6 4 --modes init state` |
| Switch-based block protection | `python scripts/diag_layers.py --layers all314 --out results/_diag_layers_all314.json`, then `python scripts/protect_select.py --budget 25e6` |

## Figures

`python scripts/paper_figures.py` draws Figs. 1–5 into `figures/`. It reads `_diag_fields_*`, `_b_eval.json` and the `records.jsonl` of the 20-step cards from `AQ_RESULTS`, which defaults to `results/`. Point it at the unpacked Zenodo archive to redraw the figures without re-running anything.

## Determinism

The rotation and calibration seeds are fixed (`SEED` in `cards/harness.py`). GPTQ, smoothing and the low-rank branch are deterministic given the calibration statistics. Re-runs on the same GPU type reproduce `summary.json` to the printed digits. Other GPU types or library versions can change the last digit through floating-point accumulation order.
