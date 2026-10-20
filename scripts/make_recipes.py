"""Write the recipe JSON files of deck 1 into recipes/. Re-run after editing DECK."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "recipes"
FULL_BF16 = {"block": "full", "dtype": "bf16"}
FULL_FP32 = {"block": "full", "dtype": "fp32"}
# layers whose input has very few tokens (per-sample time/lead-time embeddings, 13 pressure levels): BF16
# activations there cost almost no compute
G02 = {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
       "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]}
TINY_PATS = ["ln_modulation", "time_mlp", "lead_time_embed", "absolute_time_embed", "atmos_levels_embed"]
TINY = [[p, "bf16"] for p in TINY_PATS]
# activation-sensitivity localisation (corrected pipeline): W4 GPTQ everywhere, A16 by default, A4 only in one group
SENS_GROUPS = {
    "enc": ["encoder."], "dec": ["decoder."],
    "benc0": ["backbone.encoder_layers.0."], "benc1": ["backbone.encoder_layers.1."], "benc2": ["backbone.encoder_layers.2."],
    "bdec0": ["backbone.decoder_layers.0."], "bdec1": ["backbone.decoder_layers.1."], "bdec2": ["backbone.decoder_layers.2."],
    "attn": [".attn."], "mlp": [".mlp."], "mod": ["ln_modulation"], "time": ["time_mlp", "_embed"],
}
DECK = {
    # references
    "fp32": {"kind": "fp32"},
    "bf16": {"kind": "bf16"},
    "native_q0": {"kind": "native_q0"},
    # parity: simulated Q0 must reproduce native_q0 (same layers, full-width BF16 rotation, RTN, BF16 weights)
    "q0_sim": {"layers": "q0", "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}},
    # deck 1, Q0 layer set (195 NVFP4 layers)
    "c01_rot_fp32": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "rtn", "source": "fp32"}},
    "c02_rot_block16": {"layers": "q0", "rotation": {"block": 16, "dtype": "fp32"}, "weight": {"method": "rtn", "source": "fp32"}},
    "c03_rot_block128": {"layers": "q0", "rotation": {"block": 128, "dtype": "fp32"}, "weight": {"method": "rtn", "source": "fp32"}},
    "c04_no_rot": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32"}},
    "c05_gptq_full": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}},
    "c06_gptq_block16": {"layers": "q0", "rotation": {"block": 16, "dtype": "fp32"}, "weight": {"method": "gptq", "source": "fp32"}},
    "c07_weight_only_rtn": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "rtn", "source": "fp32"}, "act": "bf16"},
    "c08_weight_only_gptq": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}, "act": "bf16"},
    "c09_smooth05_gptq": {"layers": "q0", "smooth": {"alpha": 0.5}, "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}},
    # literature-driven (group D report): rotation hurts NVFP4; 4/6 and scale search are calibration-free fixes
    "c10_norot_46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32", "scale_search": [1.0, 1.5]},
                     "act_scale_search": [1.0, 1.5]},
    "c11_norot_gptq": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"}},
    "c12_norot_sweep": {"layers": "q0", "rotation": {"block": None},
                        "weight": {"method": "rtn", "source": "fp32", "scale_search": [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]},
                        "act_scale_search": [1.0, 1.5]},
    "c13_rot16_gptq_act46": {"layers": "q0", "rotation": {"block": 16, "dtype": "fp32"}, "weight": {"method": "gptq", "source": "fp32"},
                             "act_scale_search": [1.0, 1.5]},
    "a03_all_norot_gptq": {"layers": "all314", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"}},
    # deck 2 (after deck 1 showed GPTQ removes ~97% of layer weight error but only ~10% of qerr: activations dominate)
    "d01_w4a8_norot": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32", "scale_search": [1.0, 1.5]},
                       "act": "fp8"},
    # corrected pipeline: d01 (A8, RTN weights) ~= c07 (A16, RTN weights) -> A8 is nearly free; W4A8 with GPTQ weights should reach ~c08
    "d06_w4a8_gptq_norot": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"}, "act": "fp8"},
    "d02_norot_46_tinybf16": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32", "scale_search": [1.0, 1.5]},
                              "act_scale_search": [1.0, 1.5], "act_overrides": TINY},
    "d03_norot_gptq_act46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                             "act_scale_search": [1.0, 1.5]},
    "d04_norot_gptq_act46_tinybf16": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                                      "act_scale_search": [1.0, 1.5], "act_overrides": TINY},
    "d05_norot_actsweep": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32", "scale_search": [1.0, 1.5]},
                           "act_scale_search": [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]},
    # deck 2b: is GPTQ overfitting the low-rank adaLN modulation inputs? where does the weight error sit?
    "e01_wo_mod_only": {"layers": "q0:inc=ln_modulation", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32"}, "act": "bf16"},
    "e02_wo_nomod": {"layers": "q0:exc=ln_modulation", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32"}, "act": "bf16"},
    "e03_norot_gptq_skiptiny_act46": {"layers": "q0", "rotation": {"block": None},
                                      "weight": {"method": "gptq", "source": "fp32", "gptq_skip": TINY_PATS},
                                      "act_scale_search": [1.0, 1.5]},
    "e04_norot_gptq_skiptiny_act46_tinybf16": {"layers": "q0", "rotation": {"block": None},
                                               "weight": {"method": "gptq", "source": "fp32", "gptq_skip": TINY_PATS},
                                               "act_scale_search": [1.0, 1.5], "act_overrides": TINY},
    # deck 3: stack the fixes that each helped in deck 1 (no rotation, GPTQ, 4/6 activations, smoothing)
    "f01_smooth05_norot_gptq_act46": {"layers": "q0", "smooth": {"alpha": 0.5}, "rotation": {"block": None},
                                      "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "f02_smooth03_norot_gptq_act46": {"layers": "q0", "smooth": {"alpha": 0.3}, "rotation": {"block": None},
                                      "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "f03_smooth07_norot_gptq_act46": {"layers": "q0", "smooth": {"alpha": 0.7}, "rotation": {"block": None},
                                      "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "f04_smooth05_rot16_gptq_act46": {"layers": "q0", "smooth": {"alpha": 0.5}, "rotation": {"block": 16, "dtype": "fp32"},
                                      "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    # deck 3b: SVDQuant-style low-rank BF16 branch (activation difficulty moved to weights by smoothing)
    "g01_svdq_a05_r32": {"layers": "q0", "smooth": {"alpha": 0.5}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                         "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "g02_svdq_a08_r32": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                         "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "g03_svdq_a08_r64": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 64}, "rotation": {"block": None},
                         "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    # innovation N1: scale-separated activation quantization on backbone MLP layers (block-mean BF16 branch on
    # 1/pool^2 of the tokens, NVFP4 residual); weights GPTQ, no rotation/smoothing (N1 needs the raw token grid)
    "h01_n1_pool4_gptq_act46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                                "act_scale_search": [1.0, 1.5], "lowpass": {"pool": 4, "patterns": ["mlp.fc1", "mlp.fc2"]}},
    "h02_n1_pool2_gptq_act46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                                "act_scale_search": [1.0, 1.5], "lowpass": {"pool": 2, "patterns": ["mlp.fc1", "mlp.fc2"]}},
    "h03_n1_pool8_gptq_act46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                                "act_scale_search": [1.0, 1.5], "lowpass": {"pool": 8, "patterns": ["mlp.fc1", "mlp.fc2"]}},
    "h05_n1_mlp4_attn2_gptq_act46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                                     "act_scale_search": [1.0, 1.5],
                                     "lowpass": {"pool": 4, "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    # N1 on top of the strongest engineering baseline (g02, SVDQuant-style)
    "i01_svdq_a08_r32_n1": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                            "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5],
                            "lowpass": {"pool": 4, "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    "i02_svdq_a08_r32_n1mlp": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                               "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5],
                               "lowpass": {"pool": 4, "patterns": ["mlp.fc1", "mlp.fc2"]}},
    # prior-work ablations of N1 (novelty check: STaMP, HadaNorm, DIVISION), same base as h05 / i01
    "j01_hadanorm_global": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                            "act_scale_search": [1.0, 1.5],
                            "lowpass": {"pool": 4, "mode": "global", "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    "j02_stamp_haar_bf16": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                            "act_scale_search": [1.0, 1.5],
                            "lowpass": {"pool": 4, "mode": "haar", "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    "j03_stamp_haar_fp8": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                           "act_scale_search": [1.0, 1.5],
                           "lowpass": {"pool": 4, "mode": "haar", "coarse_bits": "fp8", "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    "j04_n1_coarse_fp8": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                          "act_scale_search": [1.0, 1.5],
                          "lowpass": {"pool": 4, "coarse_bits": "fp8", "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    "j05_svdq_stamp_haar_fp8": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                                "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5],
                                "lowpass": {"pool": 4, "mode": "haar", "coarse_bits": "fp8", "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    "j06_svdq_n1_coarse_fp8": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                               "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5],
                               "lowpass": {"pool": 4, "coarse_bits": "fp8", "patterns": ["mlp.fc1", "mlp.fc2"], "attn_window": [6, 12], "attn_pool": 2}},
    **{f"s_{g}": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}, "act": "bf16",
                  "act_overrides": [[p, "nvfp4"] for p in pats]}
       for g, pats in SENS_GROUPS.items()},
    # deck R2 (corrected pipeline, 2026-10-18): smoothing + low-rank was the only big lever (g02 qerr 33.7% vs Q0 147%),
    # and full-width rotation helps activations in the corrected pipeline -> combine and dissect
    "r01_svdq_a08_r32_rotfull": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": FULL_FP32,
                                 "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r02_smooth08_rotfull_gptq_act46": {"layers": "q0", "smooth": {"alpha": 0.8}, "rotation": FULL_FP32,
                                        "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r03_smooth08_norot_gptq_act46": {"layers": "q0", "smooth": {"alpha": 0.8}, "rotation": {"block": None},
                                      "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r04_svdq_a08_r16": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 16}, "rotation": {"block": None},
                         "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r05_svdq_a09_r32": {"layers": "q0", "smooth": {"alpha": 0.9}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                         "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r06_svdq_a08_r32_rot16": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": 16, "dtype": "fp32"},
                               "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r07_svdq_a08_r32_noact46": {"layers": "q0", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                                 "weight": {"method": "gptq", "source": "fp32"}},
    "r08_all314_svdq_a08_r32": {"layers": "all314", "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": None},
                                "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5]},
    "r09_all314_w4a16_gptq": {"layers": "all314", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}, "act": "bf16"},
    # decisive experiments from the 2026-10-18 innovation workflow (the pre-registered decision rules)
    "m01_q0_decoder_only": {"layers": "q0:inc=decoder.level_decoder,decoder.atmos_levels_embed",
                            "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}},
    "m02_q0_no_decoder": {"layers": "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed",
                          "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}},
    # m01 passed (6 decoder layers carry ~90% of Q0's upper-air MSE): which layer, weights or activations?
    "m01a_dec_act_only": {"layers": "q0:inc=decoder.level_decoder,decoder.atmos_levels_embed",
                          "rotation": FULL_BF16, "weight": {"method": "none", "source": "bf16"}},
    # control: the sim layer's BF16 matmul alone (autocast keeps the decoder in FP32), no NVFP4 anywhere
    "m01c_dec_bf16_ctl": {"layers": "q0:inc=decoder.level_decoder,decoder.atmos_levels_embed",
                          "rotation": FULL_BF16, "weight": {"method": "none", "source": "bf16"}, "act": "bf16"},
    "m01w_dec_weight_only": {"layers": "q0:inc=decoder.level_decoder,decoder.atmos_levels_embed",
                             "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}, "act": "bf16"},
    **{f"m1_{k}": {"layers": f"q0:inc={p}", "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}}
       for k, p in [("lvl_embed", "decoder.atmos_levels_embed"), ("to_q", "decoder.level_decoder.layers.0.0.to_q"),
                    ("to_kv", "decoder.level_decoder.layers.0.0.to_kv"), ("to_out", "decoder.level_decoder.layers.0.0.to_out"),
                    ("fc1", "decoder.level_decoder.layers.0.1.net.0"), ("fc2", "decoder.level_decoder.layers.0.1.net.2")]},
    # is the decoder damage TorchAO's block-scale floor (2^-6)? same cards with E4M3-subnormal block scales (2^-9)
    "m01_sub": {"layers": "q0:inc=decoder.level_decoder,decoder.atmos_levels_embed",
                "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}, "scale_min": "subnormal"},
    "q0_sub": {"layers": "q0", "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}, "scale_min": "subnormal"},
    "t1_g02_dec_bf16": {**G02, "act_overrides": [["decoder.", "bf16"]]},
    "t4_g02_dec_fp8": {**G02, "act_overrides": [["decoder.", "fp8"]]},
    "t2_g02_mlpfc_fp8": {**G02, "act_overrides": [[".mlp.fc", "fp8"]]},
    "gfold_g02": {**G02, "layers": "q0:exc=ln_modulation,time_mlp,_embed,level_decoder.layers.0.0.to_q"},
    "t3_g02_posemb_bf16": {**G02, "act_overrides": [["pos_embed", "bf16"], ["absolute_time_embed", "bf16"]]},
    # after T1/T4 (decoder activations BF16/FP8 close ~70% of the gap; z500 stays at +5%): add folding of the
    # state-independent layers (exact, free), and the upper bound with the 6 decoder layers left out entirely
    "t5_g02_dec_fp8_fold": {**G02, "layers": "q0:exc=ln_modulation,time_mlp,_embed,level_decoder.layers.0.0.to_q",
                            "act_overrides": [["decoder.", "fp8"]]},
    # single layers showed to_kv alone carries 87% of the decoder damage: protect only that layer
    "m03_q0_no_kv": {"layers": "q0:exc=level_decoder.layers.0.0.to_kv", "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}},
    "t8_g02_kv_bf16": {**G02, "act_overrides": [["level_decoder.layers.0.0.to_kv", "bf16"]]},
    "t9_g02_kv_fp8": {**G02, "act_overrides": [["level_decoder.layers.0.0.to_kv", "fp8"]]},
    # t8 (g02 + to_kv A16) barely helps although T1 (g02 + whole decoder A16) closes 71% of the gap: on g02 the
    # dominant decoder layer is another one. Leave-one-in: whole decoder A16 except one layer kept at A4.
    **{f"u_g02_dec16_but_{k}": {**G02, "act_overrides": [[p, "nvfp4"], ["decoder.", "bf16"]]}
       for k, p in [("to_kv", "level_decoder.layers.0.0.to_kv"),       # added after audit 07: same design for all six
                    ("to_out", "level_decoder.layers.0.0.to_out"), ("fc1", "level_decoder.layers.0.1.net.0"),
                    ("fc2", "level_decoder.layers.0.1.net.2"), ("to_q", "level_decoder.layers.0.0.to_q"),
                    ("lvl_embed", "decoder.atmos_levels_embed")]},
    # 06 plan E2: to_kv on Q0 numerics (full BF16 rotation): activation / weight / FP8 split, K vs V path,
    # error-amplitude sweep on K, and K centring over the 3 context tokens (exact for keys)
    **{f"m1kv_{k}": {"layers": "q0:inc=decoder.level_decoder.layers.0.0.to_kv", "rotation": FULL_BF16, **v}
       for k, v in [("act", {"weight": {"method": "none", "source": "bf16"}}),
                    ("w", {"weight": {"method": "rtn", "source": "bf16"}, "act": "bf16"}),
                    ("fp8", {"weight": {"method": "none", "source": "bf16"}, "act": "fp8"}),
                    ("K", {"weight": {"method": "none", "source": "bf16"},
                           "kv_split": {"pattern": "to_kv", "quant_rows": [0, 1024], "alpha": 1.0}}),
                    ("V", {"weight": {"method": "none", "source": "bf16"},
                           "kv_split": {"pattern": "to_kv", "quant_rows": [1024, 2048], "alpha": 1.0}}),
                    ("K_a05", {"weight": {"method": "none", "source": "bf16"},
                               "kv_split": {"pattern": "to_kv", "quant_rows": [0, 1024], "alpha": 0.5}}),
                    ("K_a025", {"weight": {"method": "none", "source": "bf16"},
                                "kv_split": {"pattern": "to_kv", "quant_rows": [0, 1024], "alpha": 0.25}}),
                    ("kc", {"weight": {"method": "rtn", "source": "bf16"}, "k_centre": ["level_decoder.layers.0.0.to_kv"]})]},
    # 06 plan E3: which g02 component neutralises to_kv (only to_kv quantized, g02 numerics)
    **{f"m1kv_{k}": {"layers": "q0:inc=decoder.level_decoder.layers.0.0.to_kv", "rotation": {"block": None},
                     "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5], **v}
       for k, v in [("g02", {"smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}}), ("g02_noLR", {"smooth": {"alpha": 0.8}}),
                    ("g02_noSm", {"low_rank": {"rank": 32}}), ("norot", {})]},
    "t6_g02_nodec": {**G02, "layers": "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed"},
    "t7_g02_nodec_fold": {**G02, "layers": "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed"},
    "c05_dec_fp8": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"},
                    "act_overrides": [["decoder.", "mxfp8"]]},
    "h04_n1_pool4_rtn46": {"layers": "q0", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32", "scale_search": [1.0, 1.5]},
                           "act_scale_search": [1.0, 1.5], "lowpass": {"pool": 4, "patterns": ["mlp.fc1", "mlp.fc2"]}},
    # deck 1, all 314 layers (memory target: everything NVFP4)
    "a01_all_rtn": {"layers": "all314", "rotation": FULL_FP32, "weight": {"method": "rtn", "source": "fp32"}},
    "a02_all_gptq": {"layers": "all314", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}},
}

# ---- after the claims audit (the claims audit) -------------------------------------------------------------------
T7_SET = "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed"
STATIC = "ln_modulation,time_mlp,_embed,level_decoder.layers.0.0.to_q"
C08 = DECK["c08_weight_only_gptq"]
D06 = DECK["d06_w4a8_gptq_norot"]
DECK.update({
    # P2-15: deployable FP8 activations (the "fp8" mode uses an optimistic FP32 scale per 16 elements)
    "d06_mx": {**D06, "act": "mxfp8"},
    "d06_ft": {**D06, "act": "fp8_tensor"},
    "t4_mx": {**G02, "act_overrides": [["decoder.", "mxfp8"]]},
    "t5_mx": {**G02, "layers": "q0:exc=" + STATIC, "act_overrides": [["decoder.", "mxfp8"]]},
    "t9_mx": {**G02, "act_overrides": [["level_decoder.layers.0.0.to_kv", "mxfp8"]]},
    "m1kv_mx": {**DECK["m1kv_fp8"], "act": "mxfp8"},
    "m1kv_ft": {**DECK["m1kv_fp8"], "act": "fp8_tensor"},
    # first MX runs used OCP floor scaling (clips block maxima); kept for the record, "mxfp8" is now RCEIL
    "d06_mx_floor": {**D06, "act": "mxfp8_floor"},
    # deployable W4A8 is ~72 (d06_ft 72.2, d06_mx 72.2) vs 19.1 with the optimistic per-16 FP32-scale FP8:
    # does the usual outlier treatment rescue it?
    "d07_rotfull_gptq_mx": {"layers": "q0", "rotation": FULL_FP32, "weight": {"method": "gptq", "source": "fp32"}, "act": "mxfp8"},
    "g02_mx": {**G02, "act": "mxfp8"},
    # g02_mx: surface nearly lossless but z500 +8.6/+10.6 (decoder FP8 again) -> keep the decoder in high precision
    "t7_mx": {**G02, "layers": T7_SET, "act": "mxfp8"},
    "t1_mx": {**G02, "act": "mxfp8", "act_overrides": [["decoder.", "bf16"]]},
    "m1kv_mx_floor": {**DECK["m1kv_fp8"], "act": "mxfp8_floor"},
    # P0-6: layer-set controls (Q0's 195 layers came from the buggy old pipeline) and g02's own A16 floor
    "g02_a16": {**G02, "act": "bf16"},
    "c08_t7set": {**C08, "layers": T7_SET},
    "d06_t7set": {**D06, "layers": T7_SET},
    "t7_all314": {**G02, "layers": "all314:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed"},
    "d06_all314": {**D06, "layers": "all314", "act": "mxfp8"},
    "m02_full": {"layers": "all314:inc=backbone.,encoder.pos_embed,encoder.absolute_time_embed", "rotation": FULL_BF16,
                 "weight": {"method": "rtn", "source": "bf16"}},
    # P0-8: C8 / C11 controls
    "t1_t2": {**G02, "act_overrides": [["decoder.", "bf16"], [".mlp.fc", "mxfp8"]]},
    "t1_bb_fp8": {**G02, "act": "mxfp8", "act_overrides": [["decoder.", "bf16"]]},
    "c08_norot": {**C08, "rotation": {"block": None}},
    "d06_fold": {**D06, "layers": "q0:exc=" + STATIC, "act": "mxfp8"},   # deployable FP8 (optimistic "fp8" retracted)
    # t7_all314 (205 layers) 56.8 vs t7 (138) 17.25: which of the layers Q0 kept in BF16 hurt? the encoder Perceiver
    # (level_agg, structural twin of the decoder's level_decoder) is the first suspect; and does A8 rescue all314?
    "t7_all314_noenc": {**G02, "layers": "all314:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed,encoder.level_agg"},
    "t7mx_all314": {**G02, "layers": "all314:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed", "act": "mxfp8"},
    # encoder Perceiver = 76% of the all314 surplus (t7_all314 56.8 -> noenc 31.8): keep both Perceivers in high precision
    "t7mx_all314_noenc": {**G02, "layers": "all314:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed,encoder.level_agg", "act": "mxfp8"},
    # 08 novelty/data check follow-ups. F2: FP8 scaling granularity that also keeps block maxima exact
    "d06_tok": {**D06, "act": "fp8_token"},
    "d06_t128": {**D06, "act": "fp8_tile128"},
    # F1: encoder Perceiver on the same 138-layer base (layer list file written on the server), same-machine t7,
    # and its weight / activation split (Q0 numerics, like m01a / m01w for the decoder)
    "t7_w": {**G02, "layers": T7_SET},
    "t7_plus_enc": {**G02, "layers": "recipes/layers_t7_plus_enc.txt"},
    "t7mx_plus_enc": {**G02, "layers": "recipes/layers_t7_plus_enc.txt", "act": "mxfp8"},
    **{f"m1enc{k}": {"layers": "all314:inc=encoder.level_agg", "rotation": FULL_BF16, **v}
       for k, v in [("", {"weight": {"method": "rtn", "source": "bf16"}}),
                    ("_act", {"weight": {"method": "none", "source": "bf16"}}),
                    ("_w", {"weight": {"method": "rtn", "source": "bf16"}, "act": "bf16"})]},
    # cross-model replication (AQ_MODEL=small / finetuned): rule-based layer sets only ("all314" = the model's manifest)
    **{f"x_{k}": {"layers": spec, "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}}
       for k, spec in [("all_q0num", "all314"), ("bb_q0num", "all314:inc=backbone."),
                       ("dec_only", "all314:inc=decoder.level_decoder,decoder.atmos_levels_embed"),
                       ("kv_only", "all314:inc=decoder.level_decoder.layers.0.0.to_kv")]},
    "x_g02mx_all": {**G02, "layers": "all314:exc=" + STATIC, "act": "mxfp8"},   # everything A8 except static (stress card)
    # 09 candidates. #2 exact K path of the decoder to_kv (queries are state-independent, so K logits can be folded
    # into a 1024x13 matrix at deployment; numerically = exact K rows, quantized V rows), on g02 numerics
    "g02_kfold": {**G02, "kv_split": {"pattern": "level_decoder.layers.0.0.to_kv", "quant_rows": [1024, 2048], "alpha": 1.0}},
    "t7_mx_kfold_dec": {**G02, "layers": "q0:exc=ln_modulation,time_mlp,_embed,level_decoder.layers.0.0.to_q", "act": "mxfp8",
                        "kv_split": {"pattern": "level_decoder.layers.0.0.to_kv", "quant_rows": [1024, 2048], "alpha": 1.0}},
    # #6 closed-loop calibration: statistics from 4 autoregressive steps instead of 2
    "t7_mx_cal4": {**G02, "layers": T7_SET, "act": "mxfp8", "calib": {"split": "calib", "steps": 4, "budget_gib": 2.5}},
    # 09 #8 (cross-step terms): format-compatible per-step knob. W4A4 t7 with a FIXED activation scale multiplier
    # (1.0 or 1.5) vs ALTERNATING by rollout step; if cross terms are positive and the knob decorrelates them, the
    # alternating run should beat both fixed runs at step 4 (test, 3 dates, 4 steps)
    "t7_m10": {**G02, "layers": T7_SET, "act_scale_search": None, "act_step_mults": [1.0]},
    "t7_m15": {**G02, "layers": T7_SET, "act_scale_search": None, "act_step_mults": [1.5]},
    "t7_alt": {**G02, "layers": T7_SET, "act_scale_search": None, "act_step_mults": [1.0, 1.5]},
    # B deep-dive (09): single-module sensitivity of each backbone stage's window attention (qkv + proj of all its
    # blocks), Q0 numerics; works for every AQ_MODEL (rule-based layer set)
    **{f"b_attn_{k}{i}": {"layers": f"all314:inc=backbone.{k}coder_layers.{i}.&.attn.", "rotation": FULL_BF16,
                          "weight": {"method": "rtn", "source": "bf16"}} for k in ("en", "de") for i in range(3)},
    # Perceiver attention-only cards (to_q, to_kv, to_out), Q0 numerics: same scope as the b_attn_* stage cards
    "b_attn_perc_enc": {"layers": "all314:inc=encoder.level_agg.&.0.0.to_", "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}},
    "b_attn_perc_dec": {"layers": "all314:inc=decoder.level_decoder.&.0.0.to_", "rotation": FULL_BF16, "weight": {"method": "rtn", "source": "bf16"}},
    # on-device check of the simulator (scripts/native_w4a16.py): same layers / numerics as TorchAO NVFP4Tensor weight-only
    "n_sim_w4a16_bb": {"layers": "q0:inc=backbone", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "fp32"}, "act": "bf16"},
    "n_sim_w4a4_bb16": {"layers": "q0:inc=backbone", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "bf16"}, "act": "nvfp4"},   # twin of native_w4a4 (mslk kernel)
    "n_sim_w4a16_bb16": {"layers": "q0:inc=backbone", "rotation": {"block": None}, "weight": {"method": "rtn", "source": "bf16"}, "act": "bf16"},   # twin of native_w4a16 (TorchAO quantizes the BF16-cast weight)
    # method candidate M (09): query folding of BOTH Perceivers (K rows exact), everything else quantized
    "m1enc_kv_K": {"layers": "all314:inc=encoder.level_agg.layers.0.0.to_kv", "rotation": FULL_BF16, "weight": {"method": "none", "source": "bf16"},
                   "kv_split": {"pattern": "level_agg.layers.0.0.to_kv", "quant_rows": "K", "alpha": 1.0}},
    "m1enc_kv_V": {"layers": "all314:inc=encoder.level_agg.layers.0.0.to_kv", "rotation": FULL_BF16, "weight": {"method": "none", "source": "bf16"},
                   "kv_split": {"pattern": "level_agg.layers.0.0.to_kv", "quant_rows": "V", "alpha": 1.0}},
    "m1enc_act": {"layers": "all314:inc=encoder.level_agg", "rotation": FULL_BF16, "weight": {"method": "none", "source": "bf16"}},
    **{f"f_both_{k}": {**G02, "layers": "all314:exc=ln_modulation,time_mlp,_embed,level_decoder.layers.0.0.to_q,level_agg.layers.0.0.to_q",
                       **v, "kv_split": [{"pattern": "level_decoder.layers.0.0.to_kv", "quant_rows": "V", "alpha": 1.0},
                                        {"pattern": "level_agg.layers.0.0.to_kv", "quant_rows": "V", "alpha": 1.0}]}
       for k, v in (("w4a4", {}), ("mx", {"act": "mxfp8"}))},
    # robustness for the paper: t7_mx calibrated on a disjoint January week (2024-01-18..24)
    "t7_mx_cal2": {**G02, "layers": T7_SET, "act": "mxfp8", "calib": {"split": "calib2", "steps": 2, "budget_gib": 2.5}},
    # candidate 11 (09): stochastic rounding of NVFP4 activation codes (unbiased) vs RNE, same weights
    # (plain amax block scales on both sides: the 4/6 scale search of G02 is switched off so SR and RNE differ only in rounding)
    "sr_t7": {**G02, "layers": T7_SET, "act": "nvfp4_sr", "act_scale_search": None},
    "rne_t7": {**G02, "layers": T7_SET, "act": "nvfp4", "act_scale_search": None},
    "sr_g02": {**G02, "act": "nvfp4_sr", "act_scale_search": None},
    "rne_g02": {**G02, "act": "nvfp4", "act_scale_search": None},
    "t7_mx_calsum": {**G02, "layers": T7_SET, "act": "mxfp8", "calib": {"split": "summer", "steps": 2, "budget_gib": 2.5}},   # calibrated on July 2024, evaluated on Feb/Mar
    # F3: same-layer-set A16 floor of the g02 weights
    "t7_a16": {**G02, "layers": T7_SET, "act": "bf16"},
    # F4: g02-numerics single-layer cards, same design as m1kv_g02
    **{f"m1g_{k}": {"layers": f"q0:inc={p}", "rotation": {"block": None}, "weight": {"method": "gptq", "source": "fp32"},
                    "act_scale_search": [1.0, 1.5], "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}}
       for k, p in [("to_out", "level_decoder.layers.0.0.to_out"), ("fc1", "level_decoder.layers.0.1.net.0"),
                    ("fc2", "level_decoder.layers.0.1.net.2")]},
})

if __name__ == "__main__":
    R.mkdir(exist_ok=True)
    for name, body in DECK.items():
        (R / f"{name}.json").write_text(json.dumps({"name": name, **body}, indent=1), encoding="utf-8")
    print(len(DECK), "recipes ->", R)
