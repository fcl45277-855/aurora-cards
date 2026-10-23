"""Manuscript figures for the Aurora NVFP4 paper , drawn with matplotlib from the backed-up result files.

Figure contract (one claim per figure, panels with distinct evidence roles):
  Fig. 1  Over five days, quantisation error of Perceiver origin breaks the physical balances and erodes extremes,
          while the deployable recipe tracks full precision.      (test 2024-03-01..06, 20 steps; _diag_fields_test_h20)
  Fig. 2  The fingerprints come from the decoder resampler: error geography and error spectra of four recipes.
          (test 30 days, step 2; _diag_fields_test_h2 + maps)
  Fig. 3  Origin, not magnitude, controls the loss of balance: 15-recipe dose-response.   (dev; _diag_fields_dev_h2{,_dose})
  Fig. 4  Switching dominance predicts attention-module sensitivity across three models; local error does not.
          (_b_eval.json x 3)
  Fig. 5  Deployment: ERA5 excess by lead time (winter and summer rollouts) and the accuracy-storage trade-off.

  python scripts/paper_figures.py [--results results_summary] [--out figures] [--only 1 2]
Writes <out>/fig<N>_<name>.{pdf,png,svg} plus the alignment-audit JSON/SVG of the figure skill.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

RES = Path(os.environ.get("AQ_RESULTS", str(Path(__file__).resolve().parents[1] / "results")))
OUT = Path(__file__).resolve().parents[1] / "figures"
# optional panel-alignment audit (authors' figure toolchain; not part of the release): put its folder on
# AQ_FIGURE_AUDIT to enable it
if os.environ.get("AQ_FIGURE_AUDIT"):
    sys.path.insert(0, os.environ["AQ_FIGURE_AUDIT"])
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except Exception:                                   # the audit helper is optional for a preview run
    require_matplotlib_panel_alignment = None

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5,   # 7.5 keeps mathtext superscripts >= 5 pt
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "figure.dpi": 150,
})
MM = 1 / 25.4
# one neutral family (reference), one signal family (quantised, decoder origin), one accent (deployable recipe)
C = {"era5": "#7f7f7f", "fp32": "#1f1f1f", "q0": "#c0392b", "dec": "#e67e22", "nodec": "#2980b9", "t7mx": "#27ae60",
     "c08": "#8e44ad", "t7": "#16a085", "bf16": "#95a5a6", "summer": "#27ae60"}
LEADS = [1, 2, 4, 8, 12, 16, 20]


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def save(fig, name, panels=True):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / name
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None and panels:
        try:
            require_matplotlib_panel_alignment(fig, json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg",
                                               tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
        except Exception as e:                      # report, do not block the preview
            print(f"  alignment audit: {type(e).__name__}: {str(e)[:200]}")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", base)


def label(ax, s, x=-0.18):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def plain_log(ax, axis="y"):
    """Log axis with plain tick labels (no 4x10^1 style) and no minor labels."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(mpl.ticker.ScalarFormatter())
    a.get_major_formatter().set_scientific(False)
    a.set_minor_formatter(mpl.ticker.NullFormatter())


# ---- Fig. 1: 20-step balances ---------------------------------------------------------------------

def fig1():
    r = jload(RES / "results/_diag_fields_test_h20.json")
    v14 = RES / "results/_diag_fields_test_h20_v14.json"            # t7_a16 on the same six initial dates
    if v14.exists():
        r["recipes"].update({k: v for k, v in jload(v14)["recipes"].items() if k == "t7_a16"})
    steps = sorted(int(s) for s in r["recipes"]["q0_sim"]["per_step"])
    h = [6 * s for s in steps]
    get = lambda name, key: np.array([r["recipes"][name]["per_step"][str(s)][key] for s in steps])
    ref = lambda key, who: np.array([r["recipes"]["q0_sim"]["per_step"][str(s)][who][key] for s in steps])
    fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 120 * MM))
    (a, b), (c, d) = axs
    for ax, key, ylab in ((a, "hydrostatic_residual_rms", "Hydrostatic residual (m$^{2}$ s$^{-2}$)"),
                          (b, "geostrophic_residual_rms", "500 hPa ageostrophic residual (m s$^{-1}$)"),
                          (c, "ws_q999", "99.9th-percentile 10 m wind speed (m s$^{-1}$)")):
        ax.plot(h, ref(key, "truth"), color=C["era5"], ls="--", label="ERA5")
        ax.plot(h, ref(key, "fp32"), color=C["fp32"], label="FP32 Aurora")
        ax.plot(h, get("q0_sim", key), color=C["q0"], marker="o", label="Q0 (W4A4, all 195 layers)")
        ax.plot(h, get("t7_mx", key), color=C["dec"], marker="^", label="t7_mx (W4A8, resamplers protected)")
        if "t7_a16" in r["recipes"]:
            ax.plot(h, get("t7_a16", key), color=C["t7mx"], marker="s", ms=2.5, label="t7_a16 (W4A16, resamplers protected)")
        ax.set_ylabel(ylab)
    d.plot(h, [r["recipes"]["q0_sim"]["per_step"][str(s)]["vs_fp32"]["rmse"]["z500"] for s in steps], color=C["q0"], marker="o")
    d.plot(h, [r["recipes"]["t7_mx"]["per_step"][str(s)]["vs_fp32"]["rmse"]["z500"] for s in steps], color=C["dec"], marker="^")
    if "t7_a16" in r["recipes"]:
        d.plot(h, [r["recipes"]["t7_a16"]["per_step"][str(s)]["vs_fp32"]["rmse"]["z500"] for s in steps], color=C["t7mx"], marker="s", ms=2.5)
    d.set_yscale("log")
    d.set_ylabel("z500 RMSE vs FP32 forecast (m$^{2}$ s$^{-2}$)")
    for ax in axs.ravel():
        ax.set_xlabel("Lead time (h)")
        ax.set_xticks([6, 24, 48, 72, 96, 120])
    b.set_yscale("log"); b.set_yticks([5, 10, 20, 40]); plain_log(b)
    d.set_yticks([3, 10, 30, 100, 300]); plain_log(d)
    b.legend(loc="center left", ncol=1)
    for ax, s in zip(axs.ravel(), "abcd"):
        label(ax, s)
    fig.tight_layout(w_pad=2.0, h_pad=1.5)
    save(fig, "fig1_rollout_balances")


# ---- Fig. 2: maps and spectra ---------------------------------------------------------------------

def fig2():
    r = jload(RES / "results/_diag_fields_test_h2.json")
    maps = np.load(RES / "results/_diag_fields_test_h2_maps.npz")
    dates = r["dates"]
    names = [("q0_sim", "Q0: all 195 layers"), ("m01_q0_decoder_only", "decoder resampler only"),
             ("m02_q0_no_decoder", "all except decoder resampler"), ("t7_mx", "t7_mx")]
    fig = plt.figure(figsize=(180 * MM, 100 * MM))
    gs_top = fig.add_gridspec(1, 4, top=0.90, bottom=0.60, left=0.06, right=0.90, wspace=0.10)
    gs_bot = fig.add_gridspec(1, 2, top=0.47, bottom=0.17, left=0.08, right=0.98, wspace=0.42)
    gs = {(0, i): gs_top[0, i] for i in range(4)}
    gs["bottom_left"] = gs_bot[0, 0]; gs["bottom_right"] = gs_bot[0, 1]    # str keys: slices are unhashable before 3.12
    lat = np.linspace(90, -90, 90, endpoint=False) - 1.0
    lon = np.arange(-180, 180, 2.0)
    mean_map = lambda n: np.roll(np.mean([maps[f"{n}_vs_fp32__s2__{d}__z500"] for d in dates], axis=0), 90, axis=1)
    mm = {n: mean_map(n) for n, _ in names}                          # 0..360 -> -180..180
    # one logarithmic scale for all four maps: from the t7_mx 2nd to the q0 99.5th percentile
    norm = mpl.colors.LogNorm(vmin=max(np.percentile(mm["t7_mx"], 2), 1e-3), vmax=np.percentile(mm["q0_sim"], 99.5))
    lsm_p = RES / "results/_lsm_2deg.npz"                            # 2-deg land fraction from Aurora's static fields
    lsm = np.roll(np.load(lsm_p)["lsm"], 90, axis=1) if lsm_p.exists() else None
    axes = []
    for i, (n, title) in enumerate(names):
        ax = fig.add_subplot(gs[(0, i)])
        im = ax.pcolormesh(lon, lat, np.clip(mm[n], norm.vmin, None), cmap="magma_r", norm=norm, shading="auto", rasterized=True)
        if lsm is not None:
            ax.contour(lon, lat, lsm, levels=[0.5], colors="0.35", linewidths=0.3)
        ax.set_aspect("equal"); ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_title(title, fontsize=6.5)
        ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xticks([-120, 0, 120]); ax.set_xticklabels(["120°W", "0°", "120°E"])
        ax.set_yticks([-60, 0, 60]); ax.set_yticklabels(["60°S", "0°", "60°N"] if i == 0 else [])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4)
        axes.append(ax)
    cax = fig.add_axes([0.92, 0.60, 0.012, 0.25])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("z500 squared error\nvs FP32, step 2 (m$^{4}$ s$^{-4}$, log)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6)
    # spectra
    edges = r["k_edges"]
    kmid = [np.sqrt(edges[i] * edges[i + 1]) for i in range(len(edges) - 1)]
    ax = fig.add_subplot(gs["bottom_left"])
    fp_err = np.mean([r["fp32"]["2"][d]["vs_truth"]["err_spectra"]["z500"] for d in dates], axis=0)
    ax.plot(kmid, fp_err, color=C["fp32"], ls="--", label="FP32 forecast error (vs ERA5)")
    for (n, title), col, mk in zip(names, (C["q0"], C["dec"], C["nodec"], C["t7mx"]), "osD^"):
        sp = np.mean([r["recipes"][n]["per_date"]["2"][d]["vs_fp32"]["err_spectra"]["z500"] for d in dates], axis=0)
        ax.plot(kmid, sp, color=col, marker=mk, label=title)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 10, 100, 500]); plain_log(ax, "x")
    ax.set_yticks([0.1, 1, 10, 100, 1000]); plain_log(ax)
    ax.set_xlabel("Zonal wavenumber"); ax.set_ylabel("z500 error power (m$^{4}$ s$^{-4}$)")
    ax.axvline(50, color="0.7", lw=0.6, ls=":")
    ax.text(50, ax.get_ylim()[0] * 1.3, " k = 50", fontsize=6, color="0.4", va="bottom")
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2)
    label(ax, "e", x=-0.14)
    ax2 = fig.add_subplot(gs["bottom_right"])
    keys = [("hydrostatic_residual_rms", "hydrostatic\nresidual"), ("superadiabatic_share", "super-adiabatic\nshare"),
            ("geostrophic_residual_rms", "ageostrophic\nresidual")]
    x = np.arange(len(keys)); w = 0.2
    fp = {k: np.mean([r["fp32"]["2"][d][k] for d in dates]) for k, _ in keys}
    for j, ((n, title), col) in enumerate(zip(names, (C["q0"], C["dec"], C["nodec"], C["t7mx"]))):
        vals = [np.mean([r["recipes"][n]["per_date"]["2"][d][k] for d in dates]) / fp[k] for k, _ in keys]
        ax2.bar(x + (j - 1.5) * w, vals, w, color=col, label=title)
    ax2.axhline(1, color=C["fp32"], lw=0.8, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels([t for _, t in keys])
    ax2.set_ylabel("Ratio to FP32 forecast (step 2)")
    ax2.set_yscale("log")
    ax2.set_yticks([1, 2, 4, 8]); plain_log(ax2)
    label(ax2, "f", x=-0.16)
    for ax, s in zip(axes, "abcd"):
        ax.text(0.0, 1.12, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    save(fig, "fig2_fingerprints_attribution", panels=False)


# ---- Fig. 3: dose-response ------------------------------------------------------------------------

DEC_Q = {"m1kv_K_a025": 1, "m1kv_mx": 1, "t7_mx": 0, "c08_weight_only_gptq": 1, "m01w_dec_weight_only": 1, "t7mx_all314": 0,
         "m1kv_K_a05": 1, "g02_svdq_a08_r32": 1, "m02_q0_no_decoder": 0, "t7_all314": 0, "d06_mx": 1, "m1kv_K": 1,
         "m01a_dec_act_only": 1, "m01_q0_decoder_only": 1, "q0_sim": 1}


def fig3():
    rows = {}
    for f in ("results/_diag_fields_dev_h2.json", "results/_diag_fields_dev_h2_dose.json"):
        r = jload(RES / f)
        for n, R in r["recipes"].items():
            a = R["per_step"]["2"]
            rows[n] = dict(z=a["vs_fp32"]["rmse"]["z500"], ageo=a["geostrophic_residual_rms"], hydro=a["hydrostatic_residual_rms"],
                           sup=a["superadiabatic_share"], ss=a["vs_fp32"]["err_small_share"]["z500"], vc=a["vs_fp32"]["vertical_err_corr"]["t"],
                           fp_ageo=a["fp32"]["geostrophic_residual_rms"], fp_hydro=a["fp32"]["hydrostatic_residual_rms"], fp_sup=a["fp32"]["superadiabatic_share"])
    fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 115 * MM))
    panels = [("ageo", "500 hPa ageostrophic residual (m s$^{-1}$)", "fp_ageo"), ("hydro", "Hydrostatic residual (m$^{2}$ s$^{-2}$)", "fp_hydro"),
              ("ss", "Small-scale share of z500 error (k > 50)", None), ("vc", "Vertical correlation of T error", None)]
    for ax, (key, ylab, fpkey), s in zip(axs.ravel(), panels, "abcd"):
        for n, v in rows.items():
            dec = DEC_Q[n]
            col = C["dec"] if dec else C["nodec"]
            mk = "o" if dec else "s"
            ax.scatter(v["z"], v[key], color=col, marker=mk, s=16, zorder=3, edgecolor="white", linewidth=0.4)
        if fpkey:
            ax.axhline(np.mean([v[fpkey] for v in rows.values()]), color=C["fp32"], ls="--", lw=0.8)
            ax.text(135, np.mean([v[fpkey] for v in rows.values()]) * (1.02 if key == "ageo" else 1.012), "FP32 ", color=C["fp32"],
                    va="bottom", ha="right", fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("z500 RMSE vs FP32 forecast, step 2 (m$^{2}$ s$^{-2}$)")
        ax.set_ylabel(ylab)
        label(ax, s)
    for n, txt, off in (("t7_mx", "t7_mx", (3, 3)), ("q0_sim", "Q0 / decoder only", (-7, -12)), ("m02_q0_no_decoder", "no decoder", (3, 3)),
                        ("t7_all314", "t7 all layers", (3, -7)), ("d06_mx", "d06_mx (W4A8, no smoothing)", (-4, -9))):
        v = rows[n]
        axs[0, 0].annotate(txt, (v["z"], v["ageo"]), xytext=off, textcoords="offset points", fontsize=5.5,
                           ha="right" if off[0] < 0 else "left")
    axs[0, 0].set_yscale("log"); axs[0, 0].set_yticks([5, 10, 20, 40]); plain_log(axs[0, 0])
    for ax in axs.ravel():
        ax.set_xticks([3, 10, 30, 100]); plain_log(ax, "x")
    h1 = plt.Line2D([], [], color=C["dec"], marker="o", ls="", label="decoder resampler quantised")
    h2 = plt.Line2D([], [], color=C["nodec"], marker="s", ls="", label="decoder resampler protected")
    axs[0, 1].legend(handles=[h1, h2], loc="upper left")
    fig.tight_layout(w_pad=2.0, h_pad=1.5)
    save(fig, "fig3_dose_response")


# ---- Fig. 4: predictor ----------------------------------------------------------------------------

def spearman(x, y):
    """Spearman rho with average ranks for ties (equal to scipy.stats.spearmanr)."""
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fig4():
    models = [("results", "Aurora 1.3B pretrained", "o"), ("results_finetuned", "Aurora 1.3B finetuned", "s"), ("results_small", "Aurora-small", "^")]
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM))
    preds = [("err_in_flip", "Share of attention error in switched cells"), ("A", "Logit amplification A"), ("yerr", "Local activation error (yerr)")]
    cols = [C["fp32"], C["q0"], C["nodec"]]
    for ax, (pk, xlab), s in zip(axs, preds, "abc"):
        rhos = []
        for (d, name, mk), col in zip(models, cols):
            r = jload(RES / d / "_b_eval.json")
            mods = list(r["target"])
            x = np.array([r["pred"][m].get(pk, np.nan) for m in mods]); y = np.array([r["target"][m] for m in mods])
            ok = ~np.isnan(x)
            rho = spearman(x[ok], y[ok])
            ax.scatter(x[ok], y[ok], color=col, marker=mk, s=18, edgecolor="white", linewidth=0.4, label=name)
            rhos.append((name, rho, col))
        ax.set_yscale("log"); ax.set_yticks([10, 100, 1000, 10000]); plain_log(ax)
        if pk == "A":
            ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10]); plain_log(ax, "x")
        ax.set_xlabel(xlab)
        for j, (name, rho, col) in enumerate(rhos):
            ax.text(0.98, 0.04 + 0.085 * j, f"ρ = {rho:+.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=col)
        label(ax, s)
    handles, labels_ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 1.04))
    a0 = axs[0]
    a0.text(0.33, 20000, "Perceiver resamplers →", fontsize=6, ha="left", va="center", color="0.3")
    a0.text(0.30, 150, "← Swin stages", fontsize=6, ha="left", va="center", color="0.3")
    axs[0].set_ylabel("Attention-module sensitivity\n($\\mathrm{qerr}^2 - \\mathrm{qerr}^2_{\\mathrm{BF16}}$, dev)")
    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.95))
    save(fig, "fig4_switching_predictor")


# ---- Fig. 5: deployment ---------------------------------------------------------------------------

def h20_rows(split, name):
    return [json.loads(l) for l in open(RES / f"results/{name}__{split}_h20/records.jsonl", encoding="utf-8") if l.strip()]


def excess_by_date(split, name, dates=None):
    """{step: {date: mean ERA5 excess over the nine variables (%)}} for a 20-step card, paired with FP32 by date."""
    fp = {(r["date"], r["step"], r["var"]): r["rmse_truth"] for r in h20_rows(split, "fp32")}
    per = defaultdict(lambda: defaultdict(list))
    for r in h20_rows(split, name):
        k = (r["date"], r["step"], r["var"])
        if k in fp and (dates is None or r["date"] in dates):
            per[r["step"]][r["date"]].append(100 * (r["rmse_truth"] / fp[k] - 1))
    return {s_: {d: float(np.mean(v)) for d, v in dd.items()} for s_, dd in per.items()}


def block_interval(vals_by_date, n=2000, L=3, seed=0):
    """3-day non-circular moving-block bootstrap of the mean over initial dates (same rule as the tables)."""
    dates = sorted(vals_by_date)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = []
        while len(pick) < len(dates):
            st = int(rng.integers(0, len(dates) - L + 1))
            pick.extend(dates[st:st + L])
        out.append(np.mean([vals_by_date[d] for d in pick[:len(dates)]]))
    return np.percentile(out, [2.5, 97.5])


WINTER12 = [f"2024-03-{d:02d}" for d in range(1, 13)]
# (label, storage relative to BF16 from scripts/memory_table.py, test-month qerr %, group)
STORAGE = [
    ("BF16", 1.00, 1.75, "ref"),
    ("q0", 0.49, 147.24, "w4u"), ("g02", 0.52, 33.62, "w4u"), ("c08 (Q0 set)", 0.49, 14.20, "w4u"),
    ("t7 (W4A4)", 0.55, 17.32, "w4p"), ("t7_mx (W4A8)", 0.55, 10.56, "w4p"), ("t7_a16 (W4A16)", 0.55, 9.46, "w4p"),
    ("c08@t7", 0.53, 10.61, "w4p"), ("all-layer W4A8", 0.39, 17.09, "w4p"), ("all-layer W4A16", 0.39, 14.38, "w4p"),
    ("W8A16 MX", 0.52, 18.46, "w8u"), ("W8A16 per-ch.", 0.50, 17.80, "w8u"), ("W8A8 MX", 0.52, 107.29, "w8u"),
    ("W8A8 per-ch.", 0.50, 97.71, "w8u"), ("protected W8A16", 0.57, 10.40, "w8p"), ("protected W8A8", 0.57, 25.09, "w8p"),
]
GROUP = {"ref": ("BF16 autocast", C["bf16"], "o"), "w4u": ("NVFP4, resamplers quantised", C["q0"], "o"),
         "w4p": ("NVFP4, resamplers protected", C["t7mx"], "s"), "w8u": ("FP8, all 314 layers", "#2980b9", "D"),
         "w8p": ("FP8, resamplers protected", "#7fb3d5", "D")}


def fig5():
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
    curves = (("t7_a16", C["t7mx"], "t7_a16 (W4A16)", "s", "-"), ("t7_mx", C["dec"], "t7_mx (W4A8)", "^", "-"),
              ("c08_t7set", C["c08"], "c08@t7 (W4A16)", "D", "-"), ("u8_w8a16_mx_prot", "#2980b9", "protected FP8 W8A16", "o", "-"))
    for name, col, lab, mk, ls in curves:
        try:
            per = excess_by_date("test", name, set(WINTER12))
        except FileNotFoundError:
            continue
        steps = sorted(per)
        h = [6 * s_ for s_ in steps]
        a.plot(h, [np.mean(list(per[s_].values())) for s_ in steps], color=col, marker=mk, ls=ls, label=lab)
        if name == "t7_a16":
            ci = np.array([block_interval(per[s_]) for s_ in steps])
            a.fill_between(h, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0, label="t7_a16, 95 % block bootstrap")
    try:
        per = excess_by_date("summer", "t7_mx")
        steps = sorted(per)
        a.plot([6 * s_ for s_ in steps], [np.mean(list(per[s_].values())) for s_ in steps], color=C["dec"], marker="^",
               ls="--", mfc="white", label="t7_mx, July (3 dates)")
    except FileNotFoundError:
        pass
    a.axhline(0, color="0.6", lw=0.6)
    a.set_xlabel("Lead time (h)"); a.set_ylabel("ERA5 RMSE excess over FP32 (%)")
    a.set_xticks([6, 24, 48, 72, 96, 120])
    a.legend(loc="lower left", fontsize=5.5, ncol=1)
    label(a, "a", x=-0.14)
    seen = set()
    for lab, st, q, g in STORAGE:
        name, col, mk = GROUP[g]
        b.scatter(st, q, color=col, marker=mk, s=20, edgecolor="white", linewidth=0.4, zorder=3,
                  label=None if g in seen else name)
        seen.add(g)
    # labels in free space with thin leader lines (data coordinates of the text)
    place = {"t7_a16 (W4A16)": (0.66, 6.6), "t7_mx (W4A8)": (0.66, 15.5), "protected W8A16": (0.66, 10.0),
             "W8A16 MX": (0.66, 24), "all-layer W4A16": (0.31, 7.0), "all-layer W4A8": (0.31, 30),
             "c08 (Q0 set)": (0.40, 4.3), "W8A8 MX": (0.58, 62), "q0": (0.36, 150), "BF16": (0.86, 2.6)}
    for lab, st, q, g in STORAGE:
        if lab in place:
            b.annotate(lab, (st, q), xytext=place[lab], textcoords="data", fontsize=5.3, va="center",
                       arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=0, shrinkB=2.5))
    b.set_yscale("log"); b.set_yticks([2, 5, 10, 20, 50, 100, 200]); plain_log(b)
    b.set_xlabel("Weight storage relative to BF16"); b.set_ylabel("qerr on the test month (%)")
    b.set_xlim(0.3, 1.08); b.set_xticks([0.4, 0.5, 0.6, 0.8, 1.0])
    b.legend(loc="upper right", fontsize=5.3, handletextpad=0.3)
    label(b, "b", x=-0.2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_deployment")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out:
        OUT = Path(a.out)
    for i, fn in ((1, fig1), (2, fig2), (3, fig3), (4, fig4), (5, fig5)):
        if a.only is None or i in a.only:
            fn()
