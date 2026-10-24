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
  Fig. 6  Where the deployable recipe departs from FP32: storm tracks at 120 h, high terrain at 6 h (msl); 2-deg
          RMS maps and zonal means.   (test 2024-03-01..06, 20 steps; _diag_fields_test_h20{,_v14}_maps)

  python scripts/paper_figures.py [--results results_summary] [--out figures] [--only 1 2] [--esi]
Writes <out>/fig<N>_<name>.{pdf,png,svg} plus the alignment-audit JSON/SVG of the figure skill.

--esi produces the Earth Science Informatics artwork in figures_esi/ (or --out): figures 174 mm wide (single-column
text area), lettering >= 8 pt, lines >= 0.5 pt, fonts embedded, files named Fig1..Fig6 in the order the figures are
cited in the ESI manuscript, as .pdf, .eps (vector, preferred by the journal), .tiff (600 dpi, combination art) and
.png (preview). Every figure is checked before it is written and the run stops if a limit is violated.
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
# ---- journal profiles -----------------------------------------------------------------------------
FIGW = 180 * MM                                     # C&G default; --esi sets 174 mm
ESI = False
ESI_WIDTH_MM = 174.0                                # ESI: single-column text area 174 mm, height <= 234 mm
ESI_MIN_FONT = 8.0                                  # ESI: lettering about 8-12 pt
ESI_MIN_LW = 0.5                                    # ESI: lines >= 0.3 pt; 0.5 pt keeps a margin after rescaling
ESI_RC = {"font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],   # Liberation Sans = Arial metrics (Linux)
          "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
          "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "ps.fonttype": 42,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5}
ESI_HEIGHT_MM = {"fig1_rollout_balances": 132, "fig2_fingerprints_attribution": 110, "fig3_dose_response": 128,
                 "fig4_switching_predictor": 72, "fig5_deployment": 84, "fig6_spatial_departure": 86}
# internal figure names -> ESI file names (order of citation in the ESI manuscript)
ESI_NAME = {"fig4_switching_predictor": "Fig1", "fig5_deployment": "Fig2", "fig2_fingerprints_attribution": "Fig3",
            "fig3_dose_response": "Fig4", "fig1_rollout_balances": "Fig5", "fig6_spatial_departure": "Fig6"}


def fsize(name, h_mm):
    """Figure size: journal width and the per-figure height (taller for ESI to make room for 8 pt lettering)."""
    return (FIGW, (ESI_HEIGHT_MM[name] if ESI else h_mm) * MM)


def lighten(color, alpha):
    """Opaque equivalent of `color` at `alpha` over white: EPS has no transparency."""
    r, g, b = mpl.colors.to_rgb(color)
    return (1 - alpha * (1 - r), 1 - alpha * (1 - g), 1 - alpha * (1 - b))


def _texts(fig):
    return [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]


def enforce_minimums(fig, min_font, min_lw):
    """Raise every visible text below min_font and every drawn line/edge below min_lw to the limit."""
    for t in _texts(fig):
        if t.get_fontsize() < min_font:
            t.set_fontsize(min_font)
    for ln in fig.findobj(mpl.lines.Line2D):
        if 0 < ln.get_linewidth() < min_lw:
            ln.set_linewidth(min_lw)
        if 0 < ln.get_markeredgewidth() < min_lw and ln.get_marker() not in (None, "", "None", " "):
            ln.set_markeredgewidth(min_lw)
    for p in fig.findobj(mpl.patches.Patch):
        if 0 < p.get_linewidth() < min_lw:
            p.set_linewidth(min_lw)
    for c in fig.findobj(mpl.collections.Collection):
        lws = np.atleast_1d(c.get_linewidths())
        if lws.size and np.any((lws > 0) & (lws < min_lw)):
            c.set_linewidths(np.where((lws > 0) & (lws < min_lw), min_lw, lws))
    for ax in fig.axes:
        # tick labels are regenerated on redraw from the stored tick parameters: raise those, not only the texts
        for axis, which in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            sizes = [t.get_fontsize() for t in axis.get_ticklabels(which="both")]
            if sizes and min(sizes) < min_font:
                ax.tick_params(axis=which, which="both", labelsize=min_font)
        for sp in ax.spines.values():
            if 0 < sp.get_linewidth() < min_lw:
                sp.set_linewidth(min_lw)
        for axis in (ax.xaxis, ax.yaxis):
            for tl in axis.get_majorticklines() + axis.get_minorticklines():
                if 0 < tl.get_markeredgewidth() < min_lw:
                    tl.set_markeredgewidth(min_lw)


def audit_minimums(fig, min_font, min_lw):
    """Return the violations left after layout (should be empty)."""
    bad = [f"text {t.get_text()[:30]!r} at {t.get_fontsize():.2f} pt" for t in _texts(fig) if t.get_fontsize() < min_font - 1e-6]
    bad += [f"line at {ln.get_linewidth():.2f} pt" for ln in fig.findobj(mpl.lines.Line2D)
            if ln.get_visible() and 0 < ln.get_linewidth() < min_lw - 1e-6]
    bad += [f"{type(c).__name__} edge at {np.min(np.atleast_1d(c.get_linewidths())):.2f} pt"
            for c in fig.findobj(mpl.collections.Collection)
            if c.get_visible() and np.any((np.atleast_1d(c.get_linewidths()) > 0) & (np.atleast_1d(c.get_linewidths()) < min_lw - 1e-6))]
    return bad


def fit_width(fig, layout, target_in, pad_in):
    """Resize the figure so that the saved, tightly cropped width equals target_in (layout re-run each time)."""
    for _ in range(6):
        if layout is not None:
            fig.tight_layout(**layout)
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        width = bb.width + 2 * pad_in
        if abs(width - target_in) < 0.2 * MM:
            return width
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * (target_in - 2 * pad_in) / bb.width, h, forward=True)
    return width
# one neutral family (reference), one signal family (quantised, decoder origin), one accent (deployable recipe)
C = {"era5": "#7f7f7f", "fp32": "#1f1f1f", "q0": "#c0392b", "dec": "#e67e22", "nodec": "#2980b9", "t7mx": "#27ae60",
     "c08": "#8e44ad", "t7": "#16a085", "bf16": "#95a5a6", "summer": "#27ae60"}
LEADS = [1, 2, 4, 8, 12, 16, 20]


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def save(fig, name, panels=True, layout=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if ESI:
        save_esi(fig, name, layout)
        return
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


def save_esi(fig, name, layout):
    """ESI artwork: enforce the lettering and line limits, fit the width, check, write FigN.{pdf,eps,tiff,png}."""
    pad = 0.01                                                     # inches of white margin around the artwork
    enforce_minimums(fig, ESI_MIN_FONT, ESI_MIN_LW)
    width = fit_width(fig, layout, ESI_WIDTH_MM * MM, pad)
    enforce_minimums(fig, ESI_MIN_FONT, ESI_MIN_LW)                # artists created by the re-layout (e.g. legends)
    fig.canvas.draw()
    bad = audit_minimums(fig, ESI_MIN_FONT, ESI_MIN_LW)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w_mm, h_mm = (bb.width + 2 * pad) / MM, (bb.height + 2 * pad) / MM
    if bad or w_mm > ESI_WIDTH_MM + 0.5 or h_mm > 234:
        raise SystemExit(f"{name}: ESI check failed: {w_mm:.1f} x {h_mm:.1f} mm; " + "; ".join(bad[:8]))
    base = OUT / ESI_NAME[name]
    kw = dict(bbox_inches="tight", pad_inches=pad)
    fig.savefig(f"{base}.pdf", dpi=600, **kw)                      # dpi applies to the rasterised map layers only
    fig.savefig(f"{base}.eps", dpi=600, **kw)
    fig.savefig(f"{base}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(f"{base}.png", dpi=300, **kw)
    plt.close(fig)
    print(f"wrote {base}.{{pdf,eps,tiff,png}} ({name}): {w_mm:.1f} x {h_mm:.1f} mm, "
          f"lettering >= {ESI_MIN_FONT:g} pt, lines >= {ESI_MIN_LW:g} pt")


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
    fig, axs = plt.subplots(2, 2, figsize=fsize("fig1_rollout_balances", 120))
    (a, b), (c, d) = axs
    for ax, key, ylab in ((a, "hydrostatic_residual_rms", "Hydrostatic residual (m$^{2}$ s$^{-2}$)"),
                          (b, "geostrophic_residual_rms", "500 hPa ageostrophic residual (m s$^{-1}$)"),
                          (c, "ws_q999", "99.9th pct. 10 m wind (m s$^{-1}$)" if ESI else "99.9th-percentile 10 m wind speed (m s$^{-1}$)")):
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
    save(fig, "fig1_rollout_balances", layout=dict(w_pad=2.0, h_pad=1.5))


# ---- Fig. 2: maps and spectra ---------------------------------------------------------------------

def fig2():
    r = jload(RES / "results/_diag_fields_test_h2.json")
    maps = np.load(RES / "results/_diag_fields_test_h2_maps.npz")
    dates = r["dates"]
    names = [("q0_sim", "Q0: all 195 layers"), ("m01_q0_decoder_only", "decoder resampler only"),
             ("m02_q0_no_decoder", "all except decoder resampler"), ("t7_mx", "t7_mx")]
    fig = plt.figure(figsize=fsize("fig2_fingerprints_attribution", 100))
    gs_top = fig.add_gridspec(1, 4, top=0.90, bottom=0.60, left=0.06, right=0.90, wspace=0.10)
    gs_bot = fig.add_gridspec(1, 2, top=0.47, bottom=0.22 if ESI else 0.17, left=0.08, right=0.98, wspace=0.42)
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
        ax.set_title({"m01_q0_decoder_only": "decoder only", "m02_q0_no_decoder": "all but decoder"}.get(n, title)
                     if ESI else title, fontsize=6.5)
        ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xticks([-120, 0, 120]); ax.set_xticklabels(["120°W", "0°", "120°E"])
        ax.set_yticks([-60, 0, 60]); ax.set_yticklabels(["60°S", "0°", "60°N"] if i == 0 else [])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4)
        axes.append(ax)
    cax = fig.add_axes([0.92, 0.58, 0.012, 0.32] if ESI else [0.92, 0.60, 0.012, 0.25])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("z500 squared error\nvs FP32 (m$^{4}$ s$^{-4}$)" if ESI else
                 "z500 squared error\nvs FP32, step 2 (m$^{4}$ s$^{-4}$, log)", fontsize=7.5)
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
    fig.legend(handles, labels_, loc="lower center", ncol=3 if ESI else 5, fontsize=6, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.2)
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
    fig, axs = plt.subplots(2, 2, figsize=fsize("fig3_dose_response", 115))
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
        if ESI and n == "t7_mx":                              # label in free space with a leader line to its point
            axs[0, 0].annotate(txt, (v["z"], v["ageo"]), xytext=(2.3, 7.3), textcoords="data", fontsize=5.5,
                               arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=0, shrinkB=2.5))
            continue
        axs[0, 0].annotate(txt, (v["z"], v["ageo"]), xytext=off, textcoords="offset points", fontsize=5.5,
                           ha="right" if off[0] < 0 else "left")
    axs[0, 0].set_yscale("log"); axs[0, 0].set_yticks([5, 10, 20, 40]); plain_log(axs[0, 0])
    for ax in axs.ravel():
        ax.set_xticks([3, 10, 30, 100]); plain_log(ax, "x")
    h1 = plt.Line2D([], [], color=C["dec"], marker="o", ls="", label="decoder resampler quantised")
    h2 = plt.Line2D([], [], color=C["nodec"], marker="s", ls="", label="decoder resampler protected")
    axs[0, 1].legend(handles=[h1, h2], loc="upper left")
    fig.tight_layout(w_pad=2.0, h_pad=1.5)
    save(fig, "fig3_dose_response", layout=dict(w_pad=2.0, h_pad=1.5))


# ---- Fig. 4: predictor ----------------------------------------------------------------------------

def spearman(x, y):
    """Spearman rho with average ranks for ties (equal to scipy.stats.spearmanr)."""
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fig4():
    models = [("results", "Aurora 1.3B pretrained", "o"), ("results_finetuned", "Aurora 1.3B finetuned", "s"), ("results_small", "Aurora-small", "^")]
    fig, axs = plt.subplots(1, 3, figsize=fsize("fig4_switching_predictor", 58))
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
        x0, y0, ha = (0.03, 0.50, "left") if (ESI and pk == "err_in_flip") else (0.98, 0.04, "right")
        for j, (name, rho, col) in enumerate(rhos):
            ax.text(x0, y0 + 0.085 * j, f"ρ = {rho:+.2f}", transform=ax.transAxes, ha=ha, va="bottom", fontsize=6, color=col)
        label(ax, s)
    handles, labels_ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 1.04))
    a0 = axs[0]
    a0.text(0.12 if ESI else 0.33, 20000, "Perceiver resamplers →", fontsize=6, ha="left", va="center", color="0.3")
    a0.text(0.30, 150, "← Swin stages", fontsize=6, ha="left", va="center", color="0.3")
    axs[0].set_ylabel("Attention-module sensitivity\n($\\mathrm{qerr}^2 - \\mathrm{qerr}^2_{\\mathrm{BF16}}$, dev)")
    rect = (0, 0, 1, 0.92 if ESI else 0.95)                         # room for the 8 pt legend above the panels
    fig.tight_layout(w_pad=2.0, rect=rect)
    save(fig, "fig4_switching_predictor", layout=dict(w_pad=2.0, rect=rect))


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
    fig, (a, b) = plt.subplots(1, 2, figsize=fsize("fig5_deployment", 66), gridspec_kw={"width_ratios": [1.25, 1]})
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
            band = dict(color=lighten(col, 0.15), zorder=0) if ESI else dict(color=col, alpha=0.15)
            a.fill_between(h, ci[:, 0], ci[:, 1], lw=0, label="t7_a16, 95 % block bootstrap", **band)
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
    if ESI:
        place["W8A8 MX"] = (0.33, 75)
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
    save(fig, "fig5_deployment", layout=dict(w_pad=2.5))


# ---- Fig. 6: where the deployable configuration departs from FP32 ----------------------------------

def fig6():
    """RMS maps over the six 20-step test rollouts (2024-03-01..06) on the 2-deg grid of scripts/diag_fields.py:
    FP32 vs ERA5, t7_a16 vs FP32 and q0 vs FP32, with zonal-mean RMS; z500 at 120 h and msl at 6 h."""
    a16 = np.load(RES / "results/_diag_fields_test_h20_v14_maps.npz")          # FP32 vs ERA5 and t7_a16
    qz = np.load(RES / "results/_diag_fields_test_h20_maps.npz")               # q0 on the same dates
    dates = jload(RES / "results/_diag_fields_test_h20_v14.json")["dates"]
    lat = np.linspace(90, -90, 90, endpoint=False) - 1.0
    lon = np.arange(-180, 180, 2.0)
    wts = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (90, 180))
    lsm_p = RES / "results/_lsm_2deg.npz"
    lsm = np.roll(np.load(lsm_p)["lsm"], 90, axis=1) if lsm_p.exists() else None
    src = {"fp32": (a16, "fp32_vs_truth"), "t7_a16": (a16, "t7_a16_vs_fp32"), "q0_sim": (qz, "q0_sim_vs_fp32")}
    rms = lambda n, s, v: np.roll(np.sqrt(np.mean([src[n][0][f"{src[n][1]}__s{s}__{d}__{v}"] for d in dates], axis=0)),
                                  90, axis=1)                                   # 0..360 -> -180..180
    rows = (("z500", 20, "z500, 120 h", 1.0, "m$^{2}$ s$^{-2}$"), ("msl", 1, "msl, 6 h", 0.01, "hPa"))
    cols = (("fp32", "FP32 vs ERA5", C["fp32"]), ("t7_a16", "t7_a16 vs FP32", C["t7mx"]), ("q0_sim", "q0 vs FP32", C["q0"]))
    fig = plt.figure(figsize=fsize("fig6_spatial_departure", 80))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.045, 0.22, 0.62], left=0.07, right=0.93, top=0.93,
                          bottom=0.12, wspace=0.10, hspace=0.50)    # column 4 is a spacer for the colour-bar labels

    def ticks(lo, hi):                                                 # 1-2-5 ticks with plain labels on a log axis
        t = [m * 10.0 ** k for k in range(-3, 5) for m in (1, 2, 5) if lo <= m * 10.0 ** k <= hi]
        return t, [f"{x:g}" for x in t]
    letters = iter("abcdefgh")
    for i, (v, s, rlab, scale, unit) in enumerate(rows):
        mm = {n: rms(n, s, v) * scale for n, _, _ in cols}
        norm = mpl.colors.LogNorm(vmin=np.percentile(mm["t7_a16"], 2),
                                  vmax=np.percentile(np.concatenate([mm["fp32"].ravel(), mm["q0_sim"].ravel()]), 99.5))
        for j, (n, title, col) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            im = ax.pcolormesh(lon, lat, np.clip(mm[n], norm.vmin, None), cmap="magma_r", norm=norm, shading="auto",
                               rasterized=True)
            if lsm is not None:
                ax.contour(lon, lat, lsm, levels=[0.5], colors="0.35", linewidths=0.5)
            ax.set_aspect("equal"); ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_xticks([-120, 0, 120]); ax.set_xticklabels(["120°W", "0°", "120°E"] if i == 1 else [])
            ax.set_yticks([-60, 0, 60]); ax.set_yticklabels(["60°S", "0°", "60°N"] if j == 0 else [])
            ax.tick_params(length=1.5, pad=1)
            if i == 0:
                ax.set_title(title, color=col, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{rlab}\nRMS ({unit})")
            g = np.sqrt(np.average(mm[n] ** 2, weights=wts))                    # global RMS, cos-lat weighted
            ax.text(0.03, 0.05, f"{g:.3g}", transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            ax.text(0.0, 1.03, next(letters), transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, 3]))
        t, tl = ticks(norm.vmin, norm.vmax)
        cb.set_ticks(t); cb.set_ticklabels(tl); cb.ax.minorticks_off()
        z = fig.add_subplot(gs[i, 5])
        for n, title, col in cols:
            z.plot(np.sqrt(np.mean(mm[n] ** 2, axis=1)), lat, color=col, lw=1.0, label=title)
        z.set_xscale("log"); z.set_ylim(-90, 90)
        z.set_yticks([-60, 0, 60]); z.set_yticklabels(["60°S", "0°", "60°N"])
        z.yaxis.tick_right()
        lo, hi = z.get_xlim()
        t, tl = ticks(lo, hi)
        t, tl = (t[::2], tl[::2]) if len(t) > 4 else (t, tl)
        z.set_xticks(t); z.set_xticklabels(tl); z.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        z.set_xlabel(f"Zonal RMS ({unit})")
        z.text(0.0, 1.03, next(letters), transform=z.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    save(fig, "fig6_spatial_departure", panels=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--esi", action="store_true", help="Earth Science Informatics artwork: 174 mm, >= 8 pt, >= 0.5 pt, Fig1..Fig6")
    a = ap.parse_args()
    if a.esi:
        ESI, FIGW = True, ESI_WIDTH_MM * MM
        mpl.rcParams.update(ESI_RC)
        OUT = OUT.parent / "figures_esi"
    if a.out:
        OUT = Path(a.out)
    for i, fn in ((1, fig1), (2, fig2), (3, fig3), (4, fig4), (5, fig5), (6, fig6)):
        if a.only is None or i in a.only:
            fn()
