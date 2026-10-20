"""Print the PhysMetrics-definition rows for the manuscript from results*/_physmetrics_<split>_h<steps>.json.

  python scripts/physmetrics_table.py results/_physmetrics_test_h2.json [results/_physmetrics_test_h20.json]
Rows (per recipe, per step): hydrostatic RMSE 500-850 (model, ERA5, excess), geostrophic RMSE 500 (model, ERA5, excess),
L_eff vs ERA5 and vs FP32, small-scale energy ratio, spectral divergence / residual. Markdown table on stdout.
"""
import json
import sys
from pathlib import Path


def main():
    for p in sys.argv[1:]:
        r = json.load(open(p, encoding="utf-8"))
        print(f"\n### {Path(p).name}  (model {r['model']}, split {r['split']}, {len(r['dates'])} dates, {r['steps']} steps)\n")
        steps = sorted({s for R in r["recipes"].values() for s in R["per_step"]}, key=int)
        show = [s for s in steps if int(s) in (1, 2, 4, 8, 12, 20)]
        print("| Forecast | step | hydrostatic RMSE 500–850 (ERA5 / FP32 / model) | excess vs ERA5 | geostrophic RMSE 500 (ERA5 / FP32 / model) | excess vs ERA5 | L_eff km (FP32 vs ERA5 / model vs ERA5 / model vs FP32) | small-scale ratio vs FP32 | spectral residual vs FP32 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, R in r["recipes"].items():
            for s in show:
                a = R["per_step"][s]
                print(f"| {name} | {s} | {a['truth']['hydrostatic_rmse']:.1f} / {a['fp32']['hydrostatic_rmse']:.1f} / {a['hydrostatic_rmse']:.1f} | "
                      f"{a['vs_truth']['hydrostatic_excess']:+.1f} | {a['truth']['geostrophic_rmse']:.2f} / {a['fp32']['geostrophic_rmse']:.2f} / {a['geostrophic_rmse']:.2f} | "
                      f"{a['vs_truth']['geostrophic_excess']:+.2f} | {a['fp32_vs_truth']['L_eff_km']:.0f} / {a['vs_truth']['L_eff_km']:.0f} / {a['vs_fp32']['L_eff_km']:.0f} | "
                      f"{a['vs_fp32']['small_scale_ratio']:.3f} | {a['vs_fp32']['spectral_residual']:.3f} |")
        # FP32 vs ERA5 reference line
        name0 = next(iter(r["recipes"]))
        for s in show:
            a = r["recipes"][name0]["per_step"][s]
            f = a["fp32_vs_truth"]
            print(f"| FP32 vs ERA5 | {s} | — | {f['hydrostatic_excess']:+.1f} | — | {f['geostrophic_excess']:+.2f} | {f['L_eff_km']:.0f} | {f['small_scale_ratio']:.3f} | {f['spectral_residual']:.3f} |")


if __name__ == "__main__":
    main()
