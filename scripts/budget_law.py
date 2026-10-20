"""Test the 'output error budget' explanation: Aurora decodes the absolute normalised state, so quantization noise
in the output path is ~ k * sigma_v (the normalisation scale), while qerr divides by the 6-h FP32 skill RMSE_v.
Prediction: per-variable qerr_v ~ k * A_v with A_v = sigma_v / RMSE_v(FP32 vs ERA5).

  python scripts/budget_law.py m01_q0_decoder_only q0_sim m02_q0_no_decoder g02_svdq_a08_r32 t1_g02_dec_bf16
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from aurora.normalisation import locations, scales      # noqa: E402

VARS = {"2t": "2t", "10u": "10u", "10v": "10v", "msl": "msl",
        "z500": "z_500", "t850": "t_850", "u850": "u_850", "v850": "v_850", "q700": "q_700"}


def load(n):
    return json.loads((ROOT / "results" / f"{n}__dev_h2" / "summary.json").read_text())


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy)


def main():
    f = load("fp32")
    A = {v: scales[k] / f["per_var"][f"{v}_s1"]["rmse_truth"] for v, k in VARS.items()}
    print("sigma_v / RMSE_v(FP32, 6 h):", {v: round(a, 1) for v, a in A.items()})
    for n in sys.argv[1:]:
        s = load(n)
        for group, vs in (("upper", ["z500", "t850", "u850", "v850", "q700"]), ("surface", ["2t", "10u", "10v", "msl"]),
                          ("all", list(VARS))):
            for step in (1, 2):
                q = [s["per_var"][f"{v}_s{step}"]["rmse_ref"] / f["per_var"][f"{v}_s{step}"]["rmse_truth"] for v in vs]
                a = [A[v] for v in vs]
                if max(q) < 0.05:                     # untouched heads (e.g. surface at step 1 in m01)
                    continue
                r = pearson([math.log(x) for x in a], [math.log(max(y, 1e-9)) for y in q])
                k = [y / x for x, y in zip(a, q)]
                print(f"{n:24s} {group:7s} s{step}: log-log r = {r:+.2f}; qerr/A per var "
                      + " ".join(f"{v}:{100 * kk:.2f}" for v, kk in zip(vs, k)))


if __name__ == "__main__":
    main()
