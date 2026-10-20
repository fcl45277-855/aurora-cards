"""Candidate 19 (reviewer point 2, 2026-10-20): does the replay predictor err_in_flip give a *usable* allocation rule?

Greedy protection: start from ALL314 (all 314 linear layers W4A4, q0_sim settings; Q0 lacks the encoder resampler, 2026-10-20) and keep the attention linear layers of the
top-k modules of a ranking in high precision (k = 1..4). Rankings over the eight attention modules of the pretrained model
(values from the replay analysis, dev split, replay alpha = 1, day 1):
  flip    err_in_flip descending (the predictor)                    perc_dec perc_enc en1 en0 de2 de1 de0 en2
  yerr    local activation error descending (the usual proxy)        en1 en0 de2 de0 de1 perc_enc en2 perc_dec
  params  parameter count ascending ("small modules first")          perc_enc perc_dec en0 de2 en1 de1 de0 en2
  rand1/2 random permutations (numpy seeds 1 and 2)
Pre-registered criterion (written before any card ran): at every k, qerr(flip) <= qerr(params) and
qerr(flip) < qerr(yerr) and qerr(flip) < qerr(rand); and the area under the qerr-vs-protected-bytes curve is smallest for
flip. Identical protected sets (e.g. flip k=2 == params k=2) share one card.

  python scripts/alloc_greedy.py            # writes recipes/alloc_*.json and prints queue lines (dev split, 2 steps)
  python scripts/alloc_greedy.py --table    # after the cards ran: Pareto table from results/alloc_*__dev_h2/summary.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATTERN = {"perc_enc": "encoder.level_agg.&.0.0.to_", "perc_dec": "decoder.level_decoder.&.0.0.to_",   # same unit as b_attn_*
           **{f"en{i}": f"backbone.encoder_layers.{i}.&.attn." for i in range(3)},
           **{f"de{i}": f"backbone.decoder_layers.{i}.&.attn." for i in range(3)}}
PARAMS_M = {"perc_enc": 1.0, "perc_dec": 4.2, "en1": 41.9, "en0": 6.3, "de1": 41.9, "de2": 6.3, "de0": 134.2, "en2": 134.2}
FLIP = {"perc_enc": 0.942, "perc_dec": 0.956, "en1": 0.844, "en0": 0.757, "de1": 0.221, "de2": 0.733, "de0": 0.178, "en2": 0.125}
YERR = {"perc_enc": 0.042, "perc_dec": 0.026, "en1": 0.148, "en0": 0.079, "de1": 0.045, "de2": 0.067, "de0": 0.047, "en2": 0.042}
MODS = ["perc_enc", "perc_dec", "en0", "en1", "en2", "de0", "de1", "de2"]
KMAX = 4


def rankings() -> dict:
    r = {"flip": sorted(MODS, key=lambda m: -FLIP[m]),
         "yerr": sorted(MODS, key=lambda m: (-YERR[m], m)),
         "params": sorted(MODS, key=lambda m: (PARAMS_M[m], m))}
    for seed in (1, 2):
        r[f"rand{seed}"] = [str(m) for m in np.random.RandomState(seed).permutation(MODS)]
    return r


def card_name(protected: tuple) -> str:
    return "alloc_" + "+".join(sorted(protected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    base = json.loads((ROOT / "recipes" / "q0_sim.json").read_text(encoding="utf-8"))
    plan = {"alloc_none": tuple()}   # card name -> protected set; alloc_none = k=0 baseline
    for rk, order in rankings().items():
        for k in range(1, KMAX + 1):
            prot = tuple(order[:k])
            plan.setdefault(card_name(prot), prot)
    if a.table:
        res = Path(a.results)
        print("| ranking | k | protected (attention layers of) | protected params (M) | qerr % | ERA5 excess % |")
        print("| --- | --- | --- | --- | --- | --- |")
        for rk, order in rankings().items():
            for k in range(1, KMAX + 1):
                prot = tuple(order[:k]); p = res / f"{card_name(prot)}__dev_h2" / "summary.json"
                if not p.exists():
                    print(f"| {rk} | {k} | {' '.join(prot)} | {sum(PARAMS_M[m] for m in prot):.1f} | (not run) | |"); continue
                s = json.loads(p.read_text(encoding="utf-8"))
                print(f"| {rk} | {k} | {' '.join(prot)} | {sum(PARAMS_M[m] for m in prot):.1f} | {s['qerr_pct']:.2f} | {s.get('excess_pct', float('nan')):+.2f} |")
        return
    lines = []
    for name, prot in plan.items():
        rec = dict(base); rec["name"] = name
        rec["layers"] = "all314" + (":exc=" + ",".join(PATTERN[m] for m in prot) if prot else "")
        rec["_protected"] = list(prot); rec["_protected_params_M"] = sum(PARAMS_M[m] for m in prot)
        (ROOT / "recipes" / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        lines.append(f"!AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 AQ_MEM_FRAC=0.9 python -u scripts/run_card.py --split dev --steps 2 --recipe recipes/{name}.json")
    print(f"# {len(plan)} unique cards from {len(rankings()) * KMAX} ranking/k pairs", file=sys.stderr)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
