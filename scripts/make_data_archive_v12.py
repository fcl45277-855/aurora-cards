"""Pack the per-record result files behind the revised ESI manuscript into archive v1.2 (adds the W4A16 ablation and the
analyses added in revision to the layout of v1.1).

  python scripts/make_data_archive_v12.py [--root <dir with results/ ...>] [--out <dir>]
Writes aurora-cards-results-v1.2.zip and its SHA-256 next to it. Same exclusions and path reduction as v1.1
(scripts/make_data_archive_server.py); additionally packs results_timing_w16/ (native timing without the low-rank branch).
"""
import argparse
import hashlib
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("results", "results_finetuned", "results_small", "results_v16", "results_timing_w16")
SKIP_PARTS = ("_ref_fp32_", "_build_cache", "_artefacts", "__pycache__", "_physdamage_", "_pilot_")
SKIP_SUFFIX = (".pt", ".log", ".tmp", ".part")
README = """aurora-cards result archive v1.2
================================

Per-record results of "NVFP4 quantisation of the Aurora weather foundation model: Perceiver-resampler sensitivity,
physical consistency and native-kernel cost" (Y. Guan, J. Li, L. Lu, G. Pan, J. Song, Q. Hou; Earth Science
Informatics, submitted). Code: https://github.com/fcl45277-855/aurora-cards (docs/REPRODUCE.md maps every table and
figure to the command and the files below).

v1.2 contains everything in v1.1 plus the results added for the revised manuscript:
  results/w16_*                      prespecified W4A16 ablation (main text Tables 5 and 7; Online Resource 1, S15)
  results/w16_t7_inh_*               group attribution of the inherited BF16 layers (Table 5; S17b)
  results/*__test_h20, *__oct_h20    20-step rollouts on 25 March and 9 October dates (Table 9; S17c)
  results/_acc_test_h20, _acc_oct_h20  anomaly correlation (Table 9; S17c)
  results/_w16_*, _r1_*, *_h2w16*, *_h20w16*  paired summaries and physical diagnostics of the ablation
  results/native_t7_w16_*            native accuracy without the low-rank branch (Sect. 4.5)
  results_timing_w16/                native timing without the low-rank branch (S17a)

Layout (set AQ_RESULTS to the unpacked results/ folder):
  results/            Aurora 0.25 pretrained (main results)
  results_finetuned/  Aurora 0.25 finetuned (cross-model replication)
  results_small/      Aurora-small (cross-model replication)
  results_v16/        Beryl case and PhysMetrics runs that include t7_a16
  results_timing_w16/ native timing of the W4A16 builds with and without the low-rank branch

Per card <recipe>__<split>_h<steps>/: recipe.json, build_info.json, records.jsonl (one row per initial date, step and
variable: RMSE against ERA5 and against the FP32 forecast, latitude-weighted on the 720 x 1440 grid), summary.json
(intervals in summary.json are i.i.d. date-bootstrap intervals; the paper's 3-day block intervals come from
scripts/block_ci.py, scripts/extended_analyses.py and scripts/r1_summary.py run on these records).
Aggregates start with an underscore. ERA5 and the ERA5 climatology are not included (download scripts
scripts/download_era5_day.py and scripts/clim_download.py in the code repository). Licence: CC-BY 4.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="directory that contains results/, results_finetuned/, ...")
    ap.add_argument("--out", default=str(ROOT / "release_archive"))
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "aurora-cards-results-v1.2.zip"
    n = size = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.txt", README)
        for d in DIRS:
            base = root / d
            if not base.exists():
                continue
            for f in sorted(base.rglob("*")):
                rel = f.relative_to(root).as_posix()
                if not f.is_file() or any(p in rel for p in SKIP_PARTS) or f.suffix in SKIP_SUFFIX:
                    continue
                if f.suffix == ".json":
                    txt = f.read_text(encoding="utf-8", errors="replace")
                    txt = re.sub(r'"(/root/[^"]*/)([^"/]+)"', r'"\2"', txt)
                    z.writestr(rel, txt)
                else:
                    z.write(f, rel)
                n += 1
                size += f.stat().st_size
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (out_dir / (out.name + ".sha256")).write_text(f"{h}  {out.name}\n")
    print(f"wrote {out}: {n} files, {size / 2**20:.0f} MiB uncompressed, {out.stat().st_size / 2**20:.0f} MiB zipped, sha256 {h}")


if __name__ == "__main__":
    main()
