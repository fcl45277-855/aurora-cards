"""CPU-only coordinator: two CDS downloads, then serial bounded-memory assembly."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess, sys

from crossarch_paths import ROOT
scripts=Path(__file__).resolve().parent
months={'2024-01':['2024-01-04','2024-01-06','2024-01-08'],
        '2024-03':[f'2024-03-{d:02d}' for d in range(1,13)],
        '2024-07':[f'2024-07-{d:02d}' for d in range(1,9)]}
def fetch(month):
    subprocess.run([sys.executable,'-u',str(scripts/'prepare_era5_extras.py'),'--month',month,'--fcn-only'],check=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(fetch,months))
for month,days in months.items():
    remaining=[d for d in days if not (ROOT/f'data/fourcastnet/{d}.npz').exists()]
    if remaining:
        subprocess.run([sys.executable,'-u',str(scripts/'prepare_fcn_inputs.py'),*remaining],check=True)
print('All prescribed FourCastNet predictor/confirmation/season inputs prepared',flush=True)
