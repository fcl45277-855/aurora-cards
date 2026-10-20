"""Fetch WeatherBench2's regridding.py (and its licence) at the pinned commit for prepare_holdout.py.

prepare_holdout.py executes only the pure conservative-weight functions of this file (NumPy backend) and checks its
SHA256 against the manifest written here. The expected hashes are the ones used for the paper.

    python fetch_wb2_regridding.py
"""
import hashlib
import json
import urllib.request

from crossarch_paths import ROOT

REPO = 'google-research/weatherbench2'
COMMIT = '9c976eb40e203332f17332d2374de21ffb27fa00'
EXPECTED = {'weatherbench2/regridding.py': 'd0dcbf743d651cd30afab86b7f3a4bcf4c921e0f440bb534d36b037267083e83',
            'LICENSE': '58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd'}


def main():
    out = ROOT / 'finetune_era5_20261020' / 'sources'
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, digest in EXPECTED.items():
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            raise SystemExit(f'{name}: SHA256 {got} differs from the pinned {digest}')
        (out / name.split('/')[-1]).write_bytes(data)
        files[name] = {'url': url, 'sha256': got}
    (out / 'source_manifest.json').write_text(json.dumps({'repository': REPO, 'commit': COMMIT, 'files': files},
                                                         indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
