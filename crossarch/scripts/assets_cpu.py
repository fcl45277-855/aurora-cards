"""Verified CPU-only asset preparation; run inside CROSSARCH_ROOT."""
from pathlib import Path
import hashlib, json, urllib.request, datetime
import torch

from crossarch_paths import ROOT
torch.set_num_threads(2)
assets = [
 ('weights/climax_5.625deg.ckpt', 'https://huggingface.co/microsoft/ClimaX/resolve/1f77f8ee97328afd3b8d37f72a17dd31daf0eb90/5.625deg.ckpt', '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43', 500*1024**2),
 ('weights/fourcastnet_backbone.ckpt', 'https://portal.nersc.gov/project/m4134/FCN_weights_v0/backbone.ckpt', '8a86a8c487a42283f97b9f119abcba11bc66b0103fc97854d881a0d6ace52827', 1000*1024**2),
]
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gpu_used': False, 'assets': {}}
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
for name, url, expected, cap in assets:
    target = ROOT/name
    entry = {'url': url, 'expected_sha256': expected}
    report['assets'][name] = entry
    try:
        if not target.exists() or sha(target) != expected:
            partial = target.with_suffix('.download-partial')
            print('Downloading', name, flush=True)
            with urllib.request.urlopen(url, timeout=45) as response, partial.open('wb') as output:
                total = 0
                while chunk := response.read(4*1024**2):
                    total += len(chunk)
                    if total > cap:
                        raise RuntimeError('Download size cap exceeded')
                    output.write(chunk)
                    if total % (64*1024**2) == 0:
                        print(name, total//1024**2, 'MiB', flush=True)
            if sha(partial) != expected:
                raise RuntimeError('SHA256 mismatch; original target retained')
            partial.replace(target)
        checkpoint = torch.load(target, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint.get('model_state', checkpoint))
        entry.update(sha256=sha(target), bytes=target.stat().st_size,
                     top_keys=list(checkpoint)[:30], tensor_count=len(state),
                     shapes={k:list(v.shape) for k,v in state.items() if hasattr(v,'shape')},
                     non_tensor_values={k:str(v)[:2000] for k,v in checkpoint.items() if k not in ('state_dict','model_state') and not hasattr(v,'shape')})
        print('Verified', name, entry['tensor_count'], 'tensors', flush=True)
    except Exception as error:
        entry['error'] = str(error)
        print('Asset failure', name, str(error), flush=True)
    (ROOT/'metadata/assets_cpu.json').write_text(json.dumps(report, indent=2))
