"""Fetch public upstream sources and one checkpoint. No GPU jobs or package installs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import urllib.request
import traceback

from crossarch_paths import ROOT
for sub in ('sources', 'weights', 'metadata', 'logs', 'scripts', 'configs', 'results'):
    (ROOT/sub).mkdir(parents=True, exist_ok=True)

def log(message):
    print(dt.datetime.now(dt.timezone.utc).isoformat(), message, flush=True)

def get(url, maximum=4*1024**2):
    request = urllib.request.Request(url, headers={'User-Agent':'Aurora-crossarch-feasibility/1.0'})
    with urllib.request.urlopen(request, timeout=35) as response:
        data = response.read(maximum+1)
        if len(data) > maximum:
            raise RuntimeError('bounded metadata response exceeded limit')
        return data

report = {'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'gpu_experiments_started':False, 'sources':{}, 'errors':[]}
for repo, files in {
    'microsoft/ClimaX': ['README.md','LICENSE','docs/usage.md','src/climax/arch.py','src/climax/parallelpatchembed.py','src/climax/utils/pos_embed.py','src/climax/utils/data_utils.py','src/climax/global_forecast/module.py','configs/pretrain_climax.yaml','configs/global_forecast_climax.yaml'],
    'NVlabs/FourCastNet':['README.md','LICENSE','networks/afnonet.py','config/AFNO.yaml','data_process/parallel_copy_small_set.py'],
}.items():
    try:
        info = json.loads(get('https://api.github.com/repos/'+repo+'/commits/HEAD'))
        revision = info['sha']
        record = {'revision':revision, 'files':{}}
        for name in files:
            try:
                url = 'https://raw.githubusercontent.com/'+repo+'/'+revision+'/'+name
                data = get(url)
                dest = ROOT/'sources'/repo.split('/')[-1]/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                record['files'][name] = {'url':url,'sha256':hashlib.sha256(data).hexdigest()}
            except Exception as e:
                record['files'][name] = {'error':str(e)}
        report['sources'][repo] = record
        log('source captured '+repo+' '+revision)
    except Exception as e:
        report['errors'].append({'stage':repo,'error':str(e)})
        log('source failed '+repo+' '+str(e))

try:
    url = 'https://portal.nersc.gov/project/m4134/FCN_weights_v0/'
    data = get(url)
    (ROOT/'metadata/fourcastnet_download_index.html').write_bytes(data)
    report['fourcastnet_index'] = {'url':url,'links':re.findall(r'href=[\"\x27]([^\"\x27]+)', data.decode(errors='replace'))}
    log('FourCastNet public download index recorded')
except Exception as e:
    report['errors'].append({'stage':'fourcastnet_index','error':str(e)})

try:
    info = json.loads(get('https://huggingface.co/api/models/microsoft/ClimaX'))
    revision = info['sha']
    report['climax_model_repository'] = {'revision':revision,'files':[v['rfilename'] for v in info.get('siblings',[])]}
    url = 'https://huggingface.co/microsoft/ClimaX/resolve/'+revision+'/5.625deg.ckpt'
    target = ROOT/'weights/climax_5.625deg.ckpt'
    expected = '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43'
    if not target.exists():
        log('downloading official ClimaX checkpoint (about 432 MB)')
        request = urllib.request.Request(url, headers={'User-Agent':'Aurora-crossarch-feasibility/1.0'})
        partial = target.with_suffix('.partial')
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
            total, next_log = 0, 32*1024**2
            while True:
                block = response.read(1024**2)
                if not block:
                    break
                total += len(block)
                if total > 600*1024**2:
                    raise RuntimeError('checkpoint exceeded 600 MiB download cap')
                output.write(block)
                if total >= next_log:
                    log('checkpoint downloaded MiB='+str(total//1024**2))
                    next_log += 32*1024**2
        partial.replace(target)
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    if digest != expected:
        raise RuntimeError('checkpoint SHA256 mismatch: '+digest)
    report['climax_checkpoint'] = {'url':url,'sha256':digest,'bytes':target.stat().st_size}
    import torch
    torch.set_num_threads(2)
    checkpoint = torch.load(target, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict',checkpoint)
    report['climax_checkpoint']['top_level_keys'] = list(checkpoint.keys())[:30]
    report['climax_checkpoint']['tensor_count'] = len(state)
    report['climax_checkpoint']['architecture_shapes'] = {k:list(v.shape) for k,v in state.items() if hasattr(v,'shape') and any(s in k for s in ['var_query','var_agg','channel_query','channel_agg','pos_embed','head.','token_embeds.0'])}
    report['climax_checkpoint']['normalization_keys'] = [k for k in state if any(s in k.lower() for s in ['normalize','global_mean','global_std'])]
    report['climax_checkpoint']['configuration'] = {k:str(v)[:500] for k,v in checkpoint.items() if k != 'state_dict' and not hasattr(v,'shape')}
    log('checkpoint hash and CPU structure verified')
except Exception as e:
    report['errors'].append({'stage':'climax_checkpoint','error':str(e),'traceback':traceback.format_exc()})
    log('checkpoint stage failed '+str(e))

report['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
(ROOT/'metadata/bootstrap_cpu.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
log('CPU preparation finished; see metadata/bootstrap_cpu.json')
