"""Resume bounded HTTP ranges; assemble only after verifying every response."""
from pathlib import Path
import concurrent.futures as futures
import hashlib, json, time, urllib.request
from crossarch_paths import ROOT
SIZE=896430597
CHUNK=1024**2
EXPECTED='8a86a8c487a42283f97b9f119abcba11bc66b0103fc97854d881a0d6ace52827'
URL='https://portal.nersc.gov/project/m4134/FCN_weights_v0/backbone.ckpt'
parts=ROOT/'weights/fourcastnet_parts'
parts.mkdir(exist_ok=True)
def get_part(start):
    end=min(start+CHUNK,SIZE)-1
    path=parts/f'{start:012d}.part'
    if path.exists() and path.stat().st_size==end-start+1:
        return path
    for attempt in range(3):
        try:
            req=urllib.request.Request(URL+f'?download=true&aq_range={start}',headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=40) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('Unexpected range response')
                data=r.read(CHUNK+1)
            if len(data)!=end-start+1:
                raise ValueError(f'Truncated range {start}: {len(data)}')
            path.write_bytes(data)
            return path
        except Exception:
            if attempt==2:
                raise
with futures.ThreadPoolExecutor(max_workers=24) as pool:
    tasks=[pool.submit(get_part,s) for s in range(0,SIZE,CHUNK)]
    for done,t in enumerate(futures.as_completed(tasks),1):
        t.result()
        if done%16==0:
            print(f'FourCastNet verified ranges: {done}/{len(tasks)}',flush=True)
target=ROOT/'weights/fourcastnet_backbone.assembled'
with target.open('wb') as stream:
    for start in range(0,SIZE,CHUNK):
        stream.write((parts/f'{start:012d}.part').read_bytes())
with target.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
if digest!=EXPECTED:
    raise ValueError('Assembled SHA256 mismatch; no model installed')
target.replace(ROOT/'weights/fourcastnet_backbone.ckpt')
(ROOT/'metadata/fourcastnet_download_verified.json').write_text(json.dumps({'bytes':SIZE,'sha256':digest,'url':URL,'completed_unix':time.time()},indent=2))
print('FourCastNet complete; transport SHA256 matches local HTTPS download',flush=True)
