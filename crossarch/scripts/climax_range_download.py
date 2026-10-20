"""Resume bounded HTTP ranges; assemble only after verifying every response."""
from pathlib import Path
import concurrent.futures as futures
import hashlib, json, time, urllib.request
from crossarch_paths import ROOT
SIZE=432419871
CHUNK=1024**2
EXPECTED='6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43'
URL='https://huggingface.co/microsoft/ClimaX/resolve/1f77f8ee97328afd3b8d37f72a17dd31daf0eb90/5.625deg.ckpt'
parts=ROOT/'weights/climax_parts'
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
with futures.ThreadPoolExecutor(max_workers=12) as pool:
    tasks=[pool.submit(get_part,s) for s in range(0,SIZE,CHUNK)]
    for done,t in enumerate(futures.as_completed(tasks),1):
        t.result()
        if done%16==0:
            print(f'ClimaX verified ranges: {done}/{len(tasks)}',flush=True)
target=ROOT/'weights/climax_5.625deg.assembled'
with target.open('wb') as stream:
    for start in range(0,SIZE,CHUNK):
        stream.write((parts/f'{start:012d}.part').read_bytes())
with target.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
if digest!=EXPECTED:
    raise ValueError('Assembled SHA256 mismatch; no model installed')
target.replace(ROOT/'weights/climax_5.625deg.ckpt')
(ROOT/'metadata/climax_download_verified.json').write_text(json.dumps({'bytes':SIZE,'sha256':digest,'url':URL,'completed_unix':time.time()},indent=2))
print('ClimaX complete; official SHA256 verified',flush=True)
