"""Fetch only source ranges containing inference tensors, then verify conversion.

Each raw tensor is checked against the complete original checkpoint downloaded
and SHA256-verified on Windows. The final canonical NPZ must also match exactly.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,io,json,urllib.request,zipfile
import numpy as np

from crossarch_paths import ROOT
manifest=json.loads((ROOT/'metadata/fourcastnet_inference_manifest.json').read_text())
block=1024**2; size=manifest['source_bytes']
parts=ROOT/'weights/fourcastnet_parts'; parts.mkdir(exist_ok=True)
needed=set()
for tensor in manifest['tensors']:
    first=tensor['source_offset']//block*block
    last=(tensor['source_offset']+tensor['bytes']-1)//block*block
    needed.update(range(first,last+1,block))
missing=[start for start in sorted(needed) if not (parts/f'{start:012d}.part').exists()]
print('Inference source ranges needed:',len(needed),'missing:',len(missing),flush=True)
def fetch(start):
    end=min(start+block,size)-1
    path=parts/f'{start:012d}.part'
    if path.exists() and path.stat().st_size==end-start+1:
        return
    for attempt in range(3):
        try:
            suffix=f'?download=true&aq_range={start}'+(f'&retry={attempt}' if attempt else '')
            req=urllib.request.Request(manifest['source_url']+suffix,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=40) as response:
                assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{size}'
                data=response.read(block+1)
            assert len(data)==end-start+1
            path.write_bytes(data)
            return
        except Exception:
            if attempt==2:
                raise
with ThreadPoolExecutor(max_workers=16) as pool:
    tasks=[pool.submit(fetch,start) for start in sorted(needed)]
    for count,task in enumerate(as_completed(tasks),1):
        task.result()
        if count%32==0:
            print('Inference source ranges checked:',count,'/',len(needed),flush=True)
def read_range(offset,length):
    result=bytearray()
    while length:
        base=offset//block*block; within=offset-base
        data=(parts/f'{base:012d}.part').read_bytes()
        take=min(length,len(data)-within)
        if take<=0:
            raise ValueError('Missing source bytes')
        result.extend(data[within:within+take]); offset+=take; length-=take
    return result
partial=ROOT/'weights/fourcastnet_inference.partial.npz'
with zipfile.ZipFile(partial,'w') as output:
    for tensor in manifest['tensors']:
        data=read_range(tensor['source_offset'],tensor['bytes'])
        assert hashlib.sha256(data).hexdigest()==tensor['sha256'],tensor['name']
        array=np.frombuffer(data,dtype='<f4').reshape(tensor['shape'])
        buffer=io.BytesIO(); np.lib.format.write_array(buffer,array,version=(1,0),allow_pickle=False)
        info=zipfile.ZipInfo(tensor['name']+'.npy',date_time=(1980,1,1,0,0,0)); info.create_system=0
        output.writestr(info,buffer.getvalue())
with partial.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
assert digest==manifest['inference_npz_sha256'],(digest,manifest['inference_npz_sha256'])
partial.replace(ROOT/'weights/fourcastnet_inference.npz')
(ROOT/'metadata/fourcastnet_inference_verified.json').write_text(json.dumps({'inference_sha256':digest,'tensor_count':len(manifest['tensors']),
    'source_sha256':manifest['source_sha256'],'source_hash_verified_on':'Windows host complete HTTPS download',
    'per_tensor_bytes_verified_on':'server','optimizer_state_omitted':True,'gpu_used':False},indent=2))
print('Complete inference model verified, optimizer state omitted:',digest,flush=True)
