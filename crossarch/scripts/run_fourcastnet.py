"""Original 20-channel AFNO control. Weight-only NVFP4, FP32 activations.

Spectral matrices, MLPs, patch embedding and output head are all inventoried.
No switching statistic is defined for this architecture.
"""
import argparse, json, sys, struct, zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model_support import ROOT,sha,prepare_compat,build_fcn
from attention_adapter import quantize_afno_matrix

EXPECTED='8a86a8c487a42283f97b9f119abcba11bc66b0103fc97854d881a0d6ace52827'
class ScalarFloat(float):
    """Inert allowlisted replacement for old YAML numeric checkpoint metadata."""

def load_fcn(path):
    if path.suffix=='.npz':
        manifest=json.loads((ROOT/'metadata/fourcastnet_inference_manifest.json').read_text())
        if sha(path)!=manifest['inference_npz_sha256'] or manifest['source_sha256']!=EXPECTED:
            raise ValueError('Inference conversion hash/provenance mismatch')
        # Canonical NPZ members are uncompressed. Map tensor payloads without
        # retaining an additional 300 MB heap copy in CPU mode's 2 GiB quota.
        specifications={v['name']:v for v in manifest['tensors']}
        state={}
        with zipfile.ZipFile(path) as archive, path.open('rb') as stream:
            for entry in archive.infolist():
                assert entry.compress_type==zipfile.ZIP_STORED and entry.filename.endswith('.npy')
                name=entry.filename.removesuffix('.npy')
                stream.seek(entry.header_offset)
                header=stream.read(30)
                filename_length,extra_length=struct.unpack_from('<HH',header,26)
                stream.seek(filename_length+extra_length,1)
                assert np.lib.format.read_magic(stream)==(1,0)
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                assert not fortran and dtype==np.dtype('<f4') and list(shape)==specifications[name]['shape']
                mapped=np.memmap(path,mode='c',dtype=dtype,shape=shape,offset=stream.tell(),order='C')
                state[name]=torch.from_numpy(mapped)
        with torch.device('meta'):
            model=build_fcn()
        model.load_state_dict(state,strict=True,assign=True)
        return model
    if sha(path)!=EXPECTED:
        raise ValueError('Original NERSC checkpoint does not match recorded local HTTPS hash')
    with torch.serialization.safe_globals([(ScalarFloat,'ruamel.yaml.scalarfloat.ScalarFloat')]):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    state={k.removeprefix('module.'):v for k,v in checkpoint['model_state'].items()}
    with torch.device('meta'):
        model=build_fcn()
    model.load_state_dict(state,strict=True,assign=True)
    return model

def asset_path():
    inference=ROOT/'weights/fourcastnet_inference.npz'
    return inference if inference.exists() else ROOT/'weights/fourcastnet_backbone.ckpt'

def groups(model):
    result=[('patch_embed',[('patch_embed.proj.weight',model.patch_embed.proj.weight,'conv')])]
    for i,block in enumerate(model.blocks):
        result.append((f'blocks.{i}.filter',[(f'blocks.{i}.filter.{k}',getattr(block.filter,k),'afno') for k in ['w1','w2']]))
        result.append((f'blocks.{i}.mlp',[(f'blocks.{i}.mlp.{k}.weight',getattr(block.mlp,k).weight,'linear') for k in ['fc1','fc2']]))
    result.append(('head',[('head.weight',model.head.weight,'linear')]))
    return result

@contextmanager
def quantized(parameters,qdq):
    saved=[]
    try:
        with torch.no_grad():
            for name,parameter,kind in parameters:
                saved.append((parameter,parameter.detach().clone()))
                if kind=='afno':
                    value=quantize_afno_matrix(parameter,qdq)
                elif kind=='conv':
                    value=qdq(parameter.flatten(1)).reshape_as(parameter)
                else:
                    value=qdq(parameter)
                parameter.copy_(value)
        yield
    finally:
        with torch.no_grad():
            for parameter,original in saved:
                parameter.copy_(original)

def error(x,y,lat):
    weights=torch.cos(torch.deg2rad(lat)).clamp(min=0)
    weights=weights/weights.mean()
    return float(((x-y).double().square()*weights[None,None,:,None]).mean())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('inputs',nargs='*',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    checkpoint=asset_path()
    problems=[]
    expected_asset=(json.loads((ROOT/'metadata/fourcastnet_inference_manifest.json').read_text())['inference_npz_sha256'] if checkpoint.suffix=='.npz' else EXPECTED)
    if not checkpoint.is_file() or sha(checkpoint)!=expected_asset:
        problems.append('Complete, hash-verified NERSC checkpoint missing')
    if not args.inputs:
        problems.append('Prepared real inputs must be supplied')
    validation_path=ROOT/'metadata/input_validation_cpu.json'
    validated={}
    if not validation_path.exists():
        problems.append('Independent CPU input validation report is missing')
    else:
        validated={v['date']:v['sha256'] for v in json.loads(validation_path.read_text())['inputs']}
    for path in args.inputs:
        sidecar=path.with_suffix('.json')
        if not path.is_file() or not sidecar.is_file():
            problems.append('Input/manifest missing: '+str(path)); continue
        record=json.loads(sidecar.read_text())
        if sha(path)!=record['sha256'] or not record['normalized'] or record['lead_hours']!=6 or record.get('preparation_version')!=2 or validated.get(path.stem)!=record['sha256']:
            problems.append('Input provenance/normalization failed: '+str(path))
    if args.check:
        print(json.dumps({'ready_assets':not problems,'problems':problems,'gpu_run_started':False},indent=2)); return 2 if problems else 0
    if problems:
        raise RuntimeError('; '.join(problems))
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not enabled; no scientific evaluation started')
    if args.output is None or args.output.exists():
        raise ValueError('Specify a new output file; existing results are retained')
    prepare_compat()
    import crossarch_paths  # noqa: F401  (puts the repository root, with cards/, on sys.path)
    from cards.nvfp4 import qdq_blocks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_fcn(checkpoint).cuda().eval()
    inventory=groups(model)
    result={'mode':'W4A32 RTN, FP32 reference/activations/accumulation','err_in_flip':None,
            'original_download_sha256':EXPECTED,'model_asset_sha256':sha(checkpoint),
            'groups':{name:[{'parameter':n,'shape':list(p.shape),'input_axis':-2 if kind=='afno' else 1 if kind=='conv' else -1} for n,p,kind in parameters] for name,parameters in inventory},'dates':[]}
    with torch.inference_mode():
        for path in args.inputs:
            with np.load(path,allow_pickle=False) as data:
                if list(data['variables'])!=['u10','v10','t2m','sp','msl','t850','u1000','v1000','z1000','u850','v850','z850','u500','v500','z500','t500','z50','r500','r850','tcwv']:
                    raise ValueError('Unexpected AFNO channel order')
                x=torch.tensor(data['x'][None],device='cuda'); y=torch.tensor(data['y'][None],device='cuda')
                lat=torch.tensor(data['lat'],device='cuda')
                reference=model(x)
                record={'input_sha256':sha(path),'init_time':str(data['init_time'].item()),
                        'reference_mse_vs_truth':error(reference,y,lat),'persistence_mse_vs_truth':error(x,y,lat),'damage':{}}
                for name,parameters in inventory:
                    with quantized(parameters,qdq_blocks):
                        predicted=model(x)
                    record['damage'][name]={'mse_vs_reference':error(predicted,reference,lat),
                                            'mse_vs_truth':error(predicted,y,lat),
                                            'quantized_parameters':sum(p.numel() for _,p,_ in parameters)}
                result['dates'].append(record)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
