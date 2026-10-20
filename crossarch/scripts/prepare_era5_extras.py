"""Fetch missing observed fields; no surrogate humidity or zero-filled inputs.

Writes only to this experiment's data/era5_extras directory. Existing Aurora
arrays are read-only. CDS credentials are consumed by cdsapi, never printed.
"""
from pathlib import Path
import argparse, hashlib, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr
from crossarch_paths import ROOT

def ranged_download(url,size,target):
    """Public CDS asset URL stays in memory; credentials are never serialized."""
    if size>2*1024**3:
        raise ValueError('Request exceeds the per-file 2 GiB preparation cap')
    block=1024**2
    parts=target.parent/(target.name+'.ranges'); parts.mkdir(exist_ok=True)
    def fetch(start):
        end=min(start+block,size)-1
        part=parts/str(start)
        if part.exists() and part.stat().st_size==end-start+1:
            return
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request,timeout=40) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise ValueError('Unexpected CDS range response')
                    data=response.read(block+1)
                if len(data)!=end-start+1:
                    raise ValueError('Truncated CDS range')
                part.write_bytes(data)
                return
            except Exception:
                if attempt==2:
                    raise
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch,range(0,size,block)))
    with target.open('wb') as stream:
        for start in range(0,size,block):
            stream.write((parts/str(start)).read_bytes())
    assert target.stat().st_size==size

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--month',required=True,choices=['2024-01','2024-02','2024-03','2024-07'])
    parser.add_argument('--fcn-only',action='store_true')
    args=parser.parse_args()
    days={'2024-01':['04','06','08'],'2024-02':['02'],
          '2024-03':[f'{i:02d}' for i in range(1,13)],'2024-07':[f'{i:02d}' for i in range(1,9)]}[args.month]
    out=ROOT/'data/era5_extras'; out.mkdir(parents=True,exist_ok=True)
    import cdsapi
    client=cdsapi.Client(quiet=True,progress=False,timeout=30,retry_max=1)
    common={'product_type':['reanalysis'],'year':[args.month[:4]],'month':[args.month[5:]],
            'day':days,'time':['00:00','06:00'],'data_format':'netcdf','download_format':'unarchived'}
    for suffix,dataset,extra in [
        ('surface','reanalysis-era5-single-levels',{'variable':['surface_pressure','total_column_water_vapour']}),
        ('humidity','reanalysis-era5-pressure-levels',{'variable':['relative_humidity'],'pressure_level':['500','850'] if args.fcn_only else ['50','250','500','600','700','850','925']})]:
        path=out/f'{args.month}_{suffix}.nc'
        if not path.exists():
            partial=path.with_suffix('.download.nc')
            print('Requesting',args.month,suffix,flush=True)
            result=client.retrieve(dataset,{**common,**extra})
            ranged_download(result.location,result.content_length,partial)
            with xr.open_dataset(partial) as ds:
                time='valid_time' if 'valid_time' in ds.coords else 'time'
                assert len(ds[time])==len(days)*2
                assert ds.sizes['latitude']==721 and ds.sizes['longitude']==1440
                assert float(ds.latitude[0])==90 and float(ds.latitude[-1])==-90
                for var in ds.data_vars:
                    # Check each timestamp separately within the CPU-mode memory quota.
                    for index in range(len(ds[time])):
                        assert np.isfinite(ds[var].isel({time:index}).values).all()
            partial.replace(path)
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        record={'dataset':dataset,'request':{**common,**extra},'sha256':digest,'bytes':path.stat().st_size,
                'purpose':'observed input/target fields; not normalization statistics'}
        (out/f'{args.month}_{suffix}.json').write_text(json.dumps(record,indent=2))
        print('Verified ERA5 extras',path.name,flush=True)

if __name__=='__main__':
    main()
