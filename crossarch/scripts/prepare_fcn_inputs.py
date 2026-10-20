"""Assemble original 20-channel FourCastNet data without altering Aurora files."""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np
import xarray as xr

from crossarch_paths import ROOT
LEVELS=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
CHANNELS=['u10','v10','t2m','sp','msl','t850','u1000','v1000','z1000','u850','v850','z850','u500','v500','z500','t500','z50','r500','r850','tcwv']

def pressure_levels(z_probe):
    if z_probe.shape!=(len(LEVELS),):
        raise ValueError('Unexpected number of pressure levels')
    if np.all(np.diff(z_probe)<0):
        return LEVELS
    if np.all(np.diff(z_probe)>0):
        return LEVELS[::-1]
    raise ValueError('Cannot establish the known pressure-level order')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def assemble(day,era5):
    out=ROOT/'data/fourcastnet'; out.mkdir(parents=True,exist_ok=True)
    target=out/f'{day}.npz'
    if target.exists():
        raise FileExistsError('Existing prepared input is retained: '+str(target))
    sources=[era5/f'surface_{day}.npz',era5/f'air_{day}.npz',
             ROOT/f'data/era5_extras/{day[:7]}_surface.nc',ROOT/f'data/era5_extras/{day[:7]}_humidity.nc']
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
    mean=np.load(ROOT/'metadata/global_means.npy',allow_pickle=False)[0,:20,0,0]
    std=np.load(ROOT/'metadata/global_stds.npy',allow_pickle=False)[0,:20,0,0]
    lat=np.load(ROOT/'metadata/latitude.npy',allow_pickle=False)
    lon=np.load(ROOT/'metadata/longitude.npy',allow_pickle=False)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    np.testing.assert_allclose(lat,np.linspace(90,-90,721),atol=1e-6)
    np.testing.assert_allclose(lon,np.arange(1440)*0.25,atol=1e-6)
    frames=np.empty((2,20,720,1440),dtype=np.float32)
    with np.load(sources[0],allow_pickle=False) as surf, np.load(sources[1],allow_pickle=False) as air, xr.open_dataset(sources[2]) as sfc, xr.open_dataset(sources[3]) as humidity:
        # Aurora files can store the same known levels in either order.
        z_probe=air['z'][0,:,::120,::240].mean((1,2))
        stored_levels=pressure_levels(z_probe)
        for source in [sfc,humidity]:
            np.testing.assert_allclose(source.latitude,lat,atol=1e-6)
            np.testing.assert_allclose(source.longitude,lon,atol=1e-6)
        indices=[]
        for fi,hour in enumerate([0,6]):
            time=np.datetime64(f'{day}T{hour:02d}:00:00','ns')
            i=int(np.flatnonzero(surf['valid_time'].astype('datetime64[ns]')==time).item())
            j=int(np.flatnonzero(air['valid_time'].astype('datetime64[ns]')==time).item())
            st='valid_time' if 'valid_time' in sfc.coords else 'time'
            ht='valid_time' if 'valid_time' in humidity.coords else 'time'
            pl='pressure_level' if 'pressure_level' in humidity.coords else 'level'
            indices.append(j)
            for ci,channel in enumerate(CHANNELS):
                if channel in ['u10','v10','t2m','msl']:
                    field=surf[channel][i,:-1].copy()
                elif channel in ['sp','tcwv']:
                    field=sfc[channel].sel({st:time}).values[:-1].copy()
                elif channel[0]=='r':
                    field=humidity['r'].sel({ht:time,pl:int(channel[1:])}).values[:-1].copy()
                else:
                    continue
                assert field.shape==(720,1440) and np.isfinite(field).all(),channel
                frames[fi,ci]=field
        # Read each large pressure-level array once, and release it before the next.
        for variable in ['u','v','z','t']:
            pressure=air[variable]
            assert pressure.shape==(4,13,721,1440)
            for ci,channel in enumerate(CHANNELS):
                if channel[0]!=variable or not channel[1:].isdigit() or channel in ['u10','v10']:
                    continue
                for fi,j in enumerate(indices):
                    frames[fi,ci]=pressure[j,stored_levels.index(int(channel[1:])),:-1]
            del pressure
    frames-=mean.astype(np.float32)[None,:,None,None]
    frames/=std.astype(np.float32)[None,:,None,None]
    assert np.isfinite(frames).all()
    partial=target.with_suffix('.partial.npz')
    np.savez_compressed(partial,x=frames[0],y=frames[1],lat=lat[:-1],lon=lon,
                        variables=np.array(CHANNELS),init_time=np.array(day+'T00:00:00Z'),valid_time=np.array(day+'T06:00:00Z'))
    partial.replace(target)
    record={'file':str(target),'sha256':digest(target),'normalized':True,'lead_hours':6,'preparation_version':2,
            'stored_pressure_levels_hpa':stored_levels,
            'channel_order':CHANNELS,'crop':'exclude final south-pole row',
            'input_sources':{str(p):digest(p) for p in sources},
            'statistics':{str(ROOT/'metadata'/p):digest(ROOT/'metadata'/p) for p in ['global_means.npy','global_stds.npy']}}
    target.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print('Prepared and hashed',day,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dates',nargs='+')
    parser.add_argument('--era5',type=Path,default=__import__('crossarch_paths').AQ_ERA5_DIR)
    args=parser.parse_args()
    for day in args.dates:
        assemble(day,args.era5)
