"""Independent schema, units, timestamps and channel round-trip checks."""
from pathlib import Path
import hashlib,json
import numpy as np
import xarray as xr
from crossarch_paths import ROOT
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
records=[]
mean=np.load(ROOT/'metadata/global_means.npy',allow_pickle=False)[0,:20,0,0]
std=np.load(ROOT/'metadata/global_stds.npy',allow_pickle=False)[0,:20,0,0]
for month in ['2024-01','2024-02','2024-03','2024-07']:
    for suffix,units in [('surface',{'sp':'Pa','tcwv':'kg m**-2'}),('humidity',{'r':'%'})]:
        path=ROOT/f'data/era5_extras/{month}_{suffix}.nc'
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            actual={name:ds[name].attrs.get('units') for name in units}
            for name,unit in units.items():
                if actual[name]!=unit:
                    raise ValueError(f'Unexpected units for {name}: {actual[name]}')
for path in sorted((ROOT/'data/fourcastnet').glob('????-??-??.npz')):
    sidecar=json.loads(path.with_suffix('.json').read_text())
    assert sidecar.get('preparation_version')==2
    assert digest(path)==sidecar['sha256']
    day=path.stem
    with np.load(path,allow_pickle=False) as data:
        frames={field:data[field] for field in ['x','y']}
        assert frames['x'].shape==frames['y'].shape==(20,720,1440)
        assert frames['x'].dtype==frames['y'].dtype==np.dtype('float32')
        assert np.isfinite(frames['x']).all() and np.isfinite(frames['y']).all()
        assert str(data['init_time'].item())==day+'T00:00:00Z'
        assert str(data['valid_time'].item())==day+'T06:00:00Z'
        assert list(data['variables'])==sidecar['channel_order']
        selected=np.s_[::120,::240]
        with np.load(f'{__import__("crossarch_paths").AQ_ERA5_DIR}/surface_{day}.npz',allow_pickle=False) as source:
            for fi,field in enumerate(['x','y']):
                actual=frames[field][0][selected]*std[0]+mean[0]
                np.testing.assert_allclose(actual,source['u10'][fi,:-1][selected],atol=1e-5,rtol=1e-5)
        with np.load(f'{__import__("crossarch_paths").AQ_ERA5_DIR}/air_{day}.npz',allow_pickle=False) as source:
            z=source['z']
            probe=z[0,:,::120,::240].mean((1,2))
            # Independent index by physical ordering, not the builder's sidecar.
            z500_index=7 if probe[0]>probe[-1] else 5
            for fi,field in enumerate(['x','y']):
                actual=frames[field][14][selected]*std[14]+mean[14]
                np.testing.assert_allclose(actual,z[fi,z500_index,:-1][selected],atol=0.1,rtol=1e-5)
            del z
            t=source['t']
            t850_index=10 if probe[0]>probe[-1] else 2
            for fi,field in enumerate(['x','y']):
                actual=frames[field][5][selected]*std[5]+mean[5]
                np.testing.assert_allclose(actual,t[fi,t850_index,:-1][selected],atol=1e-3,rtol=1e-5)
            del t
        with xr.open_dataset(ROOT/f'data/era5_extras/{day[:7]}_humidity.nc') as source:
            td='valid_time' if 'valid_time' in source.coords else 'time'
            pd='pressure_level' if 'pressure_level' in source.coords else 'level'
            for fi,field in enumerate(['x','y']):
                actual=frames[field][17][selected]*std[17]+mean[17]
                expected=source['r'].sel({td:np.datetime64(f'{day}T{fi*6:02d}:00:00'),pd:500}).values[:-1][selected]
                np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-5)
    records.append({'date':day,'sha256':sidecar['sha256'],'schema_roundtrip_verified':True})
    print('Input verified',day,flush=True)
assert len(records)==24, len(records)
(ROOT/'metadata/input_validation_cpu.json').write_text(json.dumps({'gpu_used':False,'inputs':records,'units_verified':True,'expected_count':24},indent=2))
print('All 24 original 20-channel inputs verified; no forecast inference performed',flush=True)
