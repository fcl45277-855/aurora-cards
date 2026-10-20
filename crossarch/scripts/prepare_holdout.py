"""Prepare fixed real ERA5 holdout inputs using pinned WB2 conservative weights."""
from pathlib import Path
import ast, functools, hashlib, json, sys
import numpy as np
from scipy.sparse import csr_matrix
from prepare_training import ROOT,TASK,VARS,OUT,LEVELS,SPLITS,FIELDS,sha,write_json

FUNCTIONS={'_assert_increasing','_latitude_cell_bounds','_latitude_area_from_bounds',
 '_latitude_area','_latitude_overlap','_conservative_latitude_weights','_align_phase_with',
 '_periodic_upper_bounds','_periodic_lower_bounds','_periodic_upper_lower_bounds',
 '_longitude_length','_periodic_overlap','_longitude_overlap','_conservative_longitude_weights'}
def official_numpy():
    source=TASK/'sources/regridding.py'
    manifest=json.loads((TASK/'sources/source_manifest.json').read_text())
    assert sha(source)==manifest['files']['weatherbench2/regridding.py']['sha256']
    tree=ast.parse(source.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in selected}==FUNCTIONS
    # Only pinned pure weight functions are executed, with float64 NumPy in place
    # of JAX. No source imports, decorators, network or model operations execute.
    for n in selected:assert not n.decorator_list
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+selected,type_ignores=[])
    ns={'np':np,'jnp':np,'functools':functools};exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),ns)
    return ns,manifest
def weights():
    ns,manifest=official_numpy()
    slat=np.load(ROOT/'metadata/latitude.npy');slon=np.load(ROOT/'metadata/longitude.npy')
    lat=np.load(TASK/'latitude.npy');lon=np.load(TASK/'longitude.npy')
    np.testing.assert_allclose(slat,np.linspace(90,-90,721),atol=1e-8)
    np.testing.assert_allclose(slon,np.arange(1440)*.25,atol=1e-8)
    wlat=ns['_conservative_latitude_weights'](slat[::-1],lat,True,False)
    wlon=ns['_conservative_longitude_weights'](slon,lon,True,True)
    np.testing.assert_allclose(wlat.sum(1),1,atol=2e-14);np.testing.assert_allclose(wlon.sum(1),1,atol=2e-14)
    assert np.isfinite(wlat).all() and np.isfinite(wlon).all() and (wlat>=0).all() and (wlon>=0).all()
    sa=ns['_latitude_area'](slat[::-1],True);ta=ns['_latitude_area'](lat,False)
    # Conservation for ALL possible input fields follows from these column identities.
    np.testing.assert_allclose(ta@wlat,sa,atol=2e-15,rtol=2e-11)
    np.testing.assert_allclose(np.full(64,360/64)@wlon,np.full(1440,.25),atol=5e-13)
    W=csr_matrix(wlat);L=csr_matrix(wlon)
    def regrid(field):
        if field.shape!=(721,1440) or not np.isfinite(field).all():raise ValueError('Invalid source field')
        return (L@(W@field[::-1].astype('float64')).T).T.astype('float32')
    np.testing.assert_allclose(regrid(np.ones((721,1440))),1,atol=1e-7)
    north=regrid(np.repeat(slat[:,None],1440,axis=1))
    assert (np.diff(north[:,0])>0).all() and north[0,0]<0<north[-1,0]
    seam=np.zeros((721,1440));seam[:,0]=1
    sm=regrid(seam);assert (sm[:,0]>0).all() and np.count_nonzero(sm[:,1:])==0
    rng=np.random.default_rng(20261020);test=rng.normal(size=(721,1440))
    src=np.sum(test[::-1]*sa[:,None])*.25
    dst=np.sum(regrid(test)*ta[:,None])*5.625
    assert abs(src-dst)<1e-6
    np.savez(TASK/'conservative_weights.npz',latitude=wlat,longitude=wlon)
    write_json(TASK/'regridding_checks.json',{'passed':True,'source_commit':manifest['commit'],
       'operator':'Pinned official WB2 weight functions, NumPy float64 backend, sparse separable application',
       'checks':['row sums','nonnegative finite weights','latitude conservation all columns',
                 'longitude conservation all columns','constant field','north/south orientation','periodic seam','random-field area integral'],
       'random_field_integral_abs_error':float(abs(src-dst)),
       'source_axis':'latitude descending, longitude ascending','target_axis':'latitude ascending, longitude ascending',
       'weights_sha256':sha(TASK/'conservative_weights.npz'),
       'scope':'Mathematical operator checks; not a pointwise comparison with original WB2 high-resolution inputs'})
    return regrid,wlat,wlon
def prepare():
    regrid,wlat,wlon=weights();records=[]
    from prepare_fcn_inputs import pressure_levels
    lat=np.load(TASK/'latitude.npy');lon=np.load(TASK/'longitude.npy')
    out=TASK/'holdout';out.mkdir(exist_ok=True)
    for split,days in SPLITS.items():
      for day in days:
        dest=out/(day+'.npz');side=dest.with_suffix('.json')
        if dest.exists():
            rec=json.loads(side.read_text());assert sha(dest)==rec['sha256'];records.append(rec);continue
        paths=[__import__('crossarch_paths').AQ_ERA5_DIR/f'{prefix}_{day}.npz' for prefix in ['surface','air']]
        frames=np.empty((2,len(VARS),32,64),dtype='float32')
        desired=np.array([day+'T00:00:00',day+'T06:00:00'],dtype='datetime64[ns]')
        with np.load(paths[0],allow_pickle=False) as surf,np.load(paths[1],allow_pickle=False) as air:
            ids=[]
            for store in [surf,air]:
                t=store['valid_time'].astype('datetime64[ns]');ids.append([int(np.flatnonzero(t==v).item()) for v in desired])
            for name,key in zip(FIELDS[:3],['t2m','u10','v10']):
                field=surf[key]
                for k,i in enumerate(ids[0]):frames[k,VARS.index(name)]=regrid(field[i])
            for name,key in zip(FIELDS[3:],['z','t','u','v','q']):
                field=air[key]
                assert field.shape==(4,13,721,1440)
                if key=='z':levels=pressure_levels(field[ids[1][0],:,::120,::240].mean((1,2)))
                for p in LEVELS:
                    for k,i in enumerate(ids[1]):frames[k,VARS.index(f'{name}_{p}')]=regrid(field[i,levels.index(p)])
                del field
        assert np.isfinite(frames).all()
        # Physically distinct pressure levels independently guard against order reversal.
        assert 45000<frames[0,VARS.index('geopotential_500')].mean()<65000
        assert 220<frames[0,VARS.index('temperature_850')].mean()<310
        assert 160000<frames[0,VARS.index('geopotential_50')].mean()<250000
        partial=dest.with_suffix('.partial.npz')
        np.savez_compressed(partial,x=frames[0],y=frames[1,[VARS.index(v) for v in OUT]],
            variables=np.array(VARS),out_variables=np.array(OUT),lat=lat,lon=lon,
            init_time=np.array(day+'T00:00:00Z'),valid_time=np.array(day+'T06:00:00Z'))
        partial.replace(dest)
        rec={'date':day,'split':split,'path':str(dest),'sha256':sha(dest),'normalized':False,
             'source_pressure_levels_hpa':levels,'source_files':{str(p):sha(p) for p in paths},
             'conservative_weights_sha256':sha(TASK/'conservative_weights.npz'),
             'units':{'temperature':'K','geopotential':'m2 s-2','wind':'m s-1','specific_humidity':'kg kg-1'},
             'input_channels':38,'output_channels':5,'lead_hours':6}
        write_json(side,rec);records.append(rec);print('Prepared holdout',day,flush=True)
    write_json(TASK/'holdout_manifest.json',{'complete':True,'inputs':{s:[r for r in records if r['split']==s] for s in SPLITS},
          'task_plan_sha256':sha(TASK/'task_plan.json'),'no_forecast_or_quantization_run':True})
if __name__=='__main__':prepare()
