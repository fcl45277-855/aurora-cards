"""Bounded, resumable public WeatherBench2 format-2 reader; no GPU operations.

Missing chunks are errors, never interpreted as Zarr fill values. Source arrays
are (time, [level], longitude, latitude); our state arrays are (time,C,lat,lon).
"""
from pathlib import Path
import argparse, concurrent.futures as cf, datetime as dt, hashlib, json, os, sys, time, urllib.request
from crossarch_paths import ROOT
TASK=ROOT/'finetune_era5_20261020'
sys.path.insert(0,str(ROOT/'cpu_deps'))
import numpy as np
import numcodecs
BASE='https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr/'
LEVELS=[50,250,500,600,700,850,925]
FIELDS=['2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind',
        'geopotential','temperature','u_component_of_wind','v_component_of_wind','specific_humidity']
VARS=FIELDS[:3]+[f'{v}_{p}' for v in FIELDS[3:] for p in LEVELS]
OUT=['geopotential_500','temperature_850']+FIELDS[:3]
SPLITS={'plumbing':['2024-02-02'],'predictor':['2024-01-04','2024-01-06','2024-01-08'],
        'confirmation':[f'2024-03-{d:02d}' for d in range(1,13)],'summer':[f'2024-07-{d:02d}' for d in range(1,9)]}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def fetch(url,path,limit=20*1024**2):
    path=Path(path);record=path.with_suffix(path.suffix+'.http.json')
    if path.is_file() and record.is_file():
        r=json.loads(record.read_text())
        if path.stat().st_size==r['bytes'] and sha(path)==r['sha256']:return r
        raise ValueError('Cached source failed its recorded checksum: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(7):
        try:
            with urllib.request.urlopen(url,timeout=100) as f:
                data=f.read(limit+1)
                if len(data)>limit:raise ValueError('HTTP object exceeds explicit bound')
                size=f.headers.get('Content-Length')
                if size is not None and len(data)!=int(size):raise ValueError('Truncated HTTP object')
                r={'url':url,'bytes':len(data),'etag':f.headers.get('ETag'),
                   'generation':f.headers.get('x-goog-generation'),'last_modified':f.headers.get('Last-Modified'),
                   'sha256':hashlib.sha256(data).hexdigest(),'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            tmp=path.with_suffix(path.suffix+'.partial');tmp.write_bytes(data);tmp.replace(path);write_json(record,r)
            return r
        except Exception:
            if attempt==6:raise
            time.sleep(min(2**attempt,15))
def metadata():
    p=TASK/'wb2_zmetadata.json'
    fetch(BASE+'.zmetadata',p)
    return json.loads(p.read_text())['metadata']
def decode(m,name,key):
    p=TASK/'source_chunks'/name/key
    fetch(BASE+name+'/'+key,p)
    spec=m[name+'/.zarray']
    assert spec['zarr_format']==2 and spec['order']=='C' and spec['filters'] is None
    raw=numcodecs.get_codec(spec['compressor']).decode(p.read_bytes())
    a=np.frombuffer(raw,dtype=spec['dtype'])
    if a.size!=np.prod(spec['chunks']):raise ValueError('Decoded byte count differs from declared chunk shape')
    return a.reshape(spec['chunks'])
def setup():
    TASK.mkdir(parents=True,exist_ok=True);m=metadata()
    for name in ['latitude','longitude','level']:
        a=decode(m,name,'0')[:m[name+'/.zarray']['shape'][0]]
        np.save(TASK/(name+'.npy'),a)
    lat=np.load(TASK/'latitude.npy');lon=np.load(TASK/'longitude.npy')
    np.testing.assert_allclose(lat,np.linspace(-87.1875,87.1875,32),atol=1e-12)
    np.testing.assert_allclose(lon,np.arange(64)*5.625,atol=1e-12)
    lev=np.load(TASK/'level.npy');assert set(LEVELS)<=set(lev.tolist())
    ts=m['time/.zarray'];assert m['time/.zattrs']['units']=='hours since 1959-01-01'
    alltime=np.concatenate([decode(m,'time',str(i)) for i in range(4)])[:ts['shape'][0]]
    times=np.datetime64('1959-01-01T00','h')+alltime.astype('timedelta64[h]')
    np.testing.assert_array_equal(np.diff(times),np.full(len(times)-1,np.timedelta64(6,'h')))
    start=int(np.flatnonzero(times==np.datetime64('2010-01-01T00','h')).item())
    end=int(np.flatnonzero(times==np.datetime64('2017-01-01T00','h')).item())
    cut=int(np.flatnonzero(times[start:end]==np.datetime64('2016-01-01T00','h')).item())
    np.save(TASK/'times.npy',times[start:end])
    jobs=[{'field':field,'chunk':c} for field in FIELDS for c in range(start//100,(end-1)//100+1)]
    plan={'version':1,'task_id':'own_era5_2010_2016_6h_v1','created_before_gpu_training':True,
          'scope':'Own bounded-data ERA5 adaptation, not a replication of the published ClimaX benchmark',
          'source_url':BASE,'source_metadata_sha256':sha(TASK/'wb2_zmetadata.json'),
          'source_time_coverage':[str(times[0]),str(times[-1])],
          'source_axes':{v:m[v+'/.zattrs']['_ARRAY_DIMENSIONS'] for v in FIELDS},
          'source_units':{v:m[v+'/.zattrs'].get('units') for v in FIELDS},
          'start_source_index':start,'end_source_index_exclusive':end,'split_index':cut,
          'state_shape':[end-start,len(VARS),32,64],'state_dtype':'float32','variables':VARS,'out_variables':OUT,
          'lead_hours':6,'lead_embedding_divisor':100,
          'training':{'start':'2010-01-01T00','end':'2015-12-31T18','states':cut,'pairs':cut-1},
          'validation':{'start':'2016-01-01T00','end':'2016-12-31T18','states':end-start-cut,'pairs':end-start-cut-1},
          'normalization':'Unweighted population mean/std per variable over TRAIN states only, float64 Chan accumulation',
          'state_grid':'latitude ascending, longitude [0,360), 32x64, conservative WB2 5.625 degrees',
          'held_out_dates':SPLITS,'download_jobs':jobs,'max_source_object_bytes':20*1024**2,
          'pretrained':{'path':str(ROOT/'weights/climax_5.625deg.ckpt'),'sha256':'6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43'},
          'optimization':{'seed':20261020,'optimizer':'AdamW','lr':0.0001,'weight_decay':0.05,
                          'batch_size':2,'accumulate':4,'max_epochs':20,'min_epochs':5,'early_stopping_patience':4,
                          'warmup_epochs':1,'schedule':'cosine to 1e-6','gradient_clip':1.0,'precision':'FP32',
                          'full_finetuning':True,'checkpoint_selection':'Minimum 2016 latitude-weighted standardized MSE only'},
          'forecast_gate':'Selected checkpoint must beat 6h persistence on aggregate validation MSE AND individually on z500 and t850; otherwise stop before quantization results',
          'gpu_training_started':False,'scientific_quantization_runs':0}
    target=TASK/'task_plan.json'
    if target.exists():
        old=json.loads(target.read_text());assert old==plan,'Frozen task plan changed'
    else:write_json(target,plan)
    print('Frozen task',plan['state_shape'],'train/validation pairs',cut-1,end-start-cut-1,'jobs',len(jobs),flush=True)
    return m,plan
def download(workers):
    m,plan=setup();specs={v:m[v+'/.zarray'] for v in FIELDS}
    for v in FIELDS:
        expected=['time','longitude','latitude'] if v in FIELDS[:3] else ['time','level','longitude','latitude']
        assert m[v+'/.zattrs']['_ARRAY_DIMENSIONS']==expected
        assert specs[v]['chunks']==([100,64,32] if v in FIELDS[:3] else [100,13,64,32])
        assert specs[v]['dtype']=='<f4'
    dest=TASK/'states.npy';progress=TASK/'assembled_chunks.json'
    if not dest.exists():
        state=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=tuple(plan['state_shape']));state.flush();del state
        write_json(progress,[])
    done=json.loads(progress.read_text());donekeys={(j['field'],j['chunk']) for j in done}
    jobs=[j for j in plan['download_jobs'] if (j['field'],j['chunk']) not in donekeys]
    lev=np.load(TASK/'level.npy');indices=[lev.tolist().index(p) for p in LEVELS]
    def worker(job):
        v=job['field'];c=job['chunk'];key=str(c)+('.0.0' if v in FIELDS[:3] else '.0.0.0')
        a=decode(m,v,key)
        low=max(c*100,plan['start_source_index']);high=min((c+1)*100,plan['end_source_index_exclusive'])
        a=a[low-c*100:high-c*100]
        if v in FIELDS[:3]:a=a[:,None].transpose(0,1,3,2);cols=[VARS.index(v)]
        else:a=a[:,indices].transpose(0,1,3,2);cols=[VARS.index(f'{v}_{p}') for p in LEVELS]
        assert a.shape==(high-low,len(cols),32,64) and np.isfinite(a).all()
        state=np.load(dest,mmap_mode='r+');begin=low-plan['start_source_index'];end=high-plan['start_source_index']
        # Each job owns disjoint time/variable blocks; metadata/progress has one writer.
        state[begin:end,cols,:,:]=a;state.flush();del state
        return dict(job,key=key,source_sha256=sha(TASK/'source_chunks'/v/key),
                    decoded_slice_sha256=hashlib.sha256(a.tobytes()).hexdigest(),state_slice=[begin,end],channels=cols)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for future in cf.as_completed(futures):
            done.append(future.result());write_json(progress,done)
            print('Assembled',len(done),'/',len(plan['download_jobs']),done[-1]['field'],done[-1]['chunk'],flush=True)
    assert len(done)==len(plan['download_jobs'])
    summarize(plan,done)
def summarize(plan,done):
    state=np.load(TASK/'states.npy',mmap_mode='r');cut=plan['split_index']
    n=0;mu=np.zeros(len(VARS));m2=np.zeros(len(VARS))
    for start in range(0,len(state),32):
        a=np.asarray(state[start:min(start+32,len(state))],dtype=np.float64)
        assert np.isfinite(a).all()
        if start>=cut:continue
        a=a[:min(len(a),cut-start)]
        a=a.transpose(1,0,2,3).reshape(len(VARS),-1);count=a.shape[1]
        avg=a.mean(axis=1);variance=((a-avg[:,None])**2).sum(axis=1);delta=avg-mu
        m2+=variance+delta**2*n*count/(n+count);mu+=delta*count/(n+count);n+=count
    sigma=np.sqrt(m2/n);assert np.isfinite(sigma).all() and (sigma>0).all()
    np.savez(TASK/'normalize_mean.npz',**dict(zip(VARS,mu)))
    np.savez(TASK/'normalize_std.npz',**dict(zip(VARS,sigma)))
    write_json(TASK/'training_data_manifest.json',{'complete':True,'task_plan_sha256':sha(TASK/'task_plan.json'),
        'states':{'path':str(TASK/'states.npy'),'sha256':sha(TASK/'states.npy'),'shape':list(state.shape)},
        'times':{'path':str(TASK/'times.npy'),'sha256':sha(TASK/'times.npy')},
        'mean':{'path':str(TASK/'normalize_mean.npz'),'sha256':sha(TASK/'normalize_mean.npz')},
        'std':{'path':str(TASK/'normalize_std.npz'),'sha256':sha(TASK/'normalize_std.npz')},
        'statistics_values':{'variables':VARS,'mean':mu.tolist(),'std':sigma.tolist(),'samples_per_channel':n},
        'chunks':len(done),'chunks_manifest_sha256':sha(TASK/'assembled_chunks.json'),'gpu_training_started':False})
    print('TRAINING DATA COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setup-only',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    setup() if a.setup_only else download(a.workers)
