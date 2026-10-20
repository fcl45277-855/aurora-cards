"""Create a handoff manifest only after all bounded CPU stages pass."""
from pathlib import Path
import datetime,hashlib,importlib.metadata,json,platform

from crossarch_paths import ROOT
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def read(name):
    return json.loads((ROOT/'metadata'/name).read_text())
preflight=read('cpu_preflight.json')
climax=read('climax_checkpoint_verified_cpu.json')
fcn=read('fourcastnet_checkpoint_verified_cpu.json')
inputs=read('input_validation_cpu.json')
assert preflight['synthetic_only'] and not preflight['gpu_used']
assert climax['strict_loading'] and climax['all_weights_finite'] and climax['tensor_count']==209
assert fcn['strict_loading'] and fcn['all_weights_finite'] and fcn['tensor_count']==150
assert len(inputs['inputs'])==24 and inputs['units_verified']
assert read('fourcastnet_asset_gate.json')['ready_assets']
assert not read('climax_asset_gate.json')['ready_assets']
assert not list((ROOT/'results').glob('*')), 'Unexpected scientific result exists; audit before finalizing'
assert not (ROOT/'metadata/frozen_predictor.json').exists(), 'Real predictor unexpectedly frozen'
for case in ['False','True']:
    path=ROOT/f'test_fixtures/analysis/synthetic_analysis_{case}.json'
    record=json.loads(path.read_text())
    assert record['synthetic_only']
    assert record['H1_top_quartile_supported']==(case=='False')
    assert record['H2_supported']==(case=='False')
packages={}
for name in ['torch','torchao','timm','numpy','scipy','xarray','netCDF4','cdsapi']:
    try:
        packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name]=None
source_files={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'sources').rglob('*') if p.is_file() and p.suffix in ['.py','.md','.yaml','.txt'] and 'release_cpu_check' not in p.parts and '__pycache__' not in p.parts}
scripts={p.name:digest(p) for p in (ROOT/'scripts').glob('*.py')}
manifest={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'server':'GPU server','experiment_root':str(ROOT),'python':platform.python_version(),'packages':packages,
          'gpu_enabled':Path('/dev/nvidia0').exists(),'gpu_experiments_started':False,'scientific_results_generated':False,
          'cpu_mode_memory_limit_bytes':int(Path('/sys/fs/cgroup/memory.max').read_text()),
          'cpu_implementation_checks_passed':len(preflight['checks']),
          'synthetic_analysis_positive_and_reverse_checks_passed':True,
          'models':{'climax':{'tensor_count':209,'attention_candidates':9,'asset_sha256':climax['checkpoint_sha256'],'strict_loading':True},
                    'fourcastnet':{'tensor_count':150,'candidate_groups':26,'asset_sha256':fcn['checkpoint_sha256'],'strict_loading':True,'optimizer_state_omitted':True}},
          'fourcastnet_input_days':24,'fourcastnet_inputs':inputs['inputs'],
          'fourcastnet_ready_except_gpu':True,
          'climax_forecast_ready_except_gpu':False,
          'climax_missing':['ERA5-finetuned task checkpoint and training provenance','Paired training normalization and exact grid coordinates','Real ClimaX inputs prepared using that verified task/grid'],
          'script_sha256':scripts,'source_sha256':source_files,
          'protocol_sha256':{name:digest(ROOT/name) for name in ['protocol.md','protocol_revision.md','input_contract.md','handoff.md']},
          'pressure_order_fix':'The first single-date preparation was isolated; all accepted inputs use version 2 with physical pressure-order detection and independent z500/t850 round-trip checks.',
          'submission_repository_commit':'9a9d843d4d3322c116796576e2ca287312d94d77'}
(ROOT/'metadata/cpu_preparation_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:manifest[k] for k in ['completed_utc','cpu_implementation_checks_passed','fourcastnet_input_days','fourcastnet_ready_except_gpu','climax_forecast_ready_except_gpu','gpu_experiments_started']},indent=2),flush=True)
