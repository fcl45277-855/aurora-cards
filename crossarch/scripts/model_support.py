"""Isolated compatibility copies; upstream files are never overwritten."""
from pathlib import Path
import ast, hashlib, importlib.util, json, re, sys
from types import SimpleNamespace
import torch

from crossarch_paths import ROOT

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def prepare_compat():
    source=ROOT/'sources/ClimaX/src/climax'
    compat=ROOT/'sources/compat'
    records=[]
    for directory in [compat/'climax',compat/'climax/utils']:
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'__init__.py').write_text('# Isolated compatibility package; see metadata/compatibility_changes.json\n')
    for name in ['arch.py','parallelpatchembed.py','utils/pos_embed.py']:
        original=source/name
        text=original.read_text()
        revised=text.replace('drop=drop_rate,','proj_drop=drop_rate,').replace('dtype=np.float)','dtype=float)').replace('from timm.models.layers.helpers import to_2tuple','from timm.layers import to_2tuple').replace('torch.linspace(0, drop_path, depth)','torch.linspace(0, drop_path, depth, device="cpu")')
        target=compat/'climax'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(revised)
        changes=[]
        for old,label in [('drop=drop_rate,','drop → proj_drop; evaluation dropout disabled'),('dtype=np.float)','np.float → float'),('from timm.models.layers.helpers import to_2tuple','timm helper import path'),('torch.linspace(0, drop_path, depth)','Construct drop-path schedule on CPU to allow meta-device inventory')]:
            if old in text:
                changes.append(label)
        records.append({'source':str(original),'source_sha256':sha(original),
                        'copy':str(target),'copy_sha256':sha(target),
                        'changes':changes})
    # Preserve the upstream PeriodicPad2d class verbatim with its licence.
    image_source=ROOT/'sources/FourCastNet/utils/img_utils.py'
    text=image_source.read_text()
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PeriodicPad2d')
    helper='\n'.join(text.splitlines()[:text.splitlines().index('import logging')])+'\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'+ast.get_source_segment(text,node)+'\n'
    (compat/'fcn_pad.py').write_text(helper)
    afno=ROOT/'sources/FourCastNet/networks/afnonet.py'
    revised=afno.read_text().replace('from numpy.lib.arraypad import pad',' # unused legacy NumPy import removed').replace('from utils.img_utils import PeriodicPad2d','from fcn_pad import PeriodicPad2d').replace('torch.linspace(0, drop_path_rate, depth)','torch.linspace(0, drop_path_rate, depth, device="cpu")')
    (compat/'afnonet.py').write_text(revised)
    records.extend([{'source':str(afno),'source_sha256':sha(afno),'copy_sha256':sha(compat/'afnonet.py'),'changes':['unused NumPy import removed','PeriodicPad2d isolated from training-only dependencies','Drop-path initialization schedule explicitly on CPU for meta-device inventory']},
                    {'source':str(image_source),'source_sha256':sha(image_source),'class':'PeriodicPad2d','copy_sha256':sha(compat/'fcn_pad.py')}])
    (ROOT/'metadata/compatibility_changes.json').write_text(json.dumps(records,indent=2))
    sys.path.insert(0,str(compat))

def variables():
    text=(ROOT/'sources/ClimaX/configs/pretrain_climax.yaml').read_text()
    return re.findall(r'"([^"]+)"',text.split('default_vars: [',1)[1].split(']',1)[0])

def build_climax(tiny=False):
    from climax.arch import ClimaX
    import climax.arch
    assert Path(climax.arch.__file__).resolve().is_relative_to(ROOT/'sources/compat')
    return ClimaX(variables() if not tiny else ['2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind'],
                  img_size=[8,16] if tiny else [32,64],embed_dim=64 if tiny else 1024,
                  depth=2 if tiny else 8,num_heads=4 if tiny else 16,drop_path=0,drop_rate=0).eval()

def build_fcn(tiny=False):
    from afnonet import AFNONet
    params=SimpleNamespace(patch_size=8,N_in_channels=20,N_out_channels=20,num_blocks=8)
    return AFNONet(params,img_size=(16,32) if tiny else (720,1440),
                   embed_dim=128 if tiny else 768,depth=2 if tiny else 12).eval()

def canonical_climax_state(state):
    mapped={}
    for key,value in state.items():
        key=key.removeprefix('net.').replace('channel','var')
        if key in mapped:
            raise ValueError('Duplicate checkpoint key after mapping: '+key)
        mapped[key]=value
    return mapped

def load_climax(model,path,expected_sha,assign=False):
    if sha(path)!=expected_sha:
        raise ValueError('Checkpoint SHA256 mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['state_dict']
    # Never drop mismatched keys or initialize a missing prediction head.
    model.load_state_dict(canonical_climax_state(state),strict=True,assign=assign)
    return model
