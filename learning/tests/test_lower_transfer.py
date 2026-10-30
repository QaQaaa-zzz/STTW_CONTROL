import hashlib,json,pickle
import numpy as np
import pytest
import torch
from sttw_control.lower_transfer import export_lower_transfer


def fixture(tmp_path):
    from pathlib import Path
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/lower_local_tracking_no_alpha_v2.json').read_text())
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(cfg))
    base=dict(checkpoint_sha256='parent-sha',config=str(parent),config_file_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),normalizer={'actor_dim':210})
    cfg['weights_only_parent']={'checkpoint_sha256':'parent-sha'}
    torch.manual_seed(93)
    actor=torch.nn.Sequential(torch.nn.Linear(210,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,2))
    saved=dict(config=cfg,update=20,policy={'actor.'+k:v for k,v in actor.state_dict().items()})
    path=tmp_path/'source.pt';torch.save(saved,path)
    return path,base,saved,actor


def test_export_is_numeric_and_immutable(tmp_path):
    path,base,_,actor=fixture(tmp_path);out=tmp_path/'export'
    result=export_lower_transfer(path,base,out)
    with (out/'actor.pkl').open('rb') as f:params=pickle.load(f)['params']
    x=np.random.default_rng(41).normal(size=(3,210)).astype(np.float32)
    with torch.no_grad():expected=actor(torch.from_numpy(x)).numpy()
    for i in range(3):
        x=x@params[f'Dense_{i}']['kernel']+params[f'Dense_{i}']['bias']
        if i<2:x=np.where(x>0,x,np.expm1(x))
    np.testing.assert_allclose(x,expected,rtol=1e-5,atol=1e-6)
    frozen=(out/'checkpoint.pt').read_bytes();path.write_bytes(b'changed source')
    assert (out/'checkpoint.pt').read_bytes()==frozen
    assert result['candidate']['checkpoint_sha256']==hashlib.sha256(frozen).hexdigest()
    assert result['candidate']['status']=='not_adopted'


@pytest.mark.parametrize('change',['normalization','parent','structure'])
def test_incompatible_export_stops_before_writing(tmp_path,change):
    path,base,saved,_=fixture(tmp_path)
    if change=='normalization':saved['config']['observation_scales'][0]=.8
    elif change=='parent':saved['config']['weights_only_parent']['checkpoint_sha256']='other'
    else:saved['policy']['actor.extra']=torch.zeros(1)
    torch.save(saved,path)
    with pytest.raises(ValueError):export_lower_transfer(path,base,tmp_path/'export')
    assert not (tmp_path/'export').exists()
