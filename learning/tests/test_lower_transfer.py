import hashlib,json,pickle
import numpy as np
import pytest
import torch
from sttw_control.lower_transfer import export_lower_transfer,select_scored_lower_checkpoint


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


def test_selection_uses_next_rollout_and_actual_exposure(tmp_path):
    training=tmp_path/'training';checkpoints=training/'checkpoints';checkpoints.mkdir(parents=True)
    (tmp_path/'status.json').write_text(json.dumps({'state':'completed'}))
    (training/'train_best_candidate.json').write_text(json.dumps({'update':0,'mean_step_reward':0.}))
    rows=[]
    for update,reward,negative in [(1,0.,1),(11,-.5,1),(21,-.1,0),(31,-.2,1)]:
        groups={'nominal_low':{'samples':10,'exposed_ticks':0},'positive_high':{'samples':10,'exposed_ticks':1},'negative_low':{'samples':10,'exposed_ticks':negative}}
        rows.append(dict(update=update,mean_step_reward=reward,command_sampling={'disturbance_groups':groups}))
    (training/'metrics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    for update in (0,10,20,30,40):torch.save({'update':update},checkpoints/f'update_{update:04d}.pt')
    result=select_scored_lower_checkpoint(tmp_path)
    assert result['selected']['source_update']==30
    assert result['selected']['scored_by_rollout']==31
    assert result['unrestricted_best']['update']==0
    rejected={x['source_update']:x['reasons'] for x in result['candidates'] if not x['eligible']}
    assert rejected[0]==['parent_weights']
    assert rejected[20]==['missing_actual_direction_or_nominal_coverage']
    assert rejected[40]==['no_next_rollout_score']


def test_selection_does_not_read_live_checkpoints(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'state':'running'}))
    with pytest.raises(ValueError,match='completed training'):select_scored_lower_checkpoint(tmp_path)
