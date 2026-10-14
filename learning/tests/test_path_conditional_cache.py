import copy
import json
from pathlib import Path
import pytest
from sttw_control import path_command_cache as cache
from sttw_control.path_command_training import load_plan
from sttw_control.preference_best import atomic_json


def config_pair():
    root=Path(__file__).parents[2]
    plan,new,_=load_plan(root/'learning/configs/path_unified_alpha_run.json')
    old=json.loads((Path(plan['baseline_parent'])/'manifest.json').read_text())
    return new,old


def test_conditional_baseline_allows_only_declared_training_metadata(tmp_path,monkeypatch):
    new,old=config_pair();parent=tmp_path/'parent';parent.mkdir()
    atomic_json(parent/'manifest.json',old)
    called=[]
    def reuse(parent,root,cfg,source,paths):
        called.append(cfg)
        assert cfg==old['config']
        return {case[0] for case in cache.cases(new)}
    monkeypatch.setattr(cache,'reuse_zero_upper',reuse)
    target=tmp_path/'new';target.mkdir()
    names=cache.reuse_conditional_baseline(parent,target,new,old['source'],{})
    assert len(names)==6 and len(called)==1
    assert json.loads((target/'baseline_reward_identity.json').read_text())['physics_executed']==0


@pytest.mark.parametrize('changed',['timing','follower','source','missing'])
def test_conditional_baseline_rejects_physics_or_incomplete_cache(tmp_path,monkeypatch,changed):
    new,old=config_pair();parent=tmp_path/'parent';parent.mkdir()
    atomic_json(parent/'manifest.json',old)
    source=copy.deepcopy(old['source'])
    if changed=='timing':new['timing']['lower_dt_s']=.01
    if changed=='follower':new['follower']['wheelbase_m_expected']+=.1
    if changed=='source':source['unexpected_changed_source']=True
    monkeypatch.setattr(cache,'reuse_zero_upper',lambda *a:set())
    target=tmp_path/'new';target.mkdir()
    with pytest.raises(ValueError):cache.reuse_conditional_baseline(parent,target,new,source,{})


def test_strong_baseline_changes_reward_network_and_curriculum_without_physics(tmp_path,monkeypatch):
    root=Path(__file__).parents[2]
    plan,new,_=load_plan(root/'learning/configs/path_strong_preference_run.json')
    old=json.loads((Path(plan['baseline_parent'])/'manifest.json').read_text())
    parent=tmp_path/'parent';parent.mkdir();atomic_json(parent/'manifest.json',old)
    monkeypatch.setattr(cache,'reuse_zero_upper',lambda *args:{name for name,_,_ in cache.cases(new)})
    target=tmp_path/'new';target.mkdir()
    assert len(cache.reuse_conditional_baseline(parent,target,new,old['source'],{}))==6
    new['residual_interface']['speed_negative_scale_m_s']+=.1
    with pytest.raises(ValueError,match='physical'):cache.reuse_conditional_baseline(parent,target,new,old['source'],{})


def test_scratch_acceptance_metadata_does_not_change_b0_physics(tmp_path,monkeypatch):
    root=Path(__file__).parents[2]
    plan,new,_=load_plan(root/'learning/configs/path_unified_scratch_run.json')
    old=json.loads((Path(plan['baseline_parent'])/'manifest.json').read_text())
    parent=tmp_path/'parent';parent.mkdir();atomic_json(parent/'manifest.json',old)
    monkeypatch.setattr(cache,'reuse_zero_upper',lambda *args:{name for name,_,_ in cache.cases(new)})
    target=tmp_path/'new';target.mkdir()
    assert len(cache.reuse_conditional_baseline(parent,target,new,old['source'],{}))==6
    new['preference_acceptance']['endpoint_path_rmse_m']=.14
    assert len(cache.reuse_conditional_baseline(parent,target,new,old['source'],{}))==6
    new['timing']['lower_dt_s']=.01
    with pytest.raises(ValueError,match='physical'):cache.reuse_conditional_baseline(parent,target,new,old['source'],{})
