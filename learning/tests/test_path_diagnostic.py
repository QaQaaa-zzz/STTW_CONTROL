"""Fixed Actor probes alter physical targets without touching control state."""
import numpy as np
import jax
import jax.numpy as j
import pytest
from sttw_control.path_command_diagnostic import transform_latent, compare_prefix
from sttw_control.path_command_cache import diagnostic_cache_identity

CFG={'residual_interface':{'speed_positive_scale_m_s':.2,'speed_negative_scale_m_s':.8,'steer_scale_rad':.1}}

def test_noop_exact_rng():
    z=j.array([.7,-.4]);key=jax.random.PRNGKey(12);before=np.asarray(key).copy()
    out,log=transform_latent(z,CFG,chi=.8,time_s=4.,spec={'beta':1.,'speed_backoff':0.})
    np.testing.assert_array_equal(z,out);np.testing.assert_array_equal(key,before)
    np.testing.assert_array_equal(log['original_target'],log['applied_target'])

def test_mirror_and_asymmetric_speed_clipping():
    for sign in [-1,1]:
        out,log=transform_latent(j.array([-.5,sign*.7]),CFG,chi=1.,time_s=2.,spec={'beta':.5,'speed_backoff':.05})
        np.testing.assert_allclose(log['applied_target'],[.8*np.tanh(-.5)-.05,sign*.05*np.tanh(.7)],rtol=2e-6)
    out,log=transform_latent(j.array([-20.,.2]),CFG,chi=1.,time_s=0.,spec={'beta':.5,'speed_backoff':.05})
    assert int(log['target_clip_count'])==1
    assert -.8<=float(log['applied_target'][0])<=.2

def test_late_fade_prefix_exact_and_fault_preserved():
    z=j.array([.3,.7]);spec={'kind':'late_fade','start_s':10.,'duration_s':.5}
    for t,f in [(9.99,1),(10,1),(10.25,.5),(10.5,0),(15,0)]:
        out,log=transform_latent(z,CFG,chi=0.,time_s=t,spec=spec)
        assert float(out[0])==float(z[0])
        np.testing.assert_allclose(log['applied_target'][1],.1*np.tanh(.7)*f,atol=1e-7)
        if t<=10:np.testing.assert_array_equal(out,z)
    out,_=transform_latent(j.array([j.nan,.2]),CFG,chi=0.,time_s=11.,spec=spec)
    assert np.isnan(out[0])

def test_cache_and_prefix():
    base={'actor_sha256':'a','case_ids':['left']}
    assert diagnostic_cache_identity(base,{'beta':.5})!=diagnostic_cache_identity(base,{'beta':.75})
    assert diagnostic_cache_identity(base,{'beta':.5})!=diagnostic_cache_identity(dict(base,actor_sha256='b'),{'beta':.5})
    d={'time':np.arange(4)*.005,'final_command':np.ones((4,2)),'policy_fault':np.zeros(4,bool)}
    compare_prefix(d,d,10.)
    bad=dict(d,final_command=np.zeros((4,2)))
    with pytest.raises(AssertionError):compare_prefix(d,bad,10.)


def test_compiled_none_noop_and_transform_cache_separation():
    from test_path_recovery import state,path,config,trace
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_policy import PathCommandActor
    class Env:
        terminal_contract='terminal_contract_v2'
        def reset(self,snapshot,path,v,alpha):return state(0)
        def set_log_template(self,*args):pass
        def observation(self,s,p,v,h):return j.zeros(351),j.zeros(352),j.bool_(False)
        def policy_step(self,s,z,p,v,h):
            logs={k:j.asarray(value) for k,value in trace(0,4).items()}
            logs['time']=(s.tick+j.arange(4))*.005
            logs['latent_z']=j.broadcast_to(z,(4,2))
            return s.replace(tick=s.tick+4,offsets=s.offsets+z),logs
    e=BatchedPanelEvaluator(Env(),None,config());params=PathCommandActor().init(jax.random.PRNGKey(0),j.zeros(351))
    paths=jax.tree.map(lambda x:x[None],path());speeds=j.array([2.6]);states=jax.tree.map(lambda x:x[None],state(3980))
    _,original=e._compiled_group(paths,speeds,j.array(1.),params,20.)
    def noop(z,*args):return transform_latent(z,CFG,chi=1.,time_s=0.,spec={'beta':1.})
    _,diagnostic=e._compiled_group(paths,speeds,j.array(1.),params,20.,diagnostic_transform=noop,diagnostic_identity={'actor':'a','case':'left','beta':1})
    a,logs=original(states,paths,speeds,params);b,probe=diagnostic(states,paths,speeds,params)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    for k in logs:np.testing.assert_array_equal(logs[k],probe[k])
    assert int(probe['active_tick'].sum())==20
    assert int(probe['diagnostic_target_clip_count'].sum())==0
    assert len(e.compiled)==2
    _,different=e._compiled_group(paths,speeds,j.array(1.),params,20.,diagnostic_transform=noop,diagnostic_identity={'actor':'b','case':'left','beta':1})
    assert different is not diagnostic and len(e.compiled)==3
    with pytest.raises(ValueError):e._compiled_group(paths,speeds,j.array(1.),params,20.,diagnostic_transform=noop)


def test_immutable_output_before_any_physics(tmp_path):
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    out=tmp_path/'existing';out.mkdir()
    evaluator=BatchedPanelEvaluator(None,None,{})
    with pytest.raises(FileExistsError):evaluator.evaluate(None,1,out,case_names=['straight_2.0'])


def test_eight_declared_episodes_are_four_paired_calls():
    from pathlib import Path
    import json
    from sttw_control.path_command_diagnostic import diagnostic_groups
    request=json.loads((Path(__file__).parents[1]/'configs/path_safety_repair_diagnostic.json').read_text())
    groups=diagnostic_groups(request)
    assert len(groups)==4 and sum(map(len,groups))==8
    assert all(len(group)==2 for group in groups)
    assert all(len({row['model'] for row in group})==1 for group in groups)
    assert groups[-1][0]['model']=='u200'


def test_backoff_crosses_asymmetric_zero_without_scaling_speed_request():
    out,log=transform_latent(j.array([.05,.7]),CFG,chi=1.,time_s=3.,spec={'beta':.5,'speed_backoff':.05})
    requested=.2*np.tanh(.05)-.05
    assert requested<0
    np.testing.assert_allclose(log['applied_target'][0],requested,rtol=2e-6)
    np.testing.assert_allclose(out[0],np.arctanh(requested/.8),rtol=2e-6)
    assert int(log['target_clip_count'])==0


def test_subset_accounts_only_real_ticks_and_preserves_terminal(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_path_recovery import state,path,config,trace
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_policy import PathCommandActor
    import sttw_control.geometric_path as geometry
    cfg=config();cfg['stage_C_evaluation']={'route_ids':['left','right','straight'],'low_speed_m_s':2.,'conflict_speed_m_s':2.6}
    monkeypatch.setattr(geometry,'build_path',lambda *args:path())
    class Env:
        terminal_contract='terminal_contract_v2'
        physics=SimpleNamespace(helpers=SimpleNamespace(pose=lambda data:j.zeros(3)))
        def reset(self,snapshot,path,v,alpha):return state(0)
        def set_log_template(self,*args):pass
        def observation(self,s,p,v,h):return j.zeros(351),j.zeros(352),j.bool_(False)
        def policy_step(self,s,z,p,v,h):
            logs={k:j.asarray(value) for k,value in trace(0,4).items()}
            logs['time']=(s.tick+j.arange(4))*.005
            logs['physical_failure']=j.full(4,s.tick>=4)
            return s.replace(tick=s.tick+4,physical=s.physical.replace(failed=s.tick>=4)),logs
    paid=[];e=BatchedPanelEvaluator(Env(),state().physical,cfg,account=paid.append)
    params=PathCommandActor().init(jax.random.PRNGKey(0),j.zeros(351))
    result=e.evaluate(params,1,tmp_path/'panel',case_names=['left_2.6'])
    assert set(result)=={'left_2.6'} and len(result['left_2.6']['time'])==8
    assert sum(paid)==8 and result['left_2.6']['physical_failure'][-1]
    assert not (tmp_path/'panel'/'right_2.6.npz').exists()


def test_execution_manifest_rejects_changed_follower_and_timing():
    import copy,json
    from pathlib import Path
    from sttw_control.path_command_diagnostic import validate_execution_manifest
    from sttw_control.path_command_fork import physical_config_identity
    cfg=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Safety_candidate.json').read_text())
    frozen={'physical_config_identity':physical_config_identity(cfg),'source':{'bank':'identity'}}
    manifest={'config':cfg,'source':frozen['source'],'terminal_contract':'terminal_contract_v2'}
    validate_execution_manifest(manifest,frozen)
    bad=copy.deepcopy(manifest);bad['config']['residual_interface']['steer_scale_rad']*=2
    with pytest.raises(ValueError):validate_execution_manifest(bad,frozen)
    bad=copy.deepcopy(manifest);bad['config']['training_future_phase_C']['episode_s']=19.
    with pytest.raises(ValueError):validate_execution_manifest(bad,frozen)
    bad=copy.deepcopy(manifest);bad['config']['timing']['upper_dt_s']=.04
    with pytest.raises(ValueError):validate_execution_manifest(bad,frozen)


def test_cache_source_rejects_swapped_actor_but_accepts_parent_fresh_critic():
    from sttw_control.path_command_diagnostic import validate_panel_source
    from sttw_control.preference_best import config_hash
    cfg={'reward':'fixed'};anchor={'checkpoint_sha256':'parent','actor_sha256':'actor'}
    source={'alpha':1,'source_update':0,'parent_checkpoint_sha256':'parent','actor_sha256':'actor','checkpoint_sha256':'freshfork','config_sha256':config_hash(cfg),'lower_actor_sha256':'lower'}
    validate_panel_source(source,anchor,cfg,'lower',alpha=1,parent=True)
    with pytest.raises(ValueError):validate_panel_source(dict(source,actor_sha256='other'),anchor,cfg,'lower',alpha=1,parent=True)
    with pytest.raises(ValueError):validate_panel_source(dict(source,parent_checkpoint_sha256='other'),anchor,cfg,'lower',alpha=1,parent=True)


def test_cache_truncation_without_terminal_is_rejected(tmp_path):
    from test_path_recovery import trace,path,config
    from sttw_control.path_command_cache import _read_complete_panel
    from sttw_control.path_command_selection import cases
    cfg=config();cfg['stage_C_evaluation']={'route_ids':['left','right','straight'],'low_speed_m_s':2.,'conflict_speed_m_s':2.6}
    geometry=path();saved={k:np.asarray(getattr(geometry,k)) for k in ('s','xy','heading','curvature','goal','turn_end')}
    for name,_,speed in cases(cfg):
        data=trace(0,12);data['v_user'][:]=speed;data['actual_forward_speed'][:]=speed
        np.savez(tmp_path/(name+'.npz'),**data);np.savez(tmp_path/(name+'_path.npz'),**saved)
    with pytest.raises(ValueError,match='partial validation'):_read_complete_panel(tmp_path,1,cfg)


def test_original_path_hash_rejects_shifted_world_path(tmp_path):
    from sttw_control.path_command_diagnostic import validate_original_path
    from sttw_control.preference_best import digest
    path=tmp_path/'path.npz';np.savez(path,xy=np.array([[0.,0.],[1.,0.]]))
    frozen_sha=digest(path);validate_original_path(path,frozen_sha)
    np.savez(path,xy=np.array([[1.,0.],[2.,0.]]))
    with pytest.raises(ValueError,match='original path'):validate_original_path(path,frozen_sha)
