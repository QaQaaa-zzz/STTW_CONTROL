import jax
import jax.numpy as j
import numpy as np
from flax import struct


def test_jit_alpha_ramp_preserves_closed_loop_state_and_is_causal():
    from sttw_control.path_command_dynamic import apply_dynamic_alpha
    @struct.dataclass
    class State:
        alpha: object
        tick: object
        history: object
        offsets: object
    cfg={'timing':{'upper_dt_s':.02,'lower_dt_s':.005},'dynamic_preference_checks':{'rate_per_s':.5,'first_change_s':.5,'return_s':6.,'max_lower_ticks':24000}}
    st=State(j.array(0.),j.int32(96),j.arange(10.),j.array([.2,.3]))
    step=jax.jit(lambda s:apply_dynamic_alpha(s,0,cfg))
    before,request=step(st)
    assert float(before.alpha)==0 and float(request)==0
    after,request=step(st.replace(tick=j.int32(100)))
    np.testing.assert_allclose(after.alpha,.01,atol=1e-7)
    assert float(request)==1
    np.testing.assert_array_equal(after.history,st.history)
    np.testing.assert_array_equal(after.offsets,st.offsets)
    assert int(after.tick)==100
    returned,request= jax.jit(lambda s:apply_dynamic_alpha(s,2,cfg))(st.replace(alpha=j.array(1.),tick=j.int32(1200)))
    assert float(request)==0
    np.testing.assert_allclose(returned.alpha,.99,atol=1e-7)


def test_unqualified_dynamic_writes_not_run_without_loading_physics(tmp_path):
    from sttw_control.path_command_dynamic import run_dynamic_preference_checks
    receipt=run_dynamic_preference_checks(tmp_path,{}, {'qualified_preference':False})
    assert receipt['state']=='not_run' and receipt['active_lower_ticks']==0
    assert (tmp_path/'dynamic_preference_checks'/'receipt.json').exists()
    assert 'fixed-alpha preference not qualified' in (tmp_path/'dynamic_preference_checks/INDEX.md').read_text()


def test_real_compiled_batch_ramp_reward_alignment_and_failure_truncation():
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_policy import make_path_actor
    @struct.dataclass
    class Physical:
        failed: object
    @struct.dataclass
    class State:
        alpha: object
        tick: object
        physical: object
        fault: object
        domain_exit: object
        history: object
    class Env:
        terminal_contract='terminal_contract_v2'
        def reset(self,snapshot,path,v,alpha):
            return State(alpha,j.int32(0),Physical(j.bool_(False)),j.bool_(False),j.bool_(False),j.int32(17))
        def observation(self,state,path,v,h):
            obs=j.zeros(351).at[336].set(state.alpha)
            return obs,j.zeros(352),j.bool_(False)
        def set_log_template(self,*args):pass
        def policy_step(self,state,z,path,v,h):
            def one(st,_):
                active=~st.physical.failed
                nxt=st.replace(tick=st.tick+active.astype(j.int32),history=st.history+active.astype(j.int32),
                    physical=Physical(st.physical.failed|(active&(st.tick+1>=path[0]))))
                return nxt,dict(active_tick=active,upper_alpha=j.where(active,st.alpha,0.),
                    reward=j.where(active,st.alpha,0.),actor_output=j.where(active,z[0],0.))
            return jax.lax.scan(one,state,None,length=4)
    cfg={'network':{'architecture':'alpha_modulated_v1'},'timing':{'upper_dt_s':.02,'lower_dt_s':.005},
        'dynamic_preference_checks':{'rate_per_s':.5,'first_change_s':.5,'return_s':6.,'max_lower_ticks':24000}}
    actor=make_path_actor(cfg);params=actor.init(jax.random.PRNGKey(7),j.zeros(351))
    params['params']['gain_2']=j.ones(128)*.2
    paths=j.array([[101.,0.]]+[[400.,0.]]*5);speeds=j.ones(6);modes=j.array([0,1,2,0,1,2],j.int32)
    evaluator=BatchedPanelEvaluator(Env(),None,cfg)
    reset,chunk=evaluator._compiled_group(paths,speeds,j.where(modes==1,1.,0.),params,2.,dynamic_modes=modes)
    states=reset(paths,speeds,j.where(modes==1,1.,0.));states,logs=chunk(states,paths,speeds,params)
    np.testing.assert_array_equal(np.array(states.tick),[101,200,200,200,200,200])
    np.testing.assert_array_equal(np.array(states.history),np.array(states.tick)+17)
    active=np.array(logs['active_tick']);assert active.sum()==1101
    applied=np.array(logs['alpha_applied']);logged=np.array(logs['upper_alpha'])
    np.testing.assert_allclose(logged[active],applied[active],atol=1e-7)
    np.testing.assert_allclose(np.array(logs['reward'])[active],applied[active],atol=1e-7)
    # Actor receives the same causal ramp, rather than the request endpoint.
    for offset in (25,40):
        alpha=applied[offset,3,0]
        expected=actor.apply(params,j.zeros(351).at[336].set(alpha))[0]
        np.testing.assert_allclose(logs['actor_output'][offset,3,0],expected,atol=1e-6)
    assert applied[24,3,0]==0 and np.isclose(applied[25,3,0],.01)


def _qualified_fixture(tmp_path):
    import json,pickle
    from pathlib import Path
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.preference_best import digest
    from sttw_control.path_command_policy import config_key
    cfg=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Strong_Preference.json').read_text())
    actor=tmp_path/'actor.pkl';params=make_path_actor(cfg).init(jax.random.PRNGKey(1),j.zeros(351));actor.write_bytes(pickle.dumps(params))
    ck=tmp_path/'model.pt';ck.write_bytes(b'learner')
    meta=dict(config_key=config_key(cfg),architecture='alpha_modulated_v1',reward_version='path_safety_preference_v2',
        completed=True,actor_parameters_sha256=None)
    best=dict(qualified_preference=True,panel_identity=meta,actor=str(actor),checkpoint=str(ck),
        actor_sha256=digest(actor),checkpoint_sha256=digest(ck),selection_schema='path_strong_preference_v1')
    return cfg,best


def test_partial_dynamic_parent_does_not_replay_or_exceed_lineage_budget(tmp_path):
    from sttw_control.path_command_dynamic import run_dynamic_preference_checks
    from sttw_control.preference_best import atomic_json
    cfg,best=_qualified_fixture(tmp_path)
    parent=tmp_path/'parent/dynamic_preference_checks';parent.mkdir(parents=True)
    atomic_json(parent/'receipt.json',dict(state='error',active_lower_ticks=1200,
        actor_sha256=best['actor_sha256'],checkpoint_sha256=best['checkpoint_sha256'],panel_identity=best['panel_identity']))
    class Evaluator:
        config=cfg
        def evaluate(self,*args,**kw):raise AssertionError('partial parent may not silently restart six cases')
    receipt=run_dynamic_preference_checks(tmp_path/'new',cfg,best,evaluator=Evaluator(),resume_parent=tmp_path/'parent')
    assert receipt['state']=='not_run' and receipt['active_lower_ticks']==0
    assert receipt['parent_consumed_lower_ticks']==1200 and receipt['remaining_lineage_lower_ticks']==22800
    assert 'replay prohibited' in (tmp_path/'new/dynamic_preference_checks/INDEX.md').read_text()


def test_completed_dynamic_parent_is_verified_and_reused_with_zero_physics(tmp_path):
    from pathlib import Path
    from sttw_control.path_command_dynamic import run_dynamic_preference_checks
    from sttw_control.preference_best import atomic_json,digest
    from sttw_control.path_command_selection import cases
    cfg,best=_qualified_fixture(tmp_path);parent=tmp_path/'parent/dynamic_preference_checks';folder=parent/'traces';folder.mkdir(parents=True)
    records={};names=[]
    for name,route,speed in cases(cfg):
        if route=='straight' or speed!=2.6:continue
        for label,initial in [('0to1',0.),('1to0',1.),('0to1to0',0.)]:
            key=name+'_'+label;names.append(key);n=4000;t=np.arange(n)*.005
            requested=np.where(t<.5,initial,1-initial)
            if label=='0to1to0':requested=np.where(t>=6.,0.,requested)
            alpha=np.empty(n);applied=initial
            for start in range(0,n,4):
                applied+=np.clip(requested[start]-applied,-.01,.01);alpha[start:start+4]=applied
            d=dict(time=np.arange(n)*.005,actual_forward_speed=np.full(n,speed),v_user=np.full(n,speed),
                path_cross_track=np.zeros(n),path_heading_error=np.zeros(n),physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool),
                upper_alpha=alpha,alpha_applied=alpha,alpha_request=requested,
                goal_section_signed_distance=np.ones(n),path_progress=np.linspace(0,20,n),goal_progress=np.full(n,10),peak_roll=np.zeros(n))
            d.update(actual_delta=np.zeros(n),actual_xy=np.column_stack([t*speed,np.zeros(n)]),chi=np.ones(n),phi=np.zeros(n),phi_dot=np.zeros(n),
                offsets=np.zeros((n,2)),target_offset=np.zeros((n,2)),filtered_offset=np.zeros((n,2)),
                nominal=np.column_stack([np.full(n,speed),np.zeros(n)]),governed=np.column_stack([np.full(n,speed),np.zeros(n)]),final_command=np.zeros((n,2)))
            from sttw_control.path_command_reporting import reward_series
            d['scored_tick_reward']=reward_series(d,alpha,cfg)[0]
            baseline=tmp_path/'new/zero_upper';baseline.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(baseline/f'{name}.npz',**d)
            np.savez_compressed(baseline/f'{name}_path.npz',s=[0,20],xy=[[0,0],[20,0]])
            np.savez_compressed(folder/f'{key}.npz',**d);np.savez_compressed(folder/f'{key}_path.npz',s=[0,20],xy=[[0,0],[20,0]])
            for file in [folder/f'{key}.npz',folder/f'{key}_path.npz']:records[file.name]=digest(file)
    model_reference=dict(actor=str(Path(best['actor']).resolve()),checkpoint=str(Path(best['checkpoint']).resolve()),source_update=best.get('source_update'))
    atomic_json(parent/'receipt.json',dict(state='completed',selected_model=model_reference,active_lower_ticks=24000,lineage_lower_ticks=24000,cases=names,
        actor_sha256=best['actor_sha256'],checkpoint_sha256=best['checkpoint_sha256'],panel_identity=best['panel_identity'],trace_files_sha256=records))
    class Evaluator:
        config=cfg
        def evaluate(self,*args,**kw):raise AssertionError('complete parent should not repeat physics')
    receipt=run_dynamic_preference_checks(tmp_path/'new',cfg,best,evaluator=Evaluator(),resume_parent=tmp_path/'parent')
    assert receipt['state']=='completed' and receipt['active_lower_ticks']==0
    assert receipt['reused_cases']==6 and receipt['lineage_lower_ticks']==24000
    other=dict(best,source_update=999)
    refused=run_dynamic_preference_checks(tmp_path/'different_model',cfg,other,evaluator=Evaluator(),resume_parent=tmp_path/'parent')
    assert refused['state']=='not_run' and refused['active_lower_ticks']==0
    assert len(list((tmp_path/'new/dynamic_preference_checks/traces').glob('*_path.npz')))==6

    assert (tmp_path/'new/dynamic_preference_checks/INDEX.md').exists()
    assert len(list((tmp_path/'new/dynamic_preference_checks').glob('*_XY.png')))==2
    assert len(list((tmp_path/'new/dynamic_preference_checks').glob('*_rewards.npz')))==6
    from sttw_control.path_command_dynamic import _metrics
    import pytest
    sample=dict(np.load(folder/f'{names[0]}.npz'))
    assert _metrics(dict(sample,time=sample['time'].astype(np.float32)),cfg,0)['observed_lower_ticks']==4000
    broken=dict(sample,alpha_request=np.zeros(4000))
    with pytest.raises(ValueError,match='ramp contract'):_metrics(broken,cfg,0)
    broken=dict(sample,time=sample['time']+.005)
    with pytest.raises(ValueError,match='true prefix'):_metrics(broken,cfg,0)
    from sttw_control.path_command_dynamic import report_dynamic_preference
    import pytest


def _research_config():
    from pathlib import Path
    import json
    c=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Strong_Preference.json').read_text())
    c['preference_selection_schema']='path_preference_research_v1'
    c['preference_acceptance']=dict(spatial_bin_width_m=.5,endpoint_path_rmse_m=.15,
        endpoint_speed_rmse_m_s=.10,min_speed_gap_m_s=.02,min_path_gap_m=.001,
        final_hold_s=.5,final_speed_error_m_s=.15,final_path_error_m=.20,
        final_heading_error_rad=.10,adjacent_monotonicity_role='diagnostic',
        adjacent_speed_reverse_m_s=.01,adjacent_path_reverse_m=.01,
        roll_limit_role='diagnostic',corridor_limit_role='diagnostic')
    c['dynamic_preference_checks'].update(enabled=True,trigger='qualified_research_preference')
    return c


def test_research_dynamic_hold_uses_current_profile_and_preserves_strict_diagnostic():
    from sttw_control.path_command_dynamic import _metrics
    cfg=_research_config();n=4000;t=np.arange(n)*.005
    request=np.where(t<.5,0.,1.);applied=np.empty(n);a=0.
    for start in range(0,n,4):
        a+=np.clip(request[start]-a,-.01,.01);applied[start:start+4]=a
    trace=dict(time=t,actual_forward_speed=np.full(n,2.12),v_user=np.full(n,2.),
        path_cross_track=np.full(n,.15),path_heading_error=np.full(n,.08),
        physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool),
        upper_alpha=applied,alpha_applied=applied,alpha_request=request,
        goal_section_signed_distance=np.ones(n),path_progress=np.full(n,20.),
        goal_progress=np.full(n,10.),peak_roll=np.zeros(n))
    m=_metrics(trace,cfg,0)
    assert m['qualified'] and m['final_hold'] and m['hold_lower_ticks']==100
    assert m['hold_window_s']==.5 and not m['legacy_strict_final_hold']


def test_explicit_disabled_dynamic_does_not_load_selected_model(tmp_path):
    from sttw_control.path_command_dynamic import run_dynamic_preference_checks
    cfg=_research_config();cfg['dynamic_preference_checks']['enabled']=False
    receipt=run_dynamic_preference_checks(tmp_path,cfg,{'qualified_research_preference':True})
    assert receipt['state']=='not_run' and receipt['reason']=='dynamic checks disabled'


def test_research_trigger_does_not_fall_back_to_legacy_qualification(tmp_path):
    from sttw_control.path_command_dynamic import run_dynamic_preference_checks
    cfg=_research_config()
    receipt=run_dynamic_preference_checks(tmp_path,cfg,{'qualified_preference':True,'qualified_research_preference':False})
    assert receipt['state']=='not_run' and receipt['active_lower_ticks']==0


def test_research_dynamic_rejects_mismatched_acceptance_before_physics(tmp_path):
    import pytest
    from sttw_control.path_command_dynamic import run_dynamic_preference_checks
    from sttw_control.path_command_selection import resolve_preference_acceptance
    from sttw_control.path_command_policy import config_key
    _,best=_qualified_fixture(tmp_path);cfg=_research_config()
    best.update(qualified_research_preference=True,selection_schema='path_preference_research_v1',acceptance_key='stale')
    best['panel_identity'].update(config_key=config_key(cfg),acceptance_key=config_key(resolve_preference_acceptance(cfg)))
    class Evaluator:
        config=cfg
        def evaluate(self,*args,**kw):raise AssertionError('acceptance mismatch may not run physics')
    with pytest.raises(ValueError,match='acceptance identity'):
        run_dynamic_preference_checks(tmp_path/'new',cfg,best,evaluator=Evaluator())
