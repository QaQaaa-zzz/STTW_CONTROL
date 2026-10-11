"""Recovery preserves all closed-loop state and the original absolute clock."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as j
from sttw_control.path_command_env import PathState
from sttw_control.geometric_path import Path,Projection
from sttw_control.path_reference import Pursuit
from sttw_control.governor_state import Snapshot,GovernorState
from sttw_control.controller import ControllerState
from sttw_control.actuator import ActuatorState
from sttw_control.frozen_local_lower import LocalLowerState
from sttw_control.path_command_recovery import (save_snapshot,load_snapshot,clone_for_alpha,
    recovery_oracle,evaluate_from_state,first_qualifying_boundary)


def state(tick=4):
    physical=Snapshot(dict(qpos=j.arange(7.),qvel=j.arange(6.),warmstart=j.arange(6.)+.1),
        ControllerState(j.arange(3.)+.3,j.array(.4),j.arange(3.)+.5),
        ActuatorState(j.arange(6.).reshape(3,2),j.array([.2,.3])),j.int32(777),
        j.array([2.6,.1]),j.array([1.,2.,.5]),j.array(.5),j.array(.6),
        GovernorState(j.ones(2),j.ones(2)*2,j.int32(3),j.array(.4),j.int32(5),j.int32(1)),j.bool_(False))
    return PathState(physical,LocalLowerState(j.ones((10,20)),j.ones(10),j.array([.4,.3]),j.array(.2)),
        dict(frames=j.arange(20.).reshape(2,10),mask=j.ones(2)),
        Projection(j.array(1.),j.array(.9),j.ones(2),j.array(.03),j.array(.1)),
        Pursuit(j.ones(2),j.ones(2),j.ones(2),j.array(.8),j.array(.1),j.bool_(False),j.bool_(False)),
        j.array([.1,.2]),j.array([.3,.4]),j.ones(2),j.ones(2),j.array(.1),j.int32(tick),j.array(0.),j.bool_(False),j.bool_(False))


def path():
    return Path(j.arange(5.),j.stack([j.arange(5.),j.zeros(5)],-1),j.zeros(5),j.array([0.,.5,.5,0.,0.]),j.array(3.),j.array(4.))


def identity():
    return dict({k: 'verified-'+k for k in ('schema','lower','model','controller','actuator','timing','bank','physical_config','terminal_contract')},terminal_contract='terminal_contract_v2')


def trace(start,n,roll=.27):
    t=np.arange(start,start+n)*.005
    return dict(time=t,active_tick=np.ones(n,bool),phi=np.full(n,roll),peak_roll=np.full(n,abs(roll)),
        physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool),policy_fault=np.zeros(n,bool),lower_fault=np.zeros(n,bool),
        actual_forward_speed=np.full(n,2.6),v_user=np.full(n,2.6),path_cross_track=np.zeros(n),path_heading_error=np.zeros(n),
        path_progress=np.full(n,5.),goal_progress=np.full(n,4.),goal_section_signed_distance=np.ones(n),chi=np.ones(n))


def takeover_prefix(n=4):
    d=trace(0,n);d['phi'][-1]=.31;d['peak_roll'][-1]=.31
    return d


def config():
    return dict(timing=dict(lower_dt_s=.005,upper_dt_s=.02),training_future_phase_C=dict(episode_s=20.),
        stage_B_baseline=dict(hold_s=1.,final_y_error_m=.1,final_path_heading_error_rad=.05,final_speed_error_m_s=.1,working_roll_rad=.302))


def test_all_leaves_exact_roundtrip_and_only_alpha_changes(tmp_path):
    s=state();p=path();prefix=trace(0,4,.31)
    save_snapshot(tmp_path/'snapshot',s,p,prefix,identity(),route='left90_R2',v_user=2.6,horizon=20.,boundary_phi=.31)
    restored,restored_path,meta,logs=load_snapshot(tmp_path/'snapshot',s,identity())
    for a,b in zip(jax.tree.leaves((s,p)),jax.tree.leaves((restored,restored_path))):
        np.testing.assert_array_equal(a,b);assert a.dtype==b.dtype
        assert getattr(a,'weak_type',False)==getattr(b,'weak_type',False)
    np.testing.assert_array_equal(logs['phi'],prefix['phi'])
    # The live reset template may have different scalar weak types; saved types win.
    strong_template=jax.tree.map(lambda x:j.asarray(np.asarray(x)),s)
    weak_restored,_,_,_=load_snapshot(tmp_path/'snapshot',strong_template,identity())
    for a,b in zip(jax.tree.leaves(s),jax.tree.leaves(weak_restored)):
        assert getattr(a,'weak_type',False)==getattr(b,'weak_type',False)
    cloned=clone_for_alpha(restored,1.)
    for a,b in zip(jax.tree.leaves(s.replace(alpha=j.array(1.))),jax.tree.leaves(cloned)):
        np.testing.assert_array_equal(a,b)
        assert getattr(a,'weak_type',False)==getattr(b,'weak_type',False)
    assert int(cloned.physical.physical_tick)==777 and int(cloned.tick)==4
    wrong=identity();wrong['lower']='wrong'
    with pytest.raises(ValueError,match='identity'):load_snapshot(tmp_path/'snapshot',s,wrong)
    with pytest.raises(ValueError,match='signature'):load_snapshot(tmp_path/'snapshot',s.replace(history={'frames':j.ones((3,10)),'mask':j.ones(3)}),identity())
    payload=tmp_path/'snapshot'/'leaves.npz';payload.write_bytes(payload.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):load_snapshot(tmp_path/'snapshot',s,identity())


def test_snapshot_rejects_nonboundary_fault_and_missing_signatures(tmp_path):
    for s in [state(3),state().replace(fault=j.bool_(True)),state().replace(domain_exit=j.bool_(True))]:
        with pytest.raises(ValueError):save_snapshot(tmp_path/'bad',s,path(),trace(0,int(s.tick)),identity(),route='left90_R2',v_user=2.6,horizon=20.,boundary_phi=.31)
    incomplete=identity();incomplete.pop('bank')
    with pytest.raises(ValueError,match='identity'):save_snapshot(tmp_path/'bad',state(),path(),trace(0,4),incomplete,route='left90_R2',v_user=2.6,horizon=20.,boundary_phi=.31)


def test_oracle_complete_hold_deadline_and_initial_violation():
    tail=trace(4,3996);tail['peak_roll'][:80]=.31
    summary=recovery_oracle(tail,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=takeover_prefix())
    assert summary['qualified'] and summary['recovery_completed_s']==pytest.approx(.9)
    assert summary['initial_overlimit_duration_s']==pytest.approx(.405)
    assert summary['takeover_peak_roll_rad']==pytest.approx(.31)
    tail['peak_roll'][1000]=.303
    assert not recovery_oracle(tail,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=takeover_prefix())['qualified']
    late=trace(4,3996);late['peak_roll'][:101]=.31
    assert not recovery_oracle(late,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=takeover_prefix())['qualified']
    with pytest.raises(ValueError,match='contiguous'):recovery_oracle(trace(0,4),config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=takeover_prefix())


def test_complete_clone_deterministic_continuation_and_absolute_horizon():
    class Executor:
        config=config();env=SimpleNamespace(terminal_contract='terminal_contract_v2');check_budget=staticmethod(lambda:None)
        accounted=0
        def account(self,n):self.accounted+=n
        def _compiled_group(self,paths,speeds,alpha,params,horizon):
            assert horizon==20.
            def chunk(states,paths,speeds,params):
                start=int(states.tick[0]);n=min(200,4000-start)
                d=trace(start,200);d['active_tick'][n:]=False
                d={k:j.asarray(v.reshape(50,1,4)) for k,v in d.items()}
                end=states.replace(tick=states.tick+n,physical=states.physical.replace(controller=states.physical.controller.replace(eso=states.physical.controller.eso+states.alpha[:,None]+n*.01)))
                return end,d
            return None,chunk
    e=Executor();s=state(3980)
    a,ta=evaluate_from_state(e,s,path(),2.6,{},0)
    b,tb=evaluate_from_state(e,s,path(),2.6,{},0)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(ta['time'],tb['time']);assert len(ta['time'])==20
    assert float(ta['time'][0])==pytest.approx(19.9) and int(a.tick)==4000
    assert e.accounted==40


def test_first_boundary_is_natural_and_ignores_padding():
    history=jax.tree.map(lambda x:j.stack([x,x,x]),state())
    history=history.replace(tick=j.array([4,8,8]))
    index=first_qualifying_boundary(history,np.array([.29,-.32,.32]),np.array([True,True,False]))
    assert index==1
    assert first_qualifying_boundary(history,np.array([.29,.41,.32]),np.array([True,True,False])) is None


def test_reused_compiled_executor_twin_continuation_contains_hidden_states():
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_policy import PathCommandActor
    class SmallEnv:
        terminal_contract='terminal_contract_v2'
        def reset(self,snapshot,path,v,alpha):return state(0).replace(alpha=alpha)
        def set_log_template(self,*args):pass
        def observation(self,s,path,v,horizon):
            # ESO, delay, both histories and filters affect the next command.
            hidden=s.physical.controller.eso.sum()+s.physical.actuator.pending.sum()+s.lower.frames.sum()+s.history['frames'].sum()+s.filtered.sum()
            return j.ones(351)*hidden*.001,j.zeros(352),j.bool_(False)
        def policy_step(self,s,z,path,v,horizon):
            end=s.replace(tick=s.tick+4,offsets=s.offsets+z*.001,filtered=s.filtered*.9+z*.01,
                lower=s.lower.replace(frames=s.lower.frames*.99+z.sum()*.01),
                physical=s.physical.replace(controller=s.physical.controller.replace(eso=s.physical.controller.eso+z.sum())))
            logs={k:j.asarray(value) for k,value in trace(0,4).items()}
            logs['time']=(s.tick+j.arange(4))*.005
            return end,logs
    e=BatchedPanelEvaluator(SmallEnv(),None,config())
    params=PathCommandActor().init(jax.random.PRNGKey(9),j.zeros(351))
    params['params']['Dense_3']['kernel']=j.ones_like(params['params']['Dense_3']['kernel'])*.005
    s=state(3980)
    a,ta=evaluate_from_state(e,s,path(),2.6,params,0)
    b,tb=evaluate_from_state(e,s,path(),2.6,params,0)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    assert len(ta['time'])==20 and int(a.tick)==4000
    np.testing.assert_array_equal(ta['time'],tb['time'])
    changed=s.replace(lower=s.lower.replace(frames=s.lower.frames+1))
    c,_=evaluate_from_state(e,changed,path(),2.6,params,0)
    assert not np.array_equal(a.physical.controller.eso,c.physical.controller.eso)
    # Same old primitive exposes actual boundary states only in explicit mode.
    paths=jax.tree.map(lambda x:x[None],path());speeds=j.array([2.6]);states=jax.tree.map(lambda x:x[None],s)
    _,chunk=e._compiled_group(paths,speeds,s.alpha,params,20.,capture_states=True)
    final,(logs,history)=chunk(states,paths,speeds,params)
    assert history.tick.shape==(50,1) and int(history.tick[0,0])==3984
    assert int(final.tick[0])==4000 and int(logs['active_tick'].sum())==20


def test_oracle_keeps_prefixed_initial_violation_and_incomplete_horizon():
    prefix=trace(0,4,.31);tail=trace(4,3996);tail['peak_roll'][:20]=.31
    summary=recovery_oracle(tail,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=prefix)
    assert summary['initial_overlimit_duration_s']==pytest.approx(.12)
    short=trace(4,200)
    summary=recovery_oracle(short,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=prefix)
    assert not summary['qualified'] and not summary['full_original_end']
    short['lower_fault'][-1]=True
    with pytest.raises(ValueError,match='engineering'):recovery_oracle(short,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=prefix)


def test_pending_cli_never_constructs_environment_or_writes(tmp_path):
    import subprocess,sys
    cli=__import__('pathlib').Path(__file__).parents[1]/'cli/path_command_recovery.py'
    result=subprocess.run([sys.executable,str(cli),'--mode','capture','--output',str(tmp_path/'unused')],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['physical_ticks']==0 and not (tmp_path/'unused').exists()


def test_capture_serializes_first_natural_boundary_using_existing_chunk_interface(tmp_path):
    from pathlib import Path as DiskPath
    from sttw_control.path_command_recovery import capture_natural_snapshots
    cfg=json.loads((DiskPath(__file__).parents[1]/'configs/STTW_Path_Safety_candidate.json').read_text())
    class CaptureExecutor:
        config=cfg;snapshot=state().physical;charged=0
        env=SimpleNamespace(physics=SimpleNamespace(
            helpers=SimpleNamespace(pose=lambda data:j.zeros(3)),
            observe=lambda data:(j.array([data['qpos'][0],0.,0.,0.,0.,0.]),j.zeros(3),j.array(2.6))))
        check_budget=staticmethod(lambda:None)
        def account(self,n):self.charged+=n
        def _compiled_group(self,paths,speeds,alpha,params,horizon,*,capture_states=False):
            assert capture_states and horizon==20.
            def reset(*args):return jax.tree.map(lambda x:x[None],state(0))
            def chunk(states,*args):
                ticks=np.arange(int(states.tick[0])+4,int(states.tick[0])+204,4)
                phis=j.full((50,),.27).at[0].set(.29).at[1].set(-.31)
                history=jax.tree.map(lambda x:j.broadcast_to(x,(50,)+x.shape),states)
                history=history.replace(tick=j.asarray(ticks[:,None]),physical=history.physical.replace(
                    data={**history.physical.data,'qpos':history.physical.data['qpos'].at[:,0,0].set(phis)}))
                logs={k:j.asarray(v.reshape(50,1,4)) for k,v in trace(0,200).items()}
                logs['phi']=j.repeat(phis,4).reshape(50,1,4);logs['peak_roll']=abs(logs['phi'])
                return jax.tree.map(lambda x:x[-1],history),(logs,history)
            return reset,chunk
    e=CaptureExecutor();result=capture_natural_snapshots(e,{},tmp_path/'capture',identity())
    assert e.charged==400
    assert result['left90_R2']['start_tick']==result['right90_R2']['start_tick']==8
    saved,p,meta,prefix=load_snapshot(tmp_path/'capture/left90_R2',state(),identity())
    assert int(saved.tick)==8 and float(saved.physical.data['qpos'][0])==pytest.approx(-.31)
    assert len(prefix['time'])==8 and float(meta['boundary_phi_rad'])==pytest.approx(-.31)
    receipt=json.loads((tmp_path/'capture/capture.json').read_text())
    assert receipt['active_prefix_lower_ticks']==400 and receipt['completed']


def test_snapshot_rejects_second_qualifying_boundary(tmp_path):
    prefix=trace(0,8,.31)
    with pytest.raises(ValueError,match='first natural'):
        save_snapshot(tmp_path/'bad',state(8),path(),prefix,identity(),route='right90_R2',v_user=2.6,horizon=20.,boundary_phi=.31)


def test_recovery_qualification_does_not_add_primary_performance_gate():
    tail=trace(4,3996);tail['path_cross_track'][:2000]=.25
    report=recovery_oracle(tail,config(),start_tick=4,initial_phi=.31,route='left90_R2',alpha=0,prefix=takeover_prefix())
    assert report['qualified'] and not report['primary_band_pass']


def test_cpu_recovery_report_original_xy_all_arms_and_png_index(tmp_path):
    from pathlib import Path as DiskPath
    from sttw_control.path_command_recovery import generate_recovery_report
    cfg=json.loads((DiskPath(__file__).parents[1]/'configs/STTW_Path_Safety_candidate.json').read_text())
    prefix=trace(0,4,.31);tail=trace(4,400)
    def enrich(d):
        n=len(d['time']);d.update(actual_xy=np.stack([d['time']*2.6,np.zeros(n)],axis=1),
            phi_dot=np.zeros(n),offsets=np.zeros((n,2)),
            interval_offset_before=np.zeros((n,2)),interval_offset_end=np.zeros((n,2)),
            interval_start_tick=(np.rint(d['time']/.005).astype(int)//4)*4,
            interval_active_ticks=np.full(n,4),interval_failed=np.zeros(n,bool))
        return d
    enrich(prefix);enrich(tail)
    names=['B0','control0','candidate0','control1','candidate1']
    result=generate_recovery_report(tmp_path,{'left90_R2':dict(path=path(),prefix=prefix,
        traces={name:tail.copy() for name in names},configs={name:cfg for name in names},
        alphas={name:int(name[-1]) if name!='B0' else 0 for name in names},
        actor_sha256={name:'sha-'+name for name in names},snapshot_sha256='snapshot',takeover_s=.02)})
    assert 'left90_R2' in result
    assert (tmp_path/'left90_R2_xy.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    assert (tmp_path/'left90_R2_path.npz').exists() and (tmp_path/'left90_R2_reward_components.png').exists()
    text=(tmp_path/'INDEX.md').read_text();assert text.index('xy.png')<text.index('diagnostics.png')
    assert all(name in text for name in names) and 'snapshot' in text
