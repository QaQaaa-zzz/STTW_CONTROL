"""Native four-tick scheduling plus explicit continuation permissions.

The archived native policy_step and correction filter execute against a tiny
deterministic plant double. No MuJoCo transitions or controller gains are fit.
"""
from dataclasses import fields
import importlib
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import json

import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import struct

from sttw_control.contact_only_environment import ContactOnlyPhysics
from sttw_control.native_goal_continuation import NativeGoalContinuation,target_permission


@struct.dataclass
class Physical:
    failed: object
    data: object


@struct.dataclass
class Follower:
    invalid: object
    numerical_fault: object


@struct.dataclass
class Projection:
    progress: object


@struct.dataclass
class State:
    physical: object
    follower: object
    filtered: object
    offsets: object
    tick: object
    fault: object
    domain_exit: object
    history: object
    projection: object
    alpha: object


@pytest.fixture(scope='module')
def native():
    root=Path(__file__).resolve().parents[2]
    archived=root/'runs/current_native_preference_v1_attempt2/native_runtime/learning/src/sttw_control'
    if not archived.is_dir():pytest.skip('archived native source is local integration evidence')
    # Explicit alias for tests only; production imports must use its sealed
    # native package in a fresh process. Never add modern files to native paths.
    name='_sttw_archived_native_contract_test'
    package=ModuleType(name);package.__path__=[str(archived)]
    sys.modules[name]=package
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:module=importlib.import_module(name+'.path_command_env')
    finally:sys.dont_write_bytecode=old
    assert Path(module.__file__).resolve().is_relative_to(archived.resolve())
    cfg=json.loads((root/'runs/native_zero_preference_v1/plan.json').read_text())['config']
    return module,cfg


def fixture_env(native,*,invalid=(),contact=(),nonfinite=(),geometry_fault=(),roll=.8,observation_fault=False):
    module,cfg=native
    env=module.PathCommandEnv.__new__(module.PathCommandEnv)
    env.config=cfg;env.terminal_contract='terminal_contract_v2'
    env.cc=SimpleNamespace(dt=cfg['timing']['lower_dt_s'])
    def sequence(t,values):
        return j.any(j.asarray(t)==j.asarray(values,dtype=j.int32))
    def physical_tick(p,*args,**kwargs):
        t=p.data
        hit=sequence(t,contact);bad=sequence(t,nonfinite)
        end=p.replace(data=t+1,failed=p.failed|hit|bad|(roll>.7))
        return end,dict(contact_failure=hit,nonfinite=bad,peak_roll=j.asarray(roll),physical_failure=end.failed)
    env.physics=SimpleNamespace(
        _step=physical_tick,
        helpers=SimpleNamespace(_contact_failure=lambda data:j.bool_(False)),
        observe=lambda data:(j.zeros(6),j.array([data,2.,.2]),j.float32(2.3)))
    def tick(st,z,path,v_user,force_pulse=None,horizon=None):
        p,log=env.physics._step(st.physical,force_pulse=force_pulse)
        bad=sequence(st.tick,invalid)
        filtered,offsets,governed,flags=module.correction_tick(st.filtered,st.offsets,z,j.array([2.6,.1]),cfg)
        log.update(active_tick=j.bool_(True),
                   raw_components=dict(offset_rate=j.float32(0)),
                   effective_components=dict(offset_rate=j.float32(0)),
                   component_capped=dict(offset_rate=j.bool_(False)),tick_reward=j.float32(-.01))
        # Match the actual native command/invalid cadence: reuse the previous
        # follower for three ticks, recompute only at the upper boundary.
        follower=jax.lax.cond((st.tick+1)%4==0,
                              lambda _:Follower(bad,j.bool_(False)),lambda _:st.follower,None)
        log['domain_exit']=follower.invalid
        end=st.replace(physical=p,follower=follower,
                       tick=st.tick+1,domain_exit=st.domain_exit|follower.invalid,
                       filtered=filtered,offsets=offsets)
        return end,log
    env.tick=tick
    env.record_frame=lambda s:s.replace(history=s.history+1)
    env.observation=lambda *args:(j.zeros(351),j.zeros(352),j.bool_(observation_fault))
    project=lambda path,xy,previous,distance,config:Projection(j.float32(0))
    def pursuit(path,progress,pose,speed,v_user,config):
        # Data advances after the physical tick; map post-pose back to the
        # actual tick schedule. Explicit replacement geometry is valid here.
        invalid_now=sequence(pose[0]-1,invalid) if path is None else j.bool_(False)
        numerical=sequence(pose[0]-1,geometry_fault) if path is None else j.bool_(False)
        return Follower(invalid_now,numerical)
    state=State(Physical(j.bool_(False),j.int32(0)),Follower(j.bool_(False),j.bool_(False)),
                j.zeros(2),j.zeros(2),j.int32(0),j.bool_(False),j.bool_(False),j.int32(7),Projection(j.float32(99)),j.float32(0))
    continuation=NativeGoalContinuation(env,contact_only_physics=ContactOnlyPhysics,
                                         project=project,pursuit=pursuit)
    # Uses actual archived set_log_template -> adapted tick, and actual native
    # policy_step's stop/scan/reward logic, rather than a reimplemented scan.
    env.set_log_template(state,None,j.float32(2.6))
    return continuation,continuation.initial(state)


def run_block(env,carry,*,target=None,zero_mode=False):
    if target is None:target=lambda obs,s:j.ones(2)
    return env.policy_step(carry,None,j.float32(2.6),20.,target,zero_mode=zero_mode)


def test_block_internal_domain_exit_does_not_freeze_the_remaining_real_ticks(native):
    env,carry=fixture_env(native,invalid=(1,))
    end,logs=jax.jit(lambda c:run_block(env,c))(carry)
    assert int(end.state.tick)==4 and int(end.state.physical.data)==4
    assert bool(end.ever_domain_exit) and not bool(end.state.domain_exit)
    assert int(end.clean_blocks)==0
    # Native follower remains valid at its 50Hz query; the read-only 200Hz
    # telemetry still catches the brief, otherwise invisible, invalidity.
    np.testing.assert_array_equal(logs['continuation_geometry_invalid'],[False,True,False,False])
    assert not np.any(logs['domain_exit'])
    np.testing.assert_array_equal(logs['cumulative_domain_exit'],[False,True,True,True])
    # The already published target remains held across the native block.
    assert np.all(logs['reference_target_permitted'])
    _,nextlogs=run_block(env,end)
    assert not np.any(nextlogs['reference_target_permitted'])
    assert int(end.state.history)==8


def test_sixteen_complete_clean_blocks_then_next_publication_reopens(native):
    env,carry=fixture_env(native,invalid=(1,))
    carry,_=run_block(env,carry)
    fn=jax.jit(lambda c:run_block(env,c))
    for i in range(16):
        carry,logs=fn(carry)
        assert not np.any(logs['reference_target_permitted'])
        assert int(carry.clean_blocks)==i+1
    assert bool(target_permission(carry))
    carry,logs=fn(carry)
    assert np.all(logs['reference_target_permitted']) and int(carry.clean_blocks)==16


def test_new_invalid_tick_interrupts_consecutive_clean_count(native):
    env,carry=fixture_env(native,invalid=(1,9))
    carry,_=run_block(env,carry)
    carry,_=run_block(env,carry)
    assert int(carry.clean_blocks)==1
    carry,logs=run_block(env,carry)
    assert int(carry.clean_blocks)==0 and bool(carry.ever_domain_exit)
    assert np.count_nonzero(logs['active_tick'])==4


@pytest.mark.parametrize('contact,nonfinite,expected_ticks',[((),(),4),((1,),(),2),((),(1,),2)])
def test_roll_screen_is_removed_but_same_tick_contact_and_fault_still_stop(native,contact,nonfinite,expected_ticks):
    env,carry=fixture_env(native,contact=contact,nonfinite=nonfinite,roll=.8)
    end,logs=jax.jit(lambda c:run_block(env,c))(carry)
    assert int(end.state.physical.data)==expected_ticks
    assert np.count_nonzero(logs['active_tick'])==expected_ticks
    assert bool(end.state.physical.failed)==(expected_ticks<4)
    if expected_ticks<4:
        after,logs2=run_block(env,end)
        assert int(after.state.physical.data)==expected_ticks
        assert not np.any(logs2['active_tick'])


def test_masked_target_keeps_filter_tail_instead_of_resetting_state(native):
    env,carry=fixture_env(native,invalid=(1,))
    carry,_=run_block(env,carry,target=lambda obs,s:j.ones(2)*3)
    before=np.asarray(carry.state.filtered)
    assert np.all(before>0)
    end,logs=run_block(env,carry,target=lambda obs,s:j.full(2,j.nan))
    assert not np.any(logs['reference_target_permitted'])
    # Blocked runtime branch did not execute its NaN output. Original filter
    # and slew still carry residual offsets; no hidden hard-zero command.
    assert not bool(end.state.fault)
    assert np.all(np.asarray(end.state.filtered)>0)
    assert np.all(np.asarray(end.state.filtered)<before)
    assert np.any(np.asarray(end.state.offsets)!=0)


def test_observation_fault_is_not_ignored_while_target_is_masked(native):
    env,carry=fixture_env(native,observation_fault=True)
    end,logs=run_block(env,carry,zero_mode=True)
    assert bool(end.state.fault) and not np.any(logs['active_tick'])
    assert int(end.state.physical.data)==0


def test_readonly_geometry_fault_preserves_actual_tick_then_stops_scan(native):
    env,carry=fixture_env(native,geometry_fault=(0,))
    end,logs=jax.jit(lambda c:run_block(env,c))(carry)
    assert bool(end.state.fault) and not bool(end.state.physical.failed)
    assert int(end.state.physical.data)==1
    np.testing.assert_array_equal(logs['active_tick'],[True,False,False,False])
    np.testing.assert_array_equal(logs['continuation_geometry_fault'],[True,False,False,False])
    assert logs['continuation_geometry_invalid'].shape==(4,)
    assert int(end.clean_blocks)==0
    after,tail=run_block(env,end)
    assert int(after.state.physical.data)==1 and not np.any(tail['active_tick'])


def test_rebind_only_at_complete_boundary_preserves_state_and_gate_history(native):
    env,carry=fixture_env(native,invalid=(1,))
    carry,_=run_block(env,carry)
    rebound=env.rebind_at_upper_boundary(carry,'new geometry',j.float32(2.6))
    assert bool(rebound.ever_domain_exit) and int(rebound.clean_blocks)==0
    assert not bool(target_permission(rebound))
    for f in fields(State):
        if f.name not in ('projection','follower','fault'):
            assert getattr(rebound.state,f.name) is getattr(carry.state,f.name),f.name
    assert bool(rebound.state.fault)==bool(carry.state.fault)
    broken=carry.replace(state=carry.state.replace(tick=j.int32(5)))
    with pytest.raises(ValueError,match='upper-block boundary'):
        env.rebind_at_upper_boundary(broken,'new geometry',2.6)


def test_zero_mode_remains_zero_but_updates_actual_history_and_clean_records(native):
    env,carry=fixture_env(native)
    end,logs=run_block(env,carry,zero_mode=True,target=lambda obs,s:j.full(2,j.nan))
    assert not np.any(logs['reference_target_permitted'])
    assert int(end.state.tick)==4 and int(end.state.history)==8
    np.testing.assert_array_equal(end.state.filtered,j.zeros(2))
    assert int(end.clean_blocks)==1 and not bool(end.state.fault)
