import copy
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]

def test_config_rejects_unknown_and_invalid_alpha(tmp_path):
    from sttw_control.teleop_commands import load_teleop_config, validate_alpha
    c=load_teleop_config(ROOT/'learning/configs/teleop_pref_governor_v1.json')
    assert c['controller']['dt_s']==.005
    import json
    raw=copy.deepcopy(c);raw['commands']['yaw_slew']=1
    p=tmp_path/'bad.json';p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):load_teleop_config(p)
    for a in [float('nan'),float('inf'),.5,-1,2]:
        with pytest.raises(ValueError):validate_alpha(a)
    assert validate_alpha(0)==0 and validate_alpha(1)==1

def test_command_slew_10000_steps_and_random_reproducibility():
    from sttw_control.teleop_commands import limit_command, random_schedule
    c=np.array([2.3,0.]); rng=np.random.default_rng(55)
    for _ in range(10000):
        nxt=limit_command(c,rng.uniform([-2,-2],[5,2]))
        assert np.all(np.abs(nxt-c)<=[.0025+1e-14,.0015+1e-14])
        assert 2<=nxt[0]<=2.6 and abs(nxt[1])<=.3
        c=nxt
    a=random_schedule(65001);b=random_schedule(65001)
    np.testing.assert_array_equal(a,b)
    assert a[-1].tolist()==[10.,2.3,0.]
    g=np.random.Generator(np.random.PCG64(65001))
    np.testing.assert_equal(a[1,1:],[g.uniform(2,2.6),g.uniform(-.3,.3)])
    assert a[2,0]==2+g.uniform(.8,1.6)

def test_reference_exact_and_unwrapped():
    from sttw_control.teleop_reference import integrate, unwrap
    import jax.numpy as jp
    p=integrate(jp.array([0.,0.,0.]),jp.array([2.3,0.]),1.)
    np.testing.assert_allclose(p,[2.3,0,0],atol=1e-6)
    d=.2;v=2.3;w=v*np.cos(np.deg2rad(25))*np.tan(d)/.408
    p=integrate(jp.zeros(3),jp.array([v,d]),2.)
    np.testing.assert_allclose(p,[v*np.sin(2*w)/w,v*(1-np.cos(2*w))/w,2*w],rtol=1e-6)
    n=integrate(jp.zeros(3),jp.array([v,-d]),2.)
    np.testing.assert_allclose(n,np.asarray(p)*[1,-1,-1],atol=1e-6)
    np.testing.assert_allclose(integrate(jp.zeros(3),jp.array([v,1e-10]),1.),[v,0,0],atol=1e-6)
    yaw=0.;old=0.
    for x in np.linspace(0,4*np.pi,1001)[1:]:
        wrapped=np.arctan2(np.sin(x),np.cos(x));yaw=unwrap(yaw,old,wrapped);old=wrapped
    assert float(yaw)>12.56 and abs(old)<1e-10  # full turns remain debt, not wrapped zero

def test_nominal_goal_preview_single_eso_and_authority():
    import jax
    import jax.numpy as jp
    from sttw_control.closed_loop_kernel import controls
    from sttw_control.controller import initial_controller,ControllerConfig,controller_step
    from sttw_control.actuator import initial_actuator,ActuatorConfig,apply_residual
    cc=ControllerConfig();ac=ActuatorConfig(steer_residual_scale=1.5,rear_residual_scale=10.)
    old=initial_controller(cc); act=initial_actuator(ac,23.)
    m=jp.array([.05,.03,.04,.02,0.,23.,23.]);raw=jp.array([2.3,.08])
    c1,a1,log=controls(old,act,m,raw,raw,True,False,cc,ac)
    c2,a2,lb=controls(old,act,m,raw,jp.array([1.5,-.35]),True,True,cc,ac)
    for l,r in zip(jax.tree.leaves((c1,a1)),jax.tree.leaves((c2,a2))):np.testing.assert_array_equal(l,r)
    np.testing.assert_array_equal(log['applied_residual'],[0,0])
    expected,_=controller_step(old,jp.array([2.3,.04,.02,.05,.03,.08]),True,cc)
    for l,r in zip(jax.tree.leaves(c1),jax.tree.leaves(expected)):np.testing.assert_array_equal(l,r)
    for goal in [jp.array([1.5,-.35]),jp.array([2.6,.35])]:
        c,a,l=controls(old,act,m,raw,goal,True,False,cc,ac)
        assert np.all(np.abs(l['applied_residual'])<=[1.5,10.])
        assert np.all(np.abs(a.previous)<=[3,60])
        for x,y in zip(jax.tree.leaves(c),jax.tree.leaves(expected)):np.testing.assert_array_equal(x,y)

def test_snapshot_no_future_and_cpu_kernel_pure_zero_equivalence():
    from sttw_control.teleop_env import TeleopEnv
    from sttw_control.governor_state import Snapshot
    import jax
    import jax.numpy as jp
    env=TeleopEnv(backend='cpu');s=env.initial()
    q=np.array(s.data.qpos);eso=np.array(s.controller.eso)
    assert not any('schedule' in x or 'future' in x for x in Snapshot.__dataclass_fields__)
    a,la=env.step(s,s.raw,False);b,lb=env.step(s,jp.array([1.5,.35]),True)
    # governed starts equal to raw; goal=raw remains exactly zero.
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    np.testing.assert_array_equal(s.data.qpos,q);np.testing.assert_array_equal(s.controller.eso,eso)
    assert int(a.physical_tick)==1
    assert float(a.data.time)==pytest.approx(.005)
    assert float(la['actual_forward_speed'])>0
    assert float(la['final_ctrl'][1])<0

def test_endpoint_limiter_retains_original_actuator_semantics():
    import jax.numpy as jp
    from sttw_control.closed_loop_kernel import controls
    from sttw_control.controller import initial_controller,ControllerConfig
    from sttw_control.actuator import initial_actuator,ActuatorConfig
    cc=ControllerConfig();ac=ActuatorConfig(steer_residual_scale=1.5,rear_residual_scale=10.)
    for steer in [-.8,.8]:
        m=jp.array([0.,0.,steer,0.,0.,23.,23.])
        for goal in [jp.array([1.5,-.35]),jp.array([2.6,.35])]:
            _,a,l=controls(initial_controller(cc),initial_actuator(ac,23.),m,jp.array([2.3,0.]),goal,True,False,cc,ac)
            assert steer*float(a.previous[0])<=0
            assert abs(float(l['applied_residual'][0]))<=1.5
            assert abs(float(l['applied_residual'][1]))<=10

def test_dry_run_does_not_create_output_or_simulate(tmp_path):
    import subprocess,sys,os
    out=tmp_path/'absent'
    result=subprocess.run([sys.executable,str(ROOT/'learning/cli/teleop_governor_review.py'),'--config',str(ROOT/'learning/configs/teleop_pref_governor_v1.json'),'--output',str(out),'--dry-run'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not out.exists()

def test_copy_isolation_is_separate_from_batched_numeric_variation():
    from sttw_control.teleop_contract_checks import copy_diagnostics
    before=[np.array([1.,2.])];after=[np.array([1.,2.])]
    report=copy_diagnostics(before,after,{'terminal_qvel':np.array([[1.],[1.00009]]),'feasible':np.array([True,True])})
    assert report['state_preserved']
    assert report['per_field_duplicate_max_abs']['terminal_qvel']>1e-5
    after[0][1]=3.
    assert not copy_diagnostics(before,after,{})['state_preserved']

def test_initial_state_respects_selected_jax_precision():
    # Higher arithmetic precision must not leave float32 state inside a float64 scan.
    import jax
    from sttw_control.teleop_env import TeleopEnv
    with jax.experimental.enable_x64():
        e=TeleopEnv(backend='cpu');s=e.initial()
        assert np.asarray(s.raw).dtype==np.float64
        np.testing.assert_array_equal(s.data.qpos,e.model.qpos0)

def test_precision_migration_preserves_complete_snapshot_values():
    import jax
    from sttw_control.teleop_env import TeleopEnv
    from sttw_control.teleop_precision import promote_snapshot
    e=TeleopEnv(backend='mjx');s=e.initial()
    with jax.experimental.enable_x64():
        migrated=promote_snapshot(s)
        for x,y in zip(jax.tree.leaves(s),jax.tree.leaves(migrated)):
            np.testing.assert_array_equal(x,y)
        assert migrated.data.qpos.dtype==np.float64
        assert migrated.data._impl.contact.geom.dtype==np.int64
        assert migrated.governor.recovery_gain.dtype==np.float64

def test_independent_summary_rebuilds_actual_costs():
    from sttw_control.teleop_contract_checks import independent_summary
    from types import SimpleNamespace
    from sttw_control.controller import ControllerConfig
    n=240
    initial=SimpleNamespace(raw=np.array([2.3,0.]),actuator=SimpleNamespace(previous=np.array([0.,23.])))
    end=SimpleNamespace(failed=False,data=SimpleNamespace(qpos=np.zeros(11),qvel=np.zeros(10)))
    logs={k:np.zeros(n) for k in ['phi','phi_dot','actual_delta','peak_roll','peak_roll_rate','e_psi_unwrapped','nonfinite']}
    logs.update(wheel_speed_proxy=np.full(n,2.3),actual_forward_speed=np.full(n,2.4),final_command=np.tile([0.,23.],(n,1)),applied_residual=np.zeros((n,2)))
    r=independent_summary(initial,end,logs,SimpleNamespace(cc=ControllerConfig()))
    assert abs(r['speed_rmse']-.1)<1e-14 and r['steer_rmse']==0 and r['smoothness']==0 and r['feasible']
