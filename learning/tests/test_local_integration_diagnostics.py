import importlib
import numpy as np

def api():
    try:return importlib.import_module('sttw_control.local_integration_diagnostics')
    except ModuleNotFoundError:assert False,'phase-consistent diagnostic API not implemented'

def test_small_tolerance_applies_only_after_published_plateau_settles():
    m=api();t=np.arange(2000)*.005;cmd=np.tile([2.3,.2],(2000,1));cmd[1500:,1]=.04
    d=dict(time=t,limited_command=cmd,raw_rates=np.zeros((2000,2)),actual_forward_speed=cmd[:,0],actual_delta=cmd[:,1]+.01,peak_roll=np.zeros(2000))
    mask=m.small_window(d,'post_return_small_positive',{'post_small_target_start_s':7.5,'post_small_target':.04,'settle_s':.5})
    assert np.flatnonzero(mask)[0]==1600
    q=m.phase_score(d,mask)
    assert abs(q-(1600*.25+400*1)/4000)<1e-10

def test_short_platform_has_no_worst_score_and_settle_is_command_based():
    m=api();t=np.arange(300)*.005;cmd=np.tile([2.3,0.],(300,1));cmd[160:,0]=2.5
    rates=np.zeros((300,2));rates[160,0]=40
    d=dict(time=t,limited_command=cmd,raw_rates=rates,actual_forward_speed=cmd[:,0]+.2,actual_delta=cmd[:,1],peak_roll=np.zeros(300))
    rows=m.platform_statistics(d,np.zeros(300,bool))
    assert all(r['worst_normalized_rmse'] is None for r in rows)
    assert rows[0]['settled']['speed']['bias']==np.mean(np.full(60,.2)) or abs(rows[0]['settled']['speed']['bias']-.2)<1e-12

def test_yaw_split_preserves_initial_error_and_exact_velocity_steer_identity():
    m=api();n=20;dt=.005;cmd=np.tile([2.3,.1],(n,1));v=np.full(n,2.2);delta=np.full(n,.08)
    yaw=np.arange(1,n+1)*dt*.4;ref=.07+np.arange(1,n+1)*dt*cmd[0,0]*m.kappa(cmd[0,1])
    d=dict(time=np.arange(n)*dt,limited_command=cmd,actual_forward_speed=v,actual_delta=delta,yaw_unwrapped=yaw,reference_yaw_unwrapped=ref,e_psi_unwrapped=ref-yaw)
    result,arrays=m.yaw_decomposition(d,initial_yaw=0.,initial_error=.07)
    np.testing.assert_allclose(arrays['A']+arrays['B']+arrays['C'],cmd[:,0]*m.kappa(cmd[:,1])-.4,atol=1e-13)
    assert result['closure_abs_error']<1e-12

def test_platform_full_exceedance_also_uses_phase_specific_tolerance():
    m=api();t=7.5+np.arange(400)*.005;cmd=np.tile([2.3,.04],(400,1))
    d=dict(time=t,limited_command=cmd,raw_rates=np.zeros((400,2)),actual_forward_speed=cmd[:,0],actual_delta=cmd[:,1]+.015,peak_roll=np.zeros(400))
    row=m.platform_statistics(d,t>=8.)[0]
    assert abs(row['full']['steer']['longest_exceed_s']-1.5)<1e-12
