import numpy as np

def test_baseline_gate_checks_each_segment_and_failure():
    from sttw_control.teleop_metrics import baseline_metrics
    n=3200;t=(np.arange(n)+1)*.005
    data=dict(time=t,actual_forward_speed=np.full(n,2.3),actual_delta=np.zeros(n),
      limited_command=np.tile([2.3,0.],(n,1)),phi=np.zeros(n),peak_roll=np.zeros(n),physical_failure=np.zeros(n,bool))
    assert baseline_metrics(data,[[0,2.3,0]])['passed']
    data['actual_forward_speed'][-100:]=2.1
    assert not baseline_metrics(data,[[0,2.3,0]])['passed']
    data['actual_forward_speed'][:]=2.3;data['physical_failure'][-1]=True
    assert not baseline_metrics(data,[[0,2.3,0]])['passed']

def test_recovery_requires_unwrapped_debt_and_full_hold():
    from sttw_control.teleop_metrics import recovery_metrics
    n=3200;t=(np.arange(n)+1)*.005
    x=dict(time=t,limited_command=np.tile([2.6,0.],(n,1)),actual_forward_speed=np.full(n,2.6),actual_delta=np.zeros(n),
      e_psi_unwrapped=np.full(n,2*np.pi),physical_failure=np.zeros(n,bool))
    assert not recovery_metrics(x)['passed']
    x['e_psi_unwrapped'][:]=0
    assert recovery_metrics(x)['passed']
    x['e_psi_unwrapped'][:]=1;x['e_psi_unwrapped'][-99:]=0
    assert not recovery_metrics(x)['passed']

def test_governor_gate_rejects_nonzero_track_and_operating_violation():
    from sttw_control.teleop_metrics import governor_nominal_metrics
    n=20;x=dict(mode=np.zeros(n,int),raw_feasible=np.ones(n,bool),applied_residual=np.zeros((n,2)),
        physical_failure=np.zeros(n,bool),fallback=np.zeros(n,bool),peak_roll=np.zeros(n),time=np.linspace(.005,16,n))
    # A short log never counts as a complete gate.
    assert not governor_nominal_metrics(x)['passed']
    n=3200;x={k:np.repeat(v[:1],n,axis=0) for k,v in x.items()};x['time']=(np.arange(n)+1)*.005
    assert governor_nominal_metrics(x)['passed']
    x['applied_residual'][7,0]=.01
    assert not governor_nominal_metrics(x)['passed']

def test_random_recovery_uses_post_random_window_only():
    from sttw_control.teleop_metrics import random_gate
    n=3200;t=(np.arange(n)+1)*.005
    x=dict(time=t,physical_failure=np.zeros(n,bool),fallback=np.zeros(n,bool),peak_roll=np.zeros(n),
      e_psi_unwrapped=np.zeros(n),actual_forward_speed=np.full(n,2.3),actual_delta=np.zeros(n),limited_command=np.tile([2.3,0.],(n,1)))
    x['e_psi_unwrapped'][2000:]=1
    assert not random_gate(x)['passed']
    x['e_psi_unwrapped'][-100:]=0
    assert random_gate(x)['passed']
