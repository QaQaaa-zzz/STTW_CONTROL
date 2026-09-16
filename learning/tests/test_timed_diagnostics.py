import copy
import numpy as np
import pytest
from sttw_control import tracking_diagnostics as diagnostics


def reference_fixture():
    dt=.1
    command=np.array([[2.,.4],[1.,-.2],[3.,.1]])
    pose=[np.array([3.,-2.,.3])]
    for speed,yaw in command[:-1]:
        x,y,heading=pose[-1];angle=yaw*dt
        distance=speed*dt*np.sinc(angle/(2*np.pi))
        pose.append(np.array([x+distance*np.cos(heading+angle/2),y+distance*np.sin(heading+angle/2),heading+angle]))
    pose=np.asarray(pose)
    return dict(time=np.arange(3)*dt,reference_pose=pose,reference_command=command,pose=pose.copy(),
                longitudinal_error=np.zeros(3),path_features=np.column_stack([np.zeros(3),np.zeros(3),command[:,1]/command[:,0]]),
                yaw_rate_world=np.r_[command[0,1],command[:-1,1]],yaw_rate_error=np.zeros(3))


def test_timed_reference_audit_independent_of_vehicle_motion():
    trace=reference_fixture()
    diagnostics.audit_timed_reference(trace,.1)
    bad=copy.deepcopy(trace);bad['reference_pose'][2,0]+=.2
    with pytest.raises(ValueError,match='reference'):
        diagnostics.audit_timed_reference(bad,.1)
    bad=copy.deepcopy(trace);bad['pose'][2,0]+=.2
    with pytest.raises(ValueError,match='error'):
        diagnostics.audit_timed_reference(bad,.1)


def test_timed_speed_uses_pre_step_command_not_steering_speed_proxy():
    trace=reference_fixture();trace['true_forward_speed']=np.array([2.,2.5,.5]);trace['motion_command']=np.zeros((3,2))
    np.testing.assert_allclose(diagnostics.speed_error(trace,{'timed_reference':{}}),[.5,-.5])
    np.testing.assert_allclose(diagnostics.reference_xy(trace,{'timed_reference':{}}),trace['reference_pose'][:,:2])


def save_timed_fixture(path):
    import json
    from dataclasses import asdict
    from sttw_control.tracking_reward import TrackingConfig,initial_return,return_observation,transition
    path.mkdir(parents=True)
    trace=reference_fixture();n=len(trace['time']);c=TrackingConfig(timed=True,start_seconds=0.)
    config=dict(tracking=asdict(c),timed_reference={},controller={'dt':.1},horizon_seconds=.2,alive_reward_rate=1.,failure_penalty=100.)
    trace.update(measurement=np.zeros((n,7)),effective_action=np.zeros((n,2)),motion_command=np.zeros((n,2)),priority_alpha=np.ones(n)*.5,
                 true_forward_speed=np.r_[2.,trace['reference_command'][:-1,0]],terminated=np.zeros(n,bool),end_code=np.zeros(n,int),event=np.zeros((n,6)),
                 qpos=np.column_stack([trace['pose'][:,:2],np.zeros(n),np.ones(n),np.zeros((n,3))]),qvel=np.zeros((n,6)),command=np.zeros((n,2)),reference_roll=np.zeros(n))
    state=initial_return(xp=np);states=[return_observation(state,c,xp=np)];rewards=[0.];parts={}
    for i in range(1,n):
        state,terms=transition(state,roll=0.,roll_rate=0.,speed_error=0.,lateral_error=0.,heading_error=0.,longitudinal_error=0.,yaw_rate_error=0.,action=np.zeros(2),alpha=.5,dt=.1,alive_rate=1.,failure_penalty=100.,failed=False,enabled=True,config=c,xp=np)
        states.append(return_observation(state,c,xp=np));rewards.append(sum(terms.values()))
        for k,v in terms.items():parts.setdefault('reward_'+k,[0.]).append(float(v))
    trace.update(return_state=np.array(states),reward=np.array(rewards),**{k:np.array(v) for k,v in parts.items()})
    np.savez_compressed(path/'trace.npz',**trace)
    (path/'declaration.json').write_text(json.dumps({'config':config}))
    return trace,config


def test_timed_audit_and_alpha_plots_without_physics(tmp_path):
    import json
    for i,alpha in enumerate([0.,.5,1.]):
        panel=tmp_path/f'evaluation/alpha_{i}/seed_7';panel.mkdir(parents=True)
        (panel/'declaration.json').write_text(json.dumps(dict(priority_alpha_override=alpha,panel={'seed':7},scenarios=['random'],checkpoint='fixed-best')))
        for policy in ['baseline','residual']:save_timed_fixture(panel/'random'/policy)
    path=tmp_path/'evaluation/alpha_0/seed_7/random/residual'
    audited=diagnostics.audit_trace(path)
    assert audited['max_reference_error']<1e-10
    assert audited['max_reward_error']<1e-10
    diagnostics.write_diagnostics(path,baseline=path.parent/'baseline')
    for stem in ['trajectory','longitudinal_error','xy_error','yaw_rate','yaw_rate_error','step_reward','cumulative_reward']:
        for ext in ['png','pdf']:assert (path/f'analysis/tracking/{stem}.{ext}').exists()
    with np.load(path/'analysis/tracking/tracking_errors.npz') as errors:
        assert errors['residual_xy_error'].shape==(2,)
        assert errors['baseline_reference_command'].shape==(2,2)
    output=diagnostics.write_alpha_error_overview(tmp_path)
    with np.load(output/'random_seed_7.npz') as arrays:
        for alpha in [0.,.5,1.]:
            for policy in ['baseline','residual']:
                for key in ['speed_error','lateral_error','longitudinal_error','yaw_rate_error']:
                    assert len(arrays[f'{policy}_alpha_{alpha}_{key}'])==2


def test_media_uses_timed_reference_without_static_path(tmp_path):
    from sttw_control.media import plot_states
    trace,config=save_timed_fixture(tmp_path/'input')
    out=tmp_path/'media';out.mkdir()
    result=plot_states(trace,config,out)
    assert result['right_error_rmse_m']==0.
    assert (out/'trajectory.png').exists()
