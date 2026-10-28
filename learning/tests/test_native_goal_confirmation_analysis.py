"""CPU offline response readback: masks, force frames, horizons and pairing."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

CLI=Path(__file__).parents[1]/'cli'
sys.path.insert(0,str(CLI))
spec=importlib.util.spec_from_file_location('goal_confirmation_analysis',CLI/'analyze_native_goal_confirmation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def channel_raw():
    n=8
    case=dict(mode='hybrid',speed_alpha=1.,steer_alpha=.5,alpha=1.)
    raw=dict(active_tick=np.ones(n,bool),queried_latent_alpha_half=np.tile([.2,.3],(n,1)),
        queried_latent_alpha_one=np.tile([.7,.8],(n,1)),composed_latent=np.tile([.7,.3],(n,1)),
        channel_alpha_pair=np.tile([1.,.5],(n,1)),latent_z=np.vstack([np.tile([.7,.3],(4,1)),np.zeros((4,2))]),
        reference_target_permitted=np.r_[np.ones(4,bool),np.zeros(4,bool)],query_observation=np.zeros((n,351)),pre_upper_yaw=np.zeros(n))
    raw['query_observation'][:,336]=1.
    return raw,case


def test_hybrid_composition_applied_only_under_actual_mask():
    raw,case=channel_raw()
    out=m.audit_response_channels(raw,case)
    assert out['masked_ticks']==4 and out['actual_permitted_ticks']==4
    raw['latent_z'][5]=[.7,.3]
    with pytest.raises(AssertionError):m.audit_response_channels(raw,case)


def test_wrong_channel_or_within_upper_query_change_rejected():
    raw,case=channel_raw();raw['composed_latent'][0,0]=.2
    with pytest.raises(AssertionError):m.audit_response_channels(raw,case)
    raw,case=channel_raw();raw['pre_upper_yaw'][1]=.1
    with pytest.raises(AssertionError):m.audit_response_channels(raw,case)


def test_onset_frame_sign_and_truncated_exposure():
    raw=dict(active_tick=np.ones(2000,bool),pre_upper_yaw=np.full(2000,np.pi),applied_force_impulse=np.zeros((2000,3)))
    raw['applied_force_impulse'][320:340,1]=.16
    case=dict(onset_s=1.6,duration_s=.1,force_N=32.)
    pulse=dict(actual_world_impulse=[0.,3.2,0.])
    result=m.onset_frame_impulse(raw,case,pulse)
    assert result['exposure']=='full_event_dose'
    assert np.isclose(result['signed_body_lateral_impulse_onset_frame_N_s'],-3.2)
    raw['active_tick'][330:]=False
    result=m.onset_frame_impulse(raw,case,dict(actual_world_impulse=[0.,1.6,0.]))
    assert result['exposure']=='truncated_exposure'
    assert result['actual_exposed_ticks']==10
    raw['active_tick'][300:]=False
    result=m.onset_frame_impulse(raw,case,dict(actual_world_impulse=[0.,0.,0.]))
    assert result['onset_pre_upper_yaw_rad'] is None


def test_summary_pairs_route_event_with_intentionally_different_state_alphas():
    rows=[]
    for route in ('left','right'):
        for event,force in (('zero',0.),('pulse',32.)):
            for method,alpha in (('fixed0.5',.5),('fixed1',1.),('speed1_steer0.5',1.)):
                rows.append(dict(route=route,event=event,force_N=force,method=method,alpha=alpha,
                    case_id=route+event+method,replanned=(method=='fixed1'),diagnostic_execution_path_user1_cost=1.,
                    original_route_sample_distance_rmse_m=.1,full_observed_speed_rmse_m_s=.1,pulse_started_after_first_arrival=False,
                    onset_frame=dict(exposure='full_event_dose'),
                    horizons={'10':dict(goal_arrived=method!='fixed1',physical_outcome='survived')}))
    summary,paired=m.response_summary(rows)
    assert summary['fixed0.5']['nominal']['n']==2
    assert summary['fixed1']['nominal']['nominal_replans']==2
    assert len(paired)==3 and all(len(p['details'])==4 for p in paired)
    assert paired[0]['details'][0]['candidate_state_alpha']==.5
    assert paired[0]['details'][0]['baseline_state_alpha']==1.
    assert paired[0]['details'][0]['arrival_change']==1


def reward_fixture():
    n=8;keys=['speed','path','primary','roll']
    raw=dict(active_tick=np.ones(n,bool),time=np.arange(n)*.005,path_progress=np.ones(n),
        actual_forward_speed=np.full(n,1.15),v_user=np.ones(n),path_cross_track=np.full(n,.2),
        path_heading_error=np.zeros(n),effective_costs=np.zeros((n,4)),physical_failure=np.zeros(n,bool),domain_exit=np.ones(n,bool))
    path=dict(s=np.arange(4.),curvature=np.zeros(4))
    config=dict(timing=dict(lower_dt_s=.005,upper_dt_s=.02),training_future_phase_C=dict(episode_s=10.,gamma=.99),
        path_reward=dict(reward_version='path_persistent_preference_v3',component_caps=dict(speed=100.,path=100.,primary=100.),speed_scale_m_s=.1,
        cross_track_scale_m=.1,path_heading_factor=1.,path_heading_scale_rad=.1,primary_weight=4.,primary_path_band_m=.1,primary_path_scale_m=.1,
        primary_speed_band_m_s=.1,primary_speed_scale_m_s=.1,path_secondary_ratio=.1,scale=1.,failure_extra=2.,failure_cost_rate_bound=5.))
    controller=dict(minimum_speed=.1,mass=10.,cg_height=.2,cg_forward=.1,trail=.02,gravity=9.81,caster=.4,wheelbase=.4)
    from analyze_native_frozen_validation import score
    view=dict(raw,domain_exit=np.zeros(n,bool));_,data=score(view,path,config,controller,keys,1.)
    raw['effective_costs']=data['effective'];raw['scored_tick_reward']=-data['scored']
    return raw,path,config,controller,keys


def test_execution_path_reward_reconstruction_does_not_treat_domain_as_contact():
    raw,path,cfg,ctrl,keys=reward_fixture()
    result=m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,8,cfg,ctrl,keys)
    np.testing.assert_allclose(result['reward'],raw['scored_tick_reward'])
    assert np.allclose(result['terminal_adjustment'],0.)
    raw['scored_tick_reward'][3]+=.1
    with pytest.raises(AssertionError,match='per-step reward'):
        m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,8,cfg,ctrl,keys)


def test_execution_path_terminal_replaces_only_actual_failure_upper_block():
    raw,path,cfg,ctrl,keys=reward_fixture();raw['physical_failure'][-1]=True
    remaining=500-7//4;c=cfg['path_reward'];g=cfg['training_future_phase_C']['gamma']
    raw['scored_tick_reward'][4:]=0.
    raw['scored_tick_reward'][7]=-c['failure_extra']-c['scale']*.02*c['failure_cost_rate_bound']*(1-g**remaining)/(1-g)
    result=m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,8,cfg,ctrl,keys)
    np.testing.assert_allclose(result['reward'],raw['scored_tick_reward'])
    assert result['terminal_adjustment'][-1]<0


def test_config_driven_response_inventory_rejects_missing_pair_and_event_drift():
    protocol=dict(zero_onset_s=1.6,forces_N=[-32.,32.],onsets_s=[1.6,4.],duration_s=.1,
                  routes=[dict(route='routeA',speed=2.8)],mappings=[[.5,.5],[1.,1.],[1.,.5]])
    inventory=[]
    for pair in protocol['mappings']:
        for force,onset in [(0.,1.6)]+list(m.itertools.product(protocol['forces_N'],protocol['onsets_s'])):
            inventory.append(dict(route='routeA',speed=2.8,speed_alpha=pair[0],steer_alpha=pair[1],
                alpha=pair[0],mode='hybrid' if pair==[1.,.5] else 'H75',force_N=force,onset_s=onset,duration_s=.1))
    m.validate_response_inventory(inventory,protocol)
    with pytest.raises(ValueError,match='inventory differs'):m.validate_response_inventory(inventory[:-1],protocol)
    inventory[-1]['onset_s']=4.1
    with pytest.raises(ValueError,match='inventory differs'):m.validate_response_inventory(inventory,protocol)


def test_user1_rescoring_is_separate_from_runtime_half_reward():
    raw,path,cfg,ctrl,keys=reward_fixture()
    from analyze_native_frozen_validation import score
    _,half=score(dict(raw,domain_exit=np.zeros(8,bool)),path,cfg,ctrl,keys,.5)
    raw['effective_costs']=half['effective'];raw['scored_tick_reward']=-half['scored']
    result=m.reconstruct_execution_path_reward(raw,dict(alpha=.5),path,None,8,cfg,ctrl,keys)
    np.testing.assert_allclose(result['reward'],raw['scored_tick_reward'])
    assert not np.allclose(result['reward'],result['user1_reward'])
    np.testing.assert_allclose(result['user1_reward'],result['user1_running_components'].sum(axis=1))


def test_response_render_covers_each_route_event_with_xy_and_component_figures(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    saved=[];monkeypatch.setattr(Figure,'savefig',lambda self,file,**kw:saved.append(Path(file).name))
    (tmp_path/'analysis').mkdir()
    raw,path,cfg,ctrl,keys=reward_fixture();n=8
    path.update(xy=np.column_stack([path['s'],np.zeros(4)]),goal=np.asarray(3.))
    raw.update(actual_xy=np.column_stack([np.arange(n)*.1,np.zeros(n)]),wheel_speed_proxy=np.ones(n),
        phi=np.zeros(n),phi_dot=np.zeros(n),actual_delta=np.zeros(n),applied_force_impulse=np.zeros((n,3)))
    rows=[];raws={};rebuilt={}
    for method in ('fixed0.5','fixed1','speed1_steer0.5'):
        case=method
        rows.append(dict(route='route',event='nominal',method=method,case_id=case,force_N=0.,active_lower_ticks=n,
            prefix=dict(ticks=n),horizons={'10':dict(goal_arrived=False,physical_outcome='survived',first_arrival_s=None)},replan_task_tick=None))
        raws[case]=raw;rebuilt[case]=m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,n,cfg,ctrl,keys)
    m.render_response(tmp_path,rows,raws,dict(route=path),dict(route=np.array([3.,0.])),{},rebuilt,keys)
    assert set(saved)=={'full_xy_nominal.png','task_xy_nominal.png','route_nominal.png','route_nominal_components.png'}


def test_actual_permission_cannot_bypass_sixteen_clean_blocks_after_domain():
    raw,case=channel_raw()
    raw['clean_blocks_at_publication']=np.r_[np.zeros(4),np.full(4,3)]
    raw['clean_blocks_at_end']=np.r_[np.full(4,3),np.full(4,4)]
    raw['cumulative_domain_exit']=np.ones(8,bool)
    m.audit_response_channels(raw,case)
    raw['reference_target_permitted'][4:]=True;raw['latent_z'][4:]=[.7,.3]
    with pytest.raises(ValueError,match='sixteen complete clean blocks'):m.audit_response_channels(raw,case)
