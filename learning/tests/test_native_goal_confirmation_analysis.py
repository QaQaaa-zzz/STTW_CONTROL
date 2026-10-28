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


def switching_fixture():
    rows=[];raws={}
    for route in ('left','right'):
        for origin in ('fixed1','hybrid'):
            for event,force in (('zero',0.),('minus',-64.),('plus',64.)):
                stay=route+origin+event
                base=dict(route=route,origin=origin,event=event,force_N=force,onset_s=1.6,
                    horizons={'10':dict(goal_arrived=True,physical_outcome='survived',first_arrival_s=6.)},
                    diagnostic_execution_path_user1_cost=10.,full_observed_speed_rmse_m_s=.1,original_route_sample_distance_rmse_m=.2,
                    active_lower_ticks=2000,prefix=dict(ticks=1200,observed_s=6.))
                rows.append(dict(base,case_id=stay,mode='H75' if origin=='fixed1' else 'hybrid',method='stay_'+origin))
                raw=dict(active_tick=np.ones(340,bool),time=np.arange(340)*.005,actual_xy=np.zeros((340,2)))
                raws[stay]=raw
                for switch in (1.,1.4,1.6):
                    case=stay+str(switch)
                    rows.append(dict(base,case_id=case,mode='switch_'+origin,method='switch_'+origin+'@'+str(switch),switch_s=switch))
                    raws[case]={k:v.copy() for k,v in raw.items()}
    return rows,raws


def test_switch_at_actual_upper_tick280_not_scan_boundary_and_statealpha_stays_one():
    raw,case=channel_raw();n=400
    raw={k:np.broadcast_to(v[0],(n,)+v.shape[1:]).copy() for k,v in raw.items()}
    case.update(mode='switch_hybrid',origin='hybrid',switch_s=1.4)
    raw.update(reference_target_permitted=np.ones(n,bool),origin_channel_alpha_pair=np.tile([1.,.5],(n,1)),
        switch_tick=np.full(n,280),switch_active=np.arange(n)>=280)
    raw['channel_alpha_pair'][280:]=[.5,.5]
    raw['composed_latent'][:]=[.7,.3];raw['composed_latent'][280:]=[.2,.3];raw['latent_z']=raw['composed_latent'].copy()
    m.audit_response_channels(raw,case)
    raw['switch_active'][280:300]=False
    with pytest.raises(AssertionError):m.audit_response_channels(raw,case)


def test_all_own_switch_prefixes_allow_only_predeclared_float_tolerance_and_exact_discrete():
    rows,raws=switching_fixture()
    protocol=dict(prefix_comparison=dict(fields=['active_tick','time','actual_xy'],absolute_tolerance=1e-5,relative_tolerance=1e-6))
    candidate=next(r for r in rows if r.get('switch_s')==1.4)
    raws[candidate['case_id']]['actual_xy'][279,0]=1e-6
    proof=m.audit_switch_prefixes(rows,raws,protocol)
    assert len(proof)==36 and {r['physical_prefix_ticks'] for r in proof}=={200,280,320}
    raws[candidate['case_id']]['actual_xy'][279,0]=1e-3
    with pytest.raises(AssertionError,match='stay prefix mismatch'):m.audit_switch_prefixes(rows,raws,protocol)
    raws[candidate['case_id']]['actual_xy'][279,0]=0.
    raws[candidate['case_id']]['active_tick'][250]=False
    with pytest.raises(ValueError,match='complete physical prefix'):m.audit_switch_prefixes(rows,raws,protocol)


def test_switching_summary_keeps_four_source_groups_and_nominal_goal_loss():
    rows,_=switching_fixture()
    stay=next(r for r in rows if r['route']=='right' and r['origin']=='hybrid' and r['event']=='minus' and r['method'].startswith('stay'))
    stay['horizons']={'10':dict(goal_arrived=False,physical_outcome='contact',first_arrival_s=None)}
    summary,paired=m.switching_summary(rows)
    assert len(summary['source_groups'])==4 and len(paired)==36
    assert summary['source_groups']['right/hybrid']['contact_rescues']==3
    nominal=next(r for r in rows if r['route']=='right' and r['origin']=='hybrid' and r['event']=='zero' and r.get('switch_s')==1.6)
    nominal['horizons']={'10':dict(goal_arrived=False,physical_outcome='survived',first_arrival_s=None)}
    summary,paired=m.switching_summary(rows)
    assert not summary['source_groups']['right/hybrid']['nominal_without_new_contact_fault_or_goal_loss']
    assert any(r['goal1m_loss'] for r in paired if r['force_N']==0)
    assert summary['adoption']=='not_automatically_adopted'


def test_switching_inventory_uses_origin_and_every_true_switch_time():
    protocol=dict(routes=[dict(route='route',speed=2.8)],origins=[dict(origin='hybrid',speed_alpha=1.,steer_alpha=.5)],
        switch_times_s=[1.,1.4,1.6],forces_N=[-64.,64.],onsets_s=[1.6],zero_onset_s=1.6,duration_s=.1)
    inventory=[]
    for switch in protocol['switch_times_s']:
        for force in (0.,-64.,64.):
            inventory.append(dict(route='route',speed=2.8,origin='hybrid',mode='switch_hybrid',alpha=1.,speed_alpha=1.,steer_alpha=.5,
                switch_s=switch,force_N=force,onset_s=1.6,duration_s=.1))
    m.validate_switching_inventory(inventory,protocol)
    inventory[-1]['switch_s']=1.5
    with pytest.raises(ValueError,match='switching inventory'):m.validate_switching_inventory(inventory,protocol)


def test_switching_render_keeps_each_origin_and_fixed_half_background(tmp_path,monkeypatch):
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
    for origin in ('fixed1','hybrid'):
        for switch in (1.,1.4,1.6,None):
            case=origin+str(switch);method=('switch_'+origin+'@'+str(switch)) if switch else 'stay_'+origin
            row=dict(route='route',event='zero',method=method,origin=origin,case_id=case,force_N=0.,active_lower_ticks=n,
                mode='switch_'+origin if switch else 'H75',prefix=dict(ticks=n),
                horizons={'10':dict(goal_arrived=False,physical_outcome='survived',first_arrival_s=None)},replan_task_tick=None)
            if switch:row['switch_s']=switch
            rows.append(row);raws[case]=raw;rebuilt[case]=m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,n,cfg,ctrl,keys)
    rows.append(dict(rows[-1],origin='background',method='background_fixed0.5',case_id='background'))
    raws['background']=raw;rebuilt['background']=rebuilt[rows[0]['case_id']]
    m.render_response(tmp_path,rows,raws,dict(route=path),dict(route=np.array([3.,0.])),{},rebuilt,keys)
    assert len(saved)==8
    assert 'full_xy_zero__fixed1.png' in saved and 'full_xy_zero__hybrid.png' in saved
    assert all(len(m.response_plot_selection(rows,label))==5 for label in m.response_plot_groups(rows))


def test_continuation_phi_dot_amendment_does_not_relax_other_fields():
    rows,raws=switching_fixture()
    for raw in raws.values():raw['phi_dot']=np.zeros(340)
    protocol=dict(prefix_comparison=dict(fields=['active_tick','actual_xy','phi_dot'],absolute_tolerance=1e-5,
        relative_tolerance=1e-6,field_absolute_tolerance=dict(phi_dot=1e-4)))
    candidate=next(r for r in rows if r.get('switch_s')==1.4)
    raw=raws[candidate['case_id']];raw['phi_dot'][279]=5e-5
    proof=m.audit_switch_prefixes(rows,raws,protocol)
    assert len(proof)==36
    raw['actual_xy'][279,0]=5e-5
    with pytest.raises(AssertionError,match='stay prefix mismatch'):m.audit_switch_prefixes(rows,raws,protocol)


def test_continuation_reuses_each_own_prefix_without_canonical_lane_copy(tmp_path):
    import json
    rows,raws=switching_fixture();new=[r for r in rows if r['mode'].startswith('switch_')]
    source=tmp_path/'failed';run=tmp_path/'resumed';(source/'evaluation').mkdir(parents=True);(run/'evaluation').mkdir(parents=True)
    arrays=dict(active_tick=np.ones((200,36),bool),time=np.tile(np.arange(200)[:,None]*.005,(1,36)),actual_xy=np.zeros((200,36,2)))
    for lane,row in enumerate(new):
        arrays['actual_xy'][:,lane,0]=lane*1e-7
        for field in arrays:raws[row['case_id']][field][:200]=arrays[field][:,lane]
    np.savez(source/'evaluation/chunk_00.npz',**arrays);np.savez(run/'evaluation/reused_source_prefix.npz',**arrays)
    (run/'evaluation/prefix_provenance.json').write_text(json.dumps(dict(source=str(source),source_indices=list(range(36)),reused_lower_slots=7200,new_physics_ticks=0)))
    (run/'evaluation/source_prefix_audit.json').write_text(json.dumps(dict(all_fields_within_tolerance=True,discrete_exact=True)))
    plan=dict(continuation_source=dict(source=str(source)),protocol=dict(prefix_comparison={}))
    assert m.audit_continuation_prefix(run,plan,rows,raws)['all_reused_source_fields_exact']
    raws[new[1]['case_id']]['actual_xy'][:200]=raws[new[0]['case_id']]['actual_xy'][:200]
    with pytest.raises(AssertionError,match='own source prefix'):m.audit_continuation_prefix(run,plan,rows,raws)


def test_user_authorized_finite_float_prefix_differences_are_descriptive_only():
    rows,raws=switching_fixture()
    protocol=dict(prefix_comparison=dict(fields=['active_tick','actual_xy'],absolute_tolerance=1e-5,
        relative_tolerance=1e-6,floating_descriptive_only=True))
    candidate=next(r for r in rows if r.get('switch_s')==1.4)
    raws[candidate['case_id']]['actual_xy'][279,0]=.001
    proof=m.audit_switch_prefixes(rows,raws,protocol)
    changed=next(r for r in proof if r['case_id']==candidate['case_id'])
    assert changed['floating_descriptive_only'] and not changed['floating_fields_within_declared_tolerance']
    assert changed['tolerance_exceedance_element_counts']==dict(actual_xy=1)
    assert changed['maximum_absolute_difference']['actual_xy']==.001
    raws[candidate['case_id']]['actual_xy'][279,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite switch prefix'):m.audit_switch_prefixes(rows,raws,protocol)
    raws[candidate['case_id']]['actual_xy'][279,0]=0.
    raws[candidate['case_id']]['active_tick'][279]=False
    with pytest.raises(ValueError,match='complete physical prefix'):m.audit_switch_prefixes(rows,raws,protocol)


def online_fixture():
    n=12;alpha=np.repeat([.5,1.,1.],4)
    raw=dict(active_tick=np.ones(n,bool),alpha_user=np.ones(n),alpha_exec=alpha,rule_latched=alpha==.5,
        rule_preview_max_curvature=np.repeat([.5,.1,0.],4),rule_accel_estimate=np.repeat([3.92,.784,0.],4),rule_alpha_target=alpha,
        channel_alpha_pair=np.column_stack([alpha,alpha]),query_observation=np.zeros((n,351)),
        composed_latent=np.zeros((n,2)),latent_z=np.zeros((n,2)),reference_target_permitted=np.ones(n,bool),
        actual_forward_speed=np.full(n,2.7),v_user=np.full(n,2.8),path_progress=np.repeat([1.8,2.5,3.5],4))
    raw['query_observation'][:,336]=alpha
    path=dict(s=np.arange(5,dtype=np.float32),curvature=np.array([0.,.5,0.,0.,0.],np.float32))
    case=dict(mode='online_scalar',alpha=1.,speed=2.8,onset_s=1.6)
    rule=dict(preview_m=1.5,engage_accel_m_s2=3.,release_accel_m_s2=2.7,active_alpha=.5)
    return raw,path,case,rule


def test_online_actual_scalar_query_and_independent_closed_interval_latch():
    raw,path,case,rule=online_fixture()
    m.audit_online_publication(raw,case)
    result=m.audit_online_rule(raw,case,path,None,rule)
    assert result['departure_from_one_times_s']==[0.]
    assert result['return_to_one_times_s']==[.02]
    assert result['completed_dwell_s']==[.02] and not result['unreturned_at_actual_endpoint']
    assert np.isclose(result['speed_error_rmse_m_s'],.1)
    raw['query_observation'][5,336]=.5
    with pytest.raises(AssertionError):m.audit_online_publication(raw,case)


def test_online_rule_latch_survives_rebound_and_reports_open_activation():
    raw,path,case,rule=online_fixture();path['curvature'][:]=.5
    replan=dict(path,curvature=np.full(5,2.85/7.84,np.float32));case['replan_task_tick']=8
    raw['alpha_exec'][:]=.5;raw['rule_alpha_target'][:]=.5;raw['rule_latched'][:]=True
    raw['rule_preview_max_curvature'][:8]=.5;raw['rule_accel_estimate'][:8]=3.92
    raw['rule_preview_max_curvature'][8:]=2.85/7.84;raw['rule_accel_estimate'][8:]=2.85
    result=m.audit_online_rule(raw,case,path,replan,rule)
    assert result['departure_count']==1 and result['return_count']==0
    assert result['unreturned_at_actual_endpoint'] and result['open_dwell_observed_s']==.06


def test_confirmation_inventory_keeps_bank_paths_and_state_query_alpha_separate():
    import json
    p=json.loads((CLI.parents[0]/'configs/h75_online_confirmation100.json').read_text());rows=[]
    events=[(0.,p['zero_onset_s'])]+list(m.itertools.product(p['forces_N'],p['onsets_s']))
    for bank,route,arm,(force,onset) in m.itertools.product(p['bank_indices'],p['routes'],p['arms'],events):
        rows.append(dict(bank_index=bank,route=route['route'],speed=route['speed'],mode=arm,alpha=1.,
            query_alpha={'fixed0':0.,'fixed_half':.5}.get(arm,1.),initial_fingerprint=p['bank_state_fingerprints'][str(bank)],
            path_key=f"bank{bank}_"+route['route'],event=str(force)+str(onset),force_N=force,onset_s=onset,duration_s=p['duration_s']))
    m.validate_confirmation_inventory(rows,p)
    assert len({m.analysis_path_key(r) for r in rows})==4
    assert len(m.response_plot_groups(rows))==10
    rows[-1]['path_key']='bank0_'+rows[-1]['route']
    with pytest.raises(ValueError,match='bank-aware'):m.validate_confirmation_inventory(rows,p)


def test_fixed_zero_query_keeps_user_state_one_and_floor_reports_inactive_zero():
    raw,path,case,rule=online_fixture();n=len(raw['active_tick'])
    case.update(mode='fixed0',path_key='bank0_route',query_alpha=0.)
    raw['alpha_exec'][:]=0.;raw['rule_latched'][:]=False;raw['query_observation'][:,336]=0.;raw['channel_alpha_pair'][:]=0.
    raw['neural_query_active']=np.ones(n,bool)
    m.audit_online_publication(raw,case)
    raw['alpha_user'][0]=0.
    with pytest.raises(AssertionError):m.audit_online_publication(raw,case)
    raw['alpha_user'][:]=1.;case.update(mode='zero_with_existing_speed_floor',query_alpha=1.)
    raw['alpha_exec'][:]=1.;raw['query_observation'][:,336]=1.;raw['channel_alpha_pair'][:]=1.
    raw['neural_query_active'][:]=False;raw['support_nominal_reference']=np.tile([2.,0.],(n,1));raw['nominal']=raw['support_nominal_reference'].copy()
    raw.update(support_floor=np.full(n,1.5),support_requested_offset=np.zeros(n),support_realized_target_offset=np.zeros(n),support_active=np.zeros(n,bool))
    m.audit_online_publication(raw,case)
    result=m.audit_floor_support(raw,case,dict(residual_interface=dict(speed_positive_scale_m_s=.4)),1.5,[2.,0.])
    assert result['equivalent_to_zero_in_observed_run'] and result['observed_support_active_ticks']==0


def test_floor_support_uses_logged_preupper_nominal_and_original_latent_mapping():
    raw,path,case,rule=online_fixture();n=len(raw['active_tick']);case.update(mode='zero_with_existing_speed_floor')
    raw.update(support_nominal_reference=np.tile([1.4,0.],(n,1)),nominal=np.tile([1.4,0.],(n,1)),
        support_floor=np.full(n,1.5),support_requested_offset=np.full(n,.1),support_realized_target_offset=np.full(n,.1),
        support_active=np.ones(n,bool),neural_query_active=np.zeros(n,bool))
    raw['composed_latent'][:]=[np.arctanh(.25),0.]
    result=m.audit_floor_support(raw,case,dict(residual_interface=dict(speed_positive_scale_m_s=.4)),1.5,[1.4,0.])
    assert not result['equivalent_to_zero_in_observed_run'] and result['observed_support_active_ticks']==n
    raw['composed_latent'][4,0]=0.
    with pytest.raises(AssertionError):m.audit_floor_support(raw,case,dict(residual_interface=dict(speed_positive_scale_m_s=.4)),1.5,[1.4,0.])


def test_confirmation_summary_pairs_only_same_bank_and_retains_floor_identity():
    rows=[]
    for bank,route,event,method in m.itertools.product((0,1),('left','right'),('zero','t2minus','t2plus','t3minus','t3plus'),
        ('online_scalar','fixed0','fixed0.5','fixed1','zero_with_existing_speed_floor')):
        rows.append(dict(bank_index=bank,route=route,event=event,method=method,mode=method,force_N=0. if event=='zero' else 64.,
            case_id=str(bank)+route+event+method,horizons={'10':dict(goal_arrived=True,physical_outcome='survived',first_arrival_s=6.)},
            prefix=dict(ticks=1200),alpha_rule=dict(unreturned_at_actual_endpoint=False),
            floor_support=dict(equivalent_to_zero_in_observed_run=True),full_observed_speed_rmse_m_s=.2,diagnostic_execution_path_user1_cost=10.))
    summary,paired=m.confirmation_summary(rows)
    assert summary['new_cases']==100 and summary['cached_cases']==0
    assert len(paired)==4 and all(len(p['details'])==20 for p in paired)
    assert summary['method_bank_subset']['online_scalar']['0']['all']['n']==10
    assert summary['method_bank_subset']['online_scalar']['1']['nominal']['n']==2
    assert len(summary['floor_support'])==20
    for p in paired:
        for d in p['details']:assert d['candidate'][0]==d['baseline'][0]


def test_confirmation_xy_panels_keep_two_routes_per_bank_and_five_methods(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    saved=[];monkeypatch.setattr(Figure,'savefig',lambda self,file,**kw:saved.append(Path(file).name))
    (tmp_path/'analysis').mkdir();raw,path,cfg,ctrl,keys=reward_fixture();n=8
    path.update(xy=np.column_stack([path['s'],np.zeros(4)]),goal=np.asarray(3.))
    raw.update(actual_xy=np.column_stack([np.arange(n)*.1,np.zeros(n)]),wheel_speed_proxy=np.ones(n),phi=np.zeros(n),phi_dot=np.zeros(n),
        actual_delta=np.zeros(n),applied_force_impulse=np.zeros((n,3)),alpha_exec=np.ones(n),rule_accel_estimate=np.zeros(n))
    rows=[];raws={};rebuilt={};paths={};goals={}
    for bank,route,method in m.itertools.product((0,1),('left','right'),('online_scalar','fixed0','fixed0.5','fixed1','zero_with_existing_speed_floor')):
        key=f'bank{bank}_'+route;case=key+method;paths[key]=path;goals[key]=np.array([3.,0.])
        rows.append(dict(bank_index=bank,route=route,path_key=key,event='zero',method=method,mode=method,case_id=case,force_N=0.,active_lower_ticks=n,
            prefix=dict(ticks=n),horizons={'10':dict(goal_arrived=False,physical_outcome='survived',first_arrival_s=None)},replan_task_tick=None))
        raws[case]=raw;rebuilt[case]=m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,n,cfg,ctrl,keys)
    m.render_response(tmp_path,rows,raws,paths,goals,{},rebuilt,keys)
    assert len(saved)==16
    assert 'full_xy_zero__bank0.png' in saved and 'full_xy_zero__bank1.png' in saved
    assert len(set(saved))==len(saved)


def tolerance_fixture(n=1800):
    raw=dict(active_tick=np.ones(n,bool),time=1.+np.arange(n)*.005,policy_fault=np.zeros(n,bool),lower_fault=np.zeros(n,bool),
        nonfinite=np.zeros(n,bool),continuation_geometry_fault=np.zeros(n,bool),contact_failure=np.zeros(n,bool),physical_failure=np.zeros(n,bool),domain_exit=np.ones(n,bool))
    case=dict(primary_window_start_s=1.,primary_window_end_s=4.3,onset_s=1.2,duration_s=.1)
    return raw,case


def test_same_state_primary_includes_preparation_and_ignores_later_contact():
    raw,case=tolerance_fixture();raw['contact_failure'][900]=True;raw['physical_failure'][900]=True
    assert m.samestate_primary_label(raw,case)['no_fall_through_window'] is True
    raw['contact_failure'][20]=True;raw['physical_failure'][20]=True
    result=m.samestate_primary_label(raw,case)
    assert result['no_fall_through_window'] is False and result['prep_contact'] and result['prep_physical_failure']
    assert result['label_status']=='fall'


def test_same_state_fault_is_invalid_not_negative_and_complete_window_required():
    raw,case=tolerance_fixture();raw['policy_fault'][100]=True
    result=m.samestate_primary_label(raw,case)
    assert result['no_fall_through_window'] is None and result['label_status']=='invalid_fault'
    raw,case=tolerance_fixture(200)
    assert m.samestate_primary_label(raw,case)['label_status']=='invalid_incomplete'
    raw['contact_failure'][-1]=True;raw['physical_failure'][-1]=True
    assert m.samestate_primary_label(raw,case)['no_fall_through_window'] is False


def test_same_state_primary_includes_transition_ending_at_declared_window_endpoint():
    raw,case=tolerance_fixture();end=round((case['primary_window_end_s']-1.)/.005)
    raw['contact_failure'][end]=True;raw['physical_failure'][end]=True
    assert m.samestate_primary_label(raw,case)['no_fall_through_window'] is True
    raw['contact_failure'][end-1]=True;raw['physical_failure'][end-1]=True
    assert m.samestate_primary_label(raw,case)['no_fall_through_window'] is False


def test_same_state_inventory_source_groups_and_paired_labels_are_not_independent_states():
    import json
    p=json.loads((CLI.parents[0]/'configs/h75_samestate_tolerance72.json').read_text());rows=[]
    for src,alpha,prep,force in m.itertools.product(p['source_rows'],p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        case=dict(route=src['route'],source_index=src['source_index'],source_origin=src['origin'],source_fingerprint=src['fingerprint'],
            mode='samestate_scalar',alpha=1.,query_alpha=alpha,candidate_alpha=alpha,preparation_s=prep,force_N=force,duration_s=.1,onset_s=1.+prep,
            primary_window_start_s=1.,primary_window_end_s=1.+prep+.1+3.,case_id=f"{src['source_index']}_{alpha}_{prep}_{force}")
        raw,_=tolerance_fixture();case.update(primary=m.samestate_primary_label(raw,case),horizons={'10':dict(goal_arrived=False,physical_outcome='survived')},
            event=f'prep{prep}_F{force}')
        rows.append(case)
    m.validate_samestate_inventory(rows,p);summary,paired=m.samestate_summary(rows)
    assert summary['conditional_suffixes']==72 and summary['independent_source_states']==4
    assert all(g['n_conditional_suffixes']==18 for g in summary['source_groups'].values())
    assert len(paired)==72 and len(m.response_plot_groups(rows))==24
    assert all(len(m.response_plot_selection(rows,key))==3 for key in m.response_plot_groups(rows))
    rows[-1]['onset_s']+=.02
    with pytest.raises(ValueError,match='preparation/onset'):m.validate_samestate_inventory(rows,p)


def test_same_state_candidate_query_changes_immediately_at_source_with_real_mask():
    raw,path,case,rule=online_fixture();n=len(raw['active_tick']);case.update(candidate_alpha=0.)
    raw['alpha_exec'][:]=0.;raw['query_observation'][:,336]=0.;raw['channel_alpha_pair'][:]=0.;raw['time']=1.+np.arange(n)*.005
    result=m.audit_samestate_query(raw,case)
    assert result['first_actual_permitted_query_s']==1.
    raw['reference_target_permitted'][4:8]=False;raw['latent_z'][4:8]=.1
    with pytest.raises(AssertionError):m.audit_samestate_query(raw,case)


def bank144_rows():
    import json
    p=json.loads((CLI.parents[0]/'configs/h75_samestate_bank144.json').read_text());rows=[]
    for group,alpha,prep,force in m.itertools.product(p['source_groups'],p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        rows.append(dict(group,source_origin=group['origin'],mode='samestate_scalar',alpha=1.,candidate_alpha=alpha,query_alpha=alpha,
            preparation_s=prep,force_N=force,onset_s=1.+prep,duration_s=.1,primary_window_start_s=1.,primary_window_end_s=1.+prep+.1+3.,
            case_id=f"{group['source_index']}_{alpha}_{prep}_{force}",raw_file='heldout-never-open' if group['bank_index']==1 else 'dev'))
    return rows,p


def test_bank144_default_label_selection_never_selects_bank1_raw_or_metrics():
    rows,p=bank144_rows();m.validate_bank144_inventory(rows,p)
    selected=m.select_bank_label_lanes(rows)
    assert len(selected)==72 and all(c['bank_index']==0 for _,c in selected)
    assert all(c['raw_file']=='dev' for _,c in selected)
    assert {c['source_index'] for _,c in selected}=={0,1,2,3}
    unsealed=m.select_bank_label_lanes(rows,include_heldout_labels=True)
    assert len(unsealed)==144 and {c['bank_index'] for _,c in unsealed}=={0,1}
    rows[-1]['split']='development'
    with pytest.raises(ValueError,match='seal metadata'):m.select_bank_label_lanes(rows)


def test_bank144_development_summary_exports_no_heldout_source_or_label():
    rows,_=bank144_rows();selected=m.select_bank_label_lanes(rows);development=[]
    for _,r in selected:
        development.append(dict(r,primary=dict(no_fall_through_window=True),horizons={'10':dict(goal_arrived=True,physical_outcome='survived')}))
    summary,paired=m.samestate_summary(development)
    assert summary['conditional_suffixes']==72 and set(summary['source_groups'])=={'0','1','2','3'}
    assert all(p['source_index'] in (0,1,2,3) for p in paired)
    assert len(paired)==72


def test_suffix_reward_terminal_keeps_absolute_task_clock_without_prefix_replay():
    raw,path,cfg,ctrl,keys=reward_fixture();raw['time']=1.+np.arange(8)*.005;raw['physical_failure'][-1]=True
    remaining=500-(200+7)//4;c=cfg['path_reward'];g=cfg['training_future_phase_C']['gamma']
    raw['scored_tick_reward'][4:]=0.
    raw['scored_tick_reward'][7]=-c['failure_extra']-c['scale']*.02*c['failure_cost_rate_bound']*(1-g**remaining)/(1-g)
    result=m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,8,cfg,ctrl,keys,start_tick=200)
    np.testing.assert_allclose(result['reward'],raw['scored_tick_reward'])
    with pytest.raises(AssertionError,match='per-step reward'):
        m.reconstruct_execution_path_reward(raw,dict(alpha=1.),path,None,8,cfg,ctrl,keys)


def test_bank144_runtime_inventory_adds_only_sealed_generated_source_fingerprint():
    inventory,p=bank144_rows()
    sources=[dict(g,source_fingerprint=f"{g['source_index']:064x}") for g in p['source_groups']]
    expected=m.enriched_bank_runtime_inventory(inventory,sources)
    assert len(expected)==144 and expected[0]['source_fingerprint']==sources[0]['source_fingerprint']
    assert 'source_fingerprint' not in inventory[0]
    assert {k:v for k,v in expected[18].items() if k!='source_fingerprint'}==inventory[18]
    wrong=[dict(c) for c in inventory];wrong[0]['source_fingerprint']='f'*64
    with pytest.raises(ValueError,match='fingerprint conflict'):m.enriched_bank_runtime_inventory(wrong,sources)
    sources[0]['bank_index']=1
    with pytest.raises(ValueError,match='source metadata'):m.enriched_bank_runtime_inventory(inventory,sources)


def test_new_confirmation_labels_require_dedicated_flag_and_exclude_old_banks(tmp_path):
    import json
    from sttw_control.native_goal_run import bank144_inventory
    p=json.loads((CLI.parents[0]/'configs/h75_samestate_confirmation144.json').read_text())
    rows=bank144_inventory(p);m.validate_bank144_inventory(rows,p)
    for heldout in (False,True):
        with pytest.raises(ValueError,match='confirmation flag'):
            m.select_bank_label_lanes(rows,include_heldout_labels=heldout)
    assert len(m.select_bank_label_lanes(rows,include_confirmation_labels=True))==144
    old,_=bank144_rows()
    with pytest.raises(ValueError,match='historical'):
        m.select_bank_label_lanes(old,include_confirmation_labels=True)
    bad=[dict(r) for r in rows];bad[-1]['bank_index']=1
    with pytest.raises(ValueError,match='identity'):
        m.select_bank_label_lanes(bad,include_confirmation_labels=True)
    # No suffix files exist: missing opt-in must reject before any label-file open.
    with pytest.raises(ValueError,match='include-confirmation-labels'):
        m.analyze_bank144(tmp_path,{'protocol':p})


def test_confirmation_uses_own_generated_paths_and_preserves_partial_audit(tmp_path):
    root=tmp_path/'new_run';ancestor=tmp_path/'old_confirmation100'
    own=root/'evaluation';old=ancestor/'evaluation';own.mkdir(parents=True);old.mkdir(parents=True)
    name='bank2_S_left_90_R1p8_path.npz'
    np.savez(own/name,xy=np.array([[1.,2.],[3.,4.]]))
    new={'protocol':{'source_initialization':'prepared_bank_all_prefixes'},'samestate_source':{'source':str(ancestor)}}
    directory=m.bank144_original_path_directory(root,new)
    assert directory==own and not (old/name).exists()
    with np.load(directory/name) as z:np.testing.assert_array_equal(z['xy'],[[1.,2.],[3.,4.]])
    assert m.bank144_original_path_directory(root,dict(new,protocol={}))==old
    out=root/'analysis_confirmation';target=m.bank144_analysis_audit_target(out)
    assert target==out/'initial_carry_review.json'
    target.write_text('preserved audit')
    assert m.bank144_analysis_audit_target(out)==out/'initial_carry_review_recheck.json'
    assert target.read_text()=='preserved audit'
    (out/'results.json').write_text('[]')
    with pytest.raises(ValueError,match='already contains'):
        m.bank144_analysis_audit_target(out)
