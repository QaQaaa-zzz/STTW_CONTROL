"""Offline capability labels never turn censoring into success."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

CLI = Path(__file__).parents[1] / 'cli'
sys.path.insert(0, str(CLI))
spec = importlib.util.spec_from_file_location('native_capability_analysis', CLI / 'analyze_native_capability.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(n=20):
    path = dict(s=np.arange(6.), xy=np.column_stack([np.arange(6.), np.zeros(6)]),
                curvature=np.array([0., 1., 0., -1., 0., 0.]), goal=np.array(5.), turn_end=np.array(4.))
    raw = dict(time=np.arange(n)*.1, actual_xy=np.tile([5.,0.],(n,1)),
               path_progress=np.full(n,5.), goal_section_signed_distance=np.zeros(n), path_cross_track=np.zeros(n), path_heading_error=np.zeros(n),
               actual_forward_speed=np.ones(n), v_user=np.ones(n),
               physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool),contact_failure=np.zeros(n,bool))
    return raw,path


def test_complete_final_window_and_both_bends():
    raw,path=fixture()
    out=module.labels(raw,path,.1,2.)
    assert out['final_1s_thresholds_met']
    assert out['reached_second_bend_before_native_stop']
    assert out['contact_observation']=='complete_no_contact'


def test_domain_stop_is_censored_and_fatal_goal_arrival_excluded():
    raw,path=fixture()
    raw['actual_xy'][:-1]=[0.,0.]
    raw['path_progress'][:-1]=0.
    raw['domain_exit'][-1]=True
    out=module.labels(raw,path,.1,2.)
    assert out['contact_observation']=='censored_no_contact'
    assert not out['goal1m_before_native_stop']
    assert not out['original_goal_crossed_before_native_stop']
    assert not out['final_1s_thresholds_met']


def test_full_window_required_instead_of_final_sample():
    raw,path=fixture()
    raw['path_heading_error'][-5]=.051
    assert not module.labels(raw,path,.1,2.)['final_1s_thresholds_met']


def test_goal_progress_without_final_section_crossing_is_not_goal_crossing():
    raw,path=fixture()
    raw["goal_section_signed_distance"][:]=-0.1
    assert not module.labels(raw,path,.1,2.)["original_goal_crossed_before_native_stop"]


def score_fixture():
    raw, path = fixture(n=20)
    raw.update(effective_costs=np.zeros((20, 4)), scored_tick_reward=np.zeros(20))
    raw['actual_forward_speed'][:] = 1.15
    raw['path_cross_track'][:] = .2
    keys = ['speed', 'path', 'primary', 'roll']
    cfg = dict(timing=dict(lower_dt_s=.1, upper_dt_s=.4),
               training_future_phase_C=dict(episode_s=2., gamma=.99),
               path_reward=dict(reward_version='path_persistent_preference_v3',
                                component_caps=dict(speed=100., path=100., primary=100.),
                                speed_scale_m_s=.1, cross_track_scale_m=.1,
                                path_heading_factor=1., path_heading_scale_rad=.1,
                                primary_weight=4., primary_path_band_m=.1,
                                primary_path_scale_m=.1, primary_speed_band_m_s=.1,
                                primary_speed_scale_m_s=.1, path_secondary_ratio=.1, scale=1.))
    controller = dict(minimum_speed=.1, mass=10., cg_height=.2, cg_forward=.1,
                      trail=.02, gravity=9.81, caster=.4, wheelbase=.4)
    return raw, path, cfg, controller, keys


def test_dynamic_runtime_alpha_is_validated_separately_from_original_user_objective():
    raw, path, cfg, controller, keys = score_fixture()
    alpha = np.linspace(1., .5, len(raw['time']))
    own, series = module.score(raw, path, cfg, controller, keys, alpha)
    raw['scored_tick_reward'] = -series['scored']
    result, data, difference = module.original_user_score(raw, path, cfg, controller, keys, alpha)
    fixed, fixed_series = module.score(raw, path, cfg, controller, keys, 1.)
    assert difference == 0.
    assert result == fixed
    np.testing.assert_array_equal(data['scored'], fixed_series['scored'])
    assert not np.isclose(result['scored_cost'], own['scored_cost'])
    import pytest
    with pytest.raises(AssertionError, match='runtime reward mismatch'):
        module.original_user_score(raw, path, cfg, controller, keys, 1.)


def test_missing_or_changed_cached_trace_is_rejected_without_substitution(tmp_path):
    import pytest
    row = dict(raw_file='raw.npz', raw_sha256='missing')
    with pytest.raises(ValueError, match='missing or changed'):
        module.verified_raw(row, tmp_path)
    np.savez(tmp_path/'raw.npz', time=np.array([0.]))
    with pytest.raises(ValueError, match='missing or changed'):
        module.verified_raw(row, tmp_path)
    row['raw_sha256'] = module.digest(tmp_path/'raw.npz')
    assert module.verified_raw(row, tmp_path)['time'].tolist() == [0.]


def test_sealed_manifest_identity_is_not_inferred_from_raw_existence(tmp_path):
    import json
    import pytest
    (tmp_path/'evaluation').mkdir()
    plan = dict(cases=[], config={'frozen':1}, initial_source={'bank':'same'}, native_runtime_root='native')
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'seal.json').write_text(json.dumps(dict(plan_sha256=module.digest(tmp_path/'plan.json'))))
    (tmp_path/'evaluation/status.json').write_text(json.dumps(dict(phase='completed', completed_cases=0)))
    manifest = dict(plan_sha256=module.digest(tmp_path/'plan.json'), cases=[], config={'frozen':2}, source=plan['initial_source'], native_runtime_root='native')
    (tmp_path/'evaluation/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest identity mismatch: config'):
        module.sealed_panel(tmp_path, 0)


def test_acceptance_uses_separate_best_strict_simple_objectives_and_rejects_missing_return():
    import json
    protocol = json.loads((CLI.parents[0]/'configs/h75_nominal_preference_schedule.json').read_text())
    contexts = [dict(route='gL', angle_deg=60), dict(route='gR', angle_deg=60),
                dict(route='tL', angle_deg=90), dict(route='tR', angle_deg=90)]
    paths = {c['route']:dict(turn_end=1.) for c in contexts}
    records, scores, traces = [], [], {}
    costs = {'fixed0':(10., 8.), 'fixed0.5':(9., 9.), 'fixed1':(5., 5.),
             'curvature_speed':(12., 7.), 'alpha_schedule':(8.5, 6.8), 'zero':(1.,1.)}
    for route in paths:
        for role, (total, channels) in costs.items():
            case=route+role
            records.append(dict(route=route, strategy=role, analysis_id=case,
                                task_recovered=role not in ('fixed1','zero'), actual_contact=False,
                                native_domain_stop=False, native_physical_stop=False))
            scores.append(dict(analysis_id=case, scored_cost=total, task_channels=channels))
            traces[case]=dict(time=np.arange(20)*.1, path_progress=np.arange(20)*.1,
                              executed_alpha=np.r_[np.full(15,.5),np.ones(5)])
    # Avoid the independent gentle regression gate masking the comparator check.
    for s in scores:
        if s['analysis_id'].endswith('fixed1'):
            s['scored_cost']=10.
    out=module.scheduling_acceptance(records,scores,traces,paths,protocol['acceptance'],contexts)
    assert out['best_strict_simple_costs']==dict(total_cost=36.,task_channels=28.)
    assert out['gates']['aggregate_total_cost_improvement']
    assert not out['gates']['aggregate_task_channels_improvement']
    assert not out['accepted']
    traces['tLalpha_schedule']['executed_alpha'][:]=.5
    out=module.scheduling_acceptance(records,scores,traces,paths,protocol['acceptance'],contexts)
    assert not out['gates']['alpha_returns_1_after_final_curve']


def test_channel_composition_checks_each_separate_query_and_applied_target():
    import pytest
    raw = dict(time=np.arange(4), queried_latent_alpha_half=np.tile([.2,.3], (4,1)),
               queried_latent_alpha_one=np.tile([.7,.8], (4,1)),
               composed_latent=np.tile([.2,.8],(4,1)), latent_z=np.tile([.2,.8],(4,1)),
               channel_alpha_pair=np.tile([.5,1.],(4,1)), original_user_alpha=np.ones(4))
    row = dict(speed_alpha=.5, steer_alpha=1.)
    assert module.verify_channel_composition(raw,row)==0.
    raw['composed_latent'][2,1]=.3
    with pytest.raises(AssertionError, match='composition mismatch'):
        module.verify_channel_composition(raw,row)
    raw['composed_latent'][2,1]=.8
    raw['latent_z'][1,0]=.7
    with pytest.raises(AssertionError, match='applied latent'):
        module.verify_channel_composition(raw,row)


def channel_fixture():
    contexts=[dict(route='gL',angle_deg=60),dict(route='gR',angle_deg=60),
              dict(route='tL',angle_deg=90),dict(route='tR',angle_deg=90)]
    records=[]; scores=[]; traces={}
    for c in contexts:
        for strategy in ('fixed0','fixed0.5','fixed1','zero','speed0.5_steer1','speed1_steer0.5'):
            case=c['route']+strategy
            records.append(dict(route=c['route'],strategy=strategy,analysis_id=case,
                                channel_strict=not(strategy=='fixed1' and c['angle_deg']==90)))
            scores.append(dict(analysis_id=case,scored_cost=10.,task_channels=8.))
            traces[case]=dict(actual_forward_speed=np.full(20,2.+(.1 if strategy.startswith('speed') else 0.)))
    return records,scores,traces,{c['route']:{} for c in contexts},contexts


def test_channel_descriptive_rules_retain_cost_exceptions_losses_and_actual_speed():
    args=channel_fixture()
    result=module.channel_interpretation(*args)
    assert result['speed_half_steer_one_covers_fixed_half']
    assert result['reverse_improves_fixed_one']
    assert result['adoption']=='not_automatically_adopted'
    assert np.isclose(result['per_route_differences'][0]['mean_signed_speed_difference_m_s'],.1)
    # One route task-channel exception breaks coverage despite strict success.
    next(s for s in args[1] if s['analysis_id']=='gLspeed0.5_steer1')['task_channels']=8.51
    result=module.channel_interpretation(*args)
    assert not result['speed_half_steer_one_covers_fixed_half']
    assert result['cost_exceptions']
    # A gentle strict loss vetoes a reverse tight gain.
    next(r for r in args[0] if r['analysis_id']=='gRspeed1_steer0.5')['channel_strict']=False
    result=module.channel_interpretation(*args)
    assert not result['reverse_improves_fixed_one']
    assert any(x['strict_loss'] for x in result['per_route_differences'])


def test_channel_pair_identity_cannot_be_swapped_or_nonfinite():
    import pytest
    raw=dict(time=np.arange(2),queried_latent_alpha_half=np.zeros((2,2)),
             queried_latent_alpha_one=np.ones((2,2)),composed_latent=np.tile([0.,1.],(2,1)),
             latent_z=np.tile([0.,1.],(2,1)),channel_alpha_pair=np.tile([.5,1.],(2,1)),
             original_user_alpha=np.ones(2))
    with pytest.raises(AssertionError):
        module.verify_channel_composition(raw,dict(speed_alpha=1.,steer_alpha=.5))
    raw['queried_latent_alpha_one'][0,0]=np.nan
    with pytest.raises(ValueError,match='invalid channel log'):
        module.verify_channel_composition(raw,dict(speed_alpha=.5,steer_alpha=1.))
