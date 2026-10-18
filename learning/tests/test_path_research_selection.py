"""Frozen research acceptance on saved trace-shaped data; no physics rollouts."""
from copy import deepcopy
import numpy as np
import pytest
from test_path_strong_selection import config as strong_config, preferred
from sttw_control import path_command_selection as selection


def config():
    cfg=strong_config()
    cfg['preference_selection_schema']='path_preference_research_v1'
    cfg['preference_acceptance']=dict(spatial_bin_width_m=.5,endpoint_path_rmse_m=.15,
        endpoint_speed_rmse_m_s=.10,min_speed_gap_m_s=.02,min_path_gap_m=.001,
        final_hold_s=.5,final_speed_error_m_s=.15,final_path_error_m=.20,
        final_heading_error_rad=.10,adjacent_monotonicity_role='diagnostic',
        adjacent_speed_reverse_m_s=.01,adjacent_path_reverse_m=.01,
        roll_limit_role='diagnostic',corridor_limit_role='diagnostic')
    return cfg


def research_panel():
    p=preferred()
    for alpha in selection.CONDITIONAL_ALPHAS:
        for d in p[selection.alpha_key(alpha)].values():
            turn=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
            d['actual_forward_speed'][turn]=d['v_user'][turn]+.12-.04*alpha
            d['path_cross_track'][turn]=.12+.01*alpha
            d['actual_forward_speed'][-100:]=d['v_user'][-100:]+.12
            d['path_cross_track'][-100:]=.15
            d['path_heading_error'][-100:]=.08
    return p


def test_research_acceptance_qualifies_relaxed_endpoints_and_hold_keeps_legacy_failure():
    s=selection.summarize_preference(research_panel(),config())
    assert s['qualified_research_preference'] and s['qualified_preference']
    assert s['qualification_role']=='fixed_alpha_research_behavior'
    assert not s['qualified_task'] and not s['legacy_strict_preference_qualified']
    assert s['research_goal_hold_failures']==0 and s['goal_hold_failures']==30
    for case in s['preference_cases'].values():
        assert case['D_v_m_s']==pytest.approx(.04)
        assert case['D_y_m']==pytest.approx(.01)
        assert not case['legacy_strong_gap_qualified']
    assert len(s['score_tuple'])==8


def test_interior_reversals_and_legacy_limits_cannot_change_research_ranking():
    p=research_panel();before=selection.summarize_preference(p,config())
    d=p['alpha_0p5']['right90_R2_2.6'];mask=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
    d['actual_forward_speed'][mask]=d['v_user'][mask]+.16
    d['peak_roll'][20]=.35;d['path_cross_track'][20]=.5
    after=selection.summarize_preference(p,config())
    assert after['qualified_research_preference']
    assert before['score_tuple']==after['score_tuple']
    assert max(after['preference_cases']['right90_R2_2.6']['adjacent_speed_reverse_changes_m_s'])>.01


@pytest.mark.parametrize('field,value', [('actual_forward_speed',.13),('path_cross_track',.1205)])
def test_bad_endpoint_or_direction_blocks_research(field,value):
    p=research_panel();d=p['alpha_1']['left90_R2_2.6'];mask=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
    d[field][mask]=d['v_user'][mask]+value if field=='actual_forward_speed' else value
    s=selection.summarize_preference(p,config())
    assert not s['qualified_research_preference'] and s['preference_direction_failures']>0


def test_missing_middle_turn_bins_keep_null_and_fail():
    p=research_panel();p['alpha_0p5']['left90_R2_2.6']['path_progress'][:]=20
    s=selection.summarize_preference(p,config())
    assert not s['qualified_research_preference'] and s['incomplete_turn_conditions']==1
    assert s['preference_cases']['left90_R2_2.6']['per_alpha']['alpha_0p5']['speed_rmse_m_s'] is None
    assert np.isfinite(s['score_tuple']).all()


def test_hold_is_100_ticks_anchored_to_original_endpoint_not_failure_tail():
    cfg=config();acceptance=selection.resolve_preference_acceptance(cfg)
    d=research_panel()['alpha_0']['left90_R2_2.6']
    m=selection.preference_hold_metrics(d,cfg,acceptance)
    assert m['final_hold'] and m['hold_lower_ticks']==100 and m['hold_window_s']==.5
    early={k:v[:100].copy() for k,v in d.items()}
    early['physical_failure'][-1]=True
    m=selection.preference_hold_metrics(early,cfg,acceptance)
    assert not m['final_hold'] and m['hold_lower_ticks']==0
    assert m['final_hold_max_speed_error_m_s'] is None
    for broken in ( {k:v[:-1] for k,v in d.items()},deepcopy(d)):
        if len(broken['time'])==4000:broken['path_heading_error'][-1]=np.nan
        assert not selection.preference_hold_metrics(broken,cfg,acceptance)['final_hold']


@pytest.mark.parametrize('change', [('unknown',1),('min_path_gap_m',0),('final_hold_s',.5001),
    ('endpoint_path_rmse_m',float('nan')),('adjacent_monotonicity_role','hard')])
def test_acceptance_rejects_unknown_invalid_and_nonintegral_windows(change):
    cfg=config();cfg['preference_acceptance'][change[0]]=change[1]
    with pytest.raises(ValueError):selection.resolve_preference_acceptance(cfg)


def test_schema_is_explicit_and_acceptance_hash_changes_with_threshold():
    cfg=config();s=selection.summarize_preference(research_panel(),cfg)
    cfg['preference_acceptance']['final_path_error_m']=.19
    changed=selection.summarize_preference(research_panel(),cfg)
    assert changed['acceptance_key']!=s['acceptance_key']
    cfg['preference_selection_schema']='unknown'
    with pytest.raises(ValueError):selection.resolve_preference_acceptance(cfg)
    old=selection.resolve_preference_acceptance(strong_config())
    assert old['final_hold_s']==1 and old['endpoint_speed_rmse_m_s']==.05


def test_nonfinite_or_failure_padding_is_rejected_and_real_early_failure_ranked():
    p=research_panel();d=p['alpha_0']['left90_R2_2.6']
    d['physical_failure'][100]=True
    with pytest.raises(ValueError,match='terminal'):selection.summarize_preference(p,config())
    d['physical_failure'][100]=False;d['path_cross_track'][100]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):selection.summarize_preference(p,config())
    p=research_panel();d=p['alpha_0']['left90_R2_2.6']
    p['alpha_0']['left90_R2_2.6']={k:v[:100].copy() for k,v in d.items()}
    p['alpha_0']['left90_R2_2.6']['domain_exit'][-1]=True
    s=selection.summarize_preference(p,config())
    assert not s['qualified_research_preference'] and s['tracking_domain_failures']==1


def test_research_best_checks_and_persists_acceptance_identity(tmp_path):
    import pickle
    import jax
    import jax.numpy as j
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.preference_best import atomic_json
    from sttw_control.path_command_policy import config_key
    cfg=config();summary=selection.summarize_preference(research_panel(),cfg)
    params=make_path_actor(cfg).init(jax.random.PRNGKey(9),j.zeros(351))
    actor=tmp_path/'actor.pkl';actor.write_bytes(pickle.dumps(params))
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'learner')
    folder=tmp_path/'validation/update_0050';folder.mkdir(parents=True)
    acceptance_sha=summary['acceptance_key']
    identity=dict(policy_mode='shared_conditional_v1',seed=42,architecture='alpha_modulated_v1',
        reward_version='path_safety_preference_v2',config_key=config_key(cfg),
        acceptance_key=acceptance_sha,source_update=50,validation_dir=str(folder))
    source={k:v for k,v in identity.items() if k!='validation_dir'}
    source.update(checkpoint_sha256=None,actor_sha256=None)
    atomic_json(folder/'source.json',source)
    meta=dict(schema='path_conditional_panel_v1',completed=True,mode='shared_conditional_v1',
        alphas=summary['alphas'],architecture=identity['architecture'],reward_version=identity['reward_version'],
        config_key=identity['config_key'],acceptance_key=acceptance_sha,
        actor_parameters_sha256=None)
    atomic_json(folder/'panel_identity.json',meta)
    best=selection.persist_preference_best(tmp_path,checkpoint,actor,summary,identity)
    assert best['selection_schema']=='path_preference_research_v1'
    assert best['acceptance']==summary['acceptance'] and best['acceptance_key']==acceptance_sha
    assert best['qualified_research_preference'] and not best['legacy_strict_preference_qualified']
    # A changed threshold cannot inherit the prior pointer's score, even with unchanged weights.
    changed=deepcopy(cfg);changed['preference_acceptance']['final_path_error_m']=.19
    other=selection.summarize_preference(research_panel(),changed)
    with pytest.raises(ValueError,match='acceptance'):
        selection.persist_preference_best(tmp_path,checkpoint,actor,other,identity)
    meta.pop('acceptance_key');atomic_json(folder/'panel_identity.json',meta)
    with pytest.raises(ValueError,match='acceptance'):
        selection.persist_preference_best(tmp_path,checkpoint,actor,summary,identity)


def test_missing_interior_coverage_retains_endpoint_error_ranking():
    p=research_panel()
    p['alpha_0p5']['left90_R2_2.6']['path_progress'][:]=20
    better=selection.summarize_preference(p,config())
    worse_panel=deepcopy(p)
    for name in ('left90_R2_2.6','right90_R2_2.6'):
        d=worse_panel['alpha_0'][name]
        turn=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
        d['path_cross_track'][turn]=.125
    worse=selection.summarize_preference(worse_panel,config())
    assert better['incomplete_turn_conditions']==worse['incomplete_turn_conditions']==1
    assert better['worst_endpoint_normalized_error']==pytest.approx(.8)
    assert worse['worst_endpoint_normalized_error']==pytest.approx(.125/.15)
    assert selection.preference_score_key(better,100)<selection.preference_score_key(worse,50)
    p['alpha_0']['left90_R2_2.6']['path_progress'][:]=20
    endpoint_missing=selection.summarize_preference(p,config())
    assert endpoint_missing['worst_endpoint_normalized_error']==np.finfo(np.float32).max
