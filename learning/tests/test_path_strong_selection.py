from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from test_path_conditional_reporting import panels,KEYS,ALPHAS
from sttw_control import path_command_selection as selection


def config():
    return json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Strong_Preference.json').read_text())


def preferred():
    p=panels()
    for key,a in zip(KEYS,ALPHAS):
        for name,d in p[key].items():
            progress=np.linspace(0,20,len(d['time']));d['path_progress']=progress
            d['goal_progress'][:]=10
            turn=(progress>=3)&(progress<=6.6415926)
            d['actual_forward_speed'][:]=d['v_user']+.01;d['path_cross_track'][:]=.01
            d['actual_forward_speed'][turn]=d['v_user'][turn]+(.09-.07*a)
            d['path_cross_track'][turn]=.04+.04*a
    return p


def test_strong_acceptance_is_full_turn_and_old_safety_limits_are_diagnostic():
    p=preferred()
    p['alpha_1']['left90_R2_2.6']['peak_roll'][20]=.35
    s=selection.summarize_preference(p,config())
    assert s['qualified_preference'] and not s['qualified_task']
    assert s['selection_schema']=='path_strong_preference_v1'
    assert s['highspeed_endpoint_failures']==0 and s['preference_continuity_failures']==0
    for case in s['preference_cases'].values():
        assert case['D_v_m_s']==pytest.approx(.07)
        assert case['D_y_m']==pytest.approx(.04)
        assert all(x['complete'] for x in case['per_alpha'].values())


def test_good_gaps_do_not_qualify_when_endpoint_itself_is_bad():
    p=preferred();d=p['alpha_0']['left90_R2_2.6'];turn=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
    d['path_cross_track'][turn]=.11
    d=p['alpha_1']['left90_R2_2.6'];d['path_cross_track'][turn]=.15
    s=selection.summarize_preference(p,config())
    assert not s['qualified_preference'] and s['highspeed_endpoint_failures']==1
    assert s['preference_cases']['left90_R2_2.6']['D_y_m']==pytest.approx(.04)


def test_middle_reverse_change_above_tolerance_is_not_qualified():
    p=preferred();d=p['alpha_0p5']['right90_R2_2.6'];mask=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
    d['actual_forward_speed'][mask]=d['v_user'][mask]+.09
    s=selection.summarize_preference(p,config())
    assert not s['qualified_preference'] and s['preference_continuity_failures']>0


def test_missing_original_turn_bin_is_not_zero_filled_or_qualified():
    p=preferred();p['alpha_0p5']['left90_R2_2.6']['path_progress'][:]=20
    s=selection.summarize_preference(p,config())
    assert not s['qualified_preference']
    bins=s['preference_cases']['left90_R2_2.6']['per_alpha']['alpha_0p5']
    assert not bins['complete'] and bins['speed_rmse_m_s'] is None


def test_whole_shared_pointer_ranking_prefers_task_then_gaps_and_preserves_safety():
    p=preferred();good=selection.summarize_preference(p,config())
    same=deepcopy(p)
    for case in ['left90_R2_2.6','right90_R2_2.6']:
        d=same['alpha_1'][case];mask=(d['path_progress']>=3)&(d['path_progress']<=6.6415926)
        d['actual_forward_speed'][mask]=d['v_user'][mask]+.07
    weak=selection.summarize_preference(same,config())
    assert selection.preference_score_key(good,100)<selection.preference_score_key(weak,50)
    assert selection.preference_score_key(good,50)<selection.preference_score_key(good,100)
    p.pop('alpha_0p25')
    with pytest.raises(ValueError,match='alpha'):selection.summarize_preference(p,config())


def strong_saved_run(tmp_path):
    import pickle
    import jax
    import jax.numpy as j
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.path_command_reporting import reward_series
    from sttw_control.path_command_evaluation import evaluate_conditional,reuse_conditional_best
    from sttw_control.geometric_path import build_path
    cfg=config();data=preferred();learner=tmp_path/'shared';learner.mkdir()
    params=make_path_actor(cfg).init(jax.random.PRNGKey(9),j.zeros(351))
    actor=learner/'actor.pkl';actor.write_bytes(pickle.dumps(params));checkpoint=learner/'model.pt';checkpoint.write_bytes(b'learner')
    class Evaluator:
        config=cfg
        def evaluate(self,params,alpha,output):
            out=Path(output);out.mkdir();traces=deepcopy(data[selection.alpha_key(alpha)])
            for name,d in traces.items():
                route=next(route for case,route,_ in selection.cases(cfg) if case==name)
                path=build_path(route,np.zeros(3),cfg)
                p={k:np.asarray(getattr(path,k)) for k in ['s','xy','heading','curvature','turn_end','goal']}
                n=len(d['time']);z=np.zeros((n,2));progress=d['path_progress']
                d.update(phi=np.zeros(n),phi_dot=np.zeros(n),offsets=z.copy(),actual_delta=np.zeros(n),
                    actual_xy=np.column_stack([np.interp(progress,p['s'],p['xy'][:,k]) for k in [0,1]]),
                    nominal=np.column_stack([d['v_user'],np.zeros(n)]),governed=np.column_stack([d['v_user'],np.zeros(n)]),
                    target_offset=z.copy(),filtered_offset=z.copy(),final_command=z.copy(),wheel_speed_proxy=d['actual_forward_speed'].copy())
                d['scored_tick_reward']=reward_series(d,alpha,cfg)[0]
                np.savez_compressed(out/f'{name}.npz',**d);np.savez_compressed(out/f'{name}_path.npz',**p)
                if alpha==0:
                    base=tmp_path/'zero_upper';base.mkdir(exist_ok=True)
                    np.savez_compressed(base/f'{name}.npz',**d);np.savez_compressed(base/f'{name}_path.npz',**p)
            return traces
    folder=learner/'validation/update_0050'
    panels=evaluate_conditional(Evaluator(),params,ALPHAS,folder)
    identity=dict(policy_mode='shared_conditional_v1',seed=42,architecture='alpha_modulated_v1',
        reward_version='path_safety_preference_v2')
    from sttw_control.preference_best import config_hash,atomic_json
    identity['config_sha256']=config_hash(cfg)
    source=dict(identity,source_update=50,checkpoint=str(checkpoint),actor=str(actor),
        checkpoint_sha256=selection.digest(checkpoint),actor_sha256=selection.digest(actor))
    atomic_json(folder/'source.json',source)
    selection.persist_conditional_best(learner,checkpoint,actor,selection.summarize_conditional(panels,cfg),dict(identity,source_update=50,validation_dir=str(folder)))
    best=selection.persist_preference_best(learner,checkpoint,actor,selection.summarize_preference(panels,cfg),dict(identity,source_update=50,validation_dir=str(folder)))
    reuse_conditional_best(learner,best,cfg)
    return cfg,best,folder


def test_one_preference_pointer_reuse_and_report_keep_old_diagnostic_pointer(tmp_path):
    from sttw_control.path_command_reporting import report_conditional_panel
    cfg,best,folder=strong_saved_run(tmp_path)
    assert best['qualified_preference'] and best['score_tuple'][-1]==50
    assert (tmp_path/'shared/best_conditional_model.json').exists()
    result=report_conditional_panel(tmp_path,cfg)
    assert result['model']['selection_schema']=='path_strong_preference_v1'
    assert result['preference']['qualified']
    assert result['reward_audit']['checked_policy_traces']==30
    assert (tmp_path/'report/ALL_XY.png').exists()


def test_preference_pointer_refuses_wrong_reward_architecture_or_parameter_identity(tmp_path):
    cfg,best,folder=strong_saved_run(tmp_path)
    source=json.loads((folder/'source.json').read_text())
    identity={k:v for k,v in source.items() if k not in ['actor','checkpoint','actor_sha256','checkpoint_sha256']}
    identity['validation_dir']=str(folder)
    identity['reward_version']='path_safety_preference_v1'
    with pytest.raises(ValueError,match='identity|reward'):
        selection.persist_preference_best(tmp_path/'shared',best['checkpoint'],best['actor'],best['metrics'],identity)
