"""Conditional panels must preserve alpha identity and select one shared model."""
from copy import deepcopy
from pathlib import Path
import json
import pickle
import jax
import jax.numpy as j
from sttw_control.path_command_policy import make_path_actor
import numpy as np
import pytest
from test_path_task_selection import CFG, panel
from sttw_control import path_command_selection as selection
from sttw_control import path_command_evaluation as evaluation

ALPHAS=(0.,.25,.5,.75,1.)
KEYS=('alpha_0','alpha_0p25','alpha_0p5','alpha_0p75','alpha_1')

def config():
    c=deepcopy(CFG);c['task_selection_schema']='path_conditional_task_v1'
    c['alpha_conditioning']={'mode':'shared_conditional_v1','evaluation_alphas':list(ALPHAS)}
    return c

def actor_parameters():
    return make_path_actor(config()).init(jax.random.PRNGKey(9),j.zeros(351))

def save_actor(path):
    path.write_bytes(pickle.dumps(actor_parameters()))

def panels():
    result={}
    for key,a in zip(KEYS,ALPHAS):
        traces=panel()
        for d in traces.values():d['upper_alpha']=np.full(len(d['time']),a)
        result[key]=traces
    return result

def test_conditional_middle_does_not_inherit_endpoint_speed_gate():
    p=panels()
    for key in KEYS[1:-1]:
        for d in p[key].values():d['actual_forward_speed'][:]=2.08
    summary=selection.summarize_conditional(p,config())
    assert summary['declared_episodes']==30
    assert summary['endpoint_primary_failures']==0
    assert summary['qualified_task']
    assert summary['per_alpha']['alpha_0p5']['cases']['left90_R2_2.6']['primary_band_failure'] is None
    assert summary['per_alpha']['alpha_0p5']['cases']['left90_R2_2.6']['weighted_normalized_error']==pytest.approx(.85)
    p['alpha_0p25']['left90_R2_2.6']['peak_roll'][0]=.32
    assert not selection.summarize_conditional(p,config())['qualified_task']
    assert selection.summarize_conditional(p,config())['working_failures']==1
    p['alpha_1']['right90_R2_2.6']['actual_forward_speed'][:]=2.08
    assert selection.summarize_conditional(p,config())['endpoint_primary_failures']==1

def test_conditional_rejects_missing_or_mislabeled_alpha_and_preserves_old_summary():
    p=panels();p.pop('alpha_0p25')
    with pytest.raises(ValueError,match='alpha'):selection.summarize_conditional(p,config())
    p=panels();p['alpha_0p25']['left90_R2_2.6']['upper_alpha'][:]=.75
    with pytest.raises(ValueError,match='alpha'):selection.summarize_conditional(p,config())
    for a,k in zip(ALPHAS,KEYS):
        assert selection.alpha_key(a)==k
        assert selection.alpha_value(k)==a
    for key in ('alpha_0.25','alpha_00','alpha_0p250','alpha_nan'):
        with pytest.raises(ValueError):selection.alpha_value(key)
    old=selection.summarize_task(panel(),0,CFG)
    assert old['selection_schema']=='path_task_best_v2'
    assert old['qualified_task']

def test_conditional_severity_uses_all_alpha_cases():
    first=panels();second=panels()
    first['alpha_0p25']['left90_R2_2.6']['peak_roll'][0]=.32
    second['alpha_0p75']['left90_R2_2.6']['peak_roll'][:20]=.31
    a=selection.summarize_conditional(first,config());b=selection.summarize_conditional(second,config())
    assert b['score_tuple']<a['score_tuple']
    assert b['roll_exceedance_seconds']==pytest.approx(.1)

class SavedEvaluator:
    """Avoid external simulator; preserve the real evaluator's disk side effects."""
    config=config()
    def evaluate(self,params,alpha,output):
        out=Path(output);out.mkdir()
        data=panel()
        for name,d in data.items():
            d['upper_alpha']=np.full(len(d['time']),alpha)
            np.savez_compressed(out/f'{name}.npz',**d)
            np.savez_compressed(out/f'{name}_path.npz',s=[0,20],xy=[[0,0],[20,0]],curvature=[0,0],turn_end=0.)
        return data

def test_evaluate_conditional_writes_unambiguous_grid_and_rejects_duplicate(tmp_path):
    result=evaluation.evaluate_conditional(SavedEvaluator(),actor_parameters(),ALPHAS,tmp_path/'panel')
    assert tuple(result)==KEYS
    meta=json.loads((tmp_path/'panel/panel_identity.json').read_text())
    assert meta['alphas']==dict(zip(KEYS,ALPHAS))
    assert meta['completed'] and meta['actor_parameters_sha256'] is None
    assert (tmp_path/'panel/alpha_0p25/left90_R2_2.6.npz').exists()
    with pytest.raises(ValueError,match='alpha'):
        evaluation.evaluate_conditional(SavedEvaluator(),{},[0,.5,.5,1],tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

def test_one_conditional_best_and_final_reuse_preserve_all_alphas(tmp_path):
    ck=tmp_path/'checkpoint.pt';ck.write_bytes(b'learner')
    actor=tmp_path/'actor.pkl';save_actor(actor)
    out=tmp_path/'validation/update_0050'
    p=evaluation.evaluate_conditional(SavedEvaluator(),actor_parameters(),ALPHAS,out)
    summary=selection.summarize_conditional(p,config())
    source=dict(mode='shared_conditional_v1',seed=42,source_update=50,checkpoint=str(ck),actor=str(actor),checkpoint_sha256=None,actor_sha256=None)
    (out/'source.json').write_text(json.dumps(source))
    identity=dict(mode='shared_conditional_v1',source_update=50,seed=42,validation_dir=str(out))
    (tmp_path/'best_task_model.json').write_text('historical')
    best=selection.persist_conditional_best(tmp_path,ck,actor,summary,identity)
    assert best['selection_schema']=='path_conditional_task_v1'
    assert (tmp_path/'best_task_model.json').read_text()=='historical'
    assert (tmp_path/'best_conditional_model.json').exists()
    reused=evaluation.reuse_conditional_best(tmp_path,best,config())
    assert tuple(reused)==KEYS
    assert json.loads((tmp_path/'final_best/source.json').read_text())['actor_sha256']==best['actor_sha256']
    for key in KEYS:
        np.testing.assert_array_equal(reused[key]['left90_R2_2.6']['upper_alpha'],float(selection.alpha_value(key)))
    source['source_update']=49;(out/'source.json').write_text(json.dumps(source))
    with pytest.raises(ValueError,match='source|SHA|identity'):
        evaluation.reuse_conditional_best(tmp_path,best,config())


def report_run(tmp_path):
    """Exercise the training layout and evaluate -> select -> reuse handoff."""
    from sttw_control.path_command_reporting import reward_series
    cfg=config();learner=tmp_path/'shared';learner.mkdir()
    class ReportingEvaluator(SavedEvaluator):
        config=cfg
        def evaluate(self,params,alpha,output):
            data=super().evaluate(params,alpha,output);dest=Path(output)
            for name,d in data.items():
                n=len(d['time']);z=np.zeros((n,2));progress=np.linspace(0,20,n)
                d.update(phi=np.zeros(n),phi_dot=np.zeros(n),offsets=z.copy(),actual_delta=np.zeros(n),
                         actual_xy=np.column_stack([progress,np.full(n,.01)]),path_progress=progress,
                         nominal=np.column_stack([np.full(n,2.),np.zeros(n)]),governed=np.column_stack([np.full(n,2.01),np.zeros(n)]),
                         target_offset=z.copy(),filtered_offset=z.copy(),final_command=z.copy(),wheel_speed_proxy=np.full(n,2.))
                d['scored_tick_reward']=reward_series(d,alpha,cfg)[0]
                np.savez_compressed(dest/f'{name}.npz',**d)
                path=dict(s=np.array([0.,3.,4.,20.]),xy=np.array([[0.,0.],[3.,0.],[4.,0.],[20.,0.]]),
                          curvature=np.array([0.,1.,1.,0.]),turn_end=4.,goal=10.)
                np.savez_compressed(dest/f'{name}_path.npz',**path)
                if alpha==0:
                    zero=tmp_path/'zero_upper';zero.mkdir(exist_ok=True)
                    np.savez_compressed(zero/f'{name}.npz',**d)
                    np.savez_compressed(zero/f'{name}_path.npz',**path)
            return data
    checkpoint=learner/'model.pt';checkpoint.write_bytes(b'learner')
    actor=learner/'actor.pkl';save_actor(actor)
    destination=learner/'validation/update_0050'
    p=evaluation.evaluate_conditional(ReportingEvaluator(),actor_parameters(),ALPHAS,destination)
    identity=dict(policy_mode='shared_conditional_v1',seed=42)
    provenance=dict(identity,source_update=50,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=None,
                    actor=str(actor.resolve()),actor_sha256=None,validation_dir=str(destination.resolve()))
    (destination/'source.json').write_text(json.dumps(provenance))
    best=selection.persist_conditional_best(learner,checkpoint,actor,selection.summarize_conditional(p,cfg),
        dict(identity,source_update=50,validation_dir=str(destination.resolve())))
    evaluation.reuse_conditional_best(learner,best,cfg)
    return cfg


def test_conditional_report_covers_all_alphas_and_rebuilds_rewards(tmp_path):
    from sttw_control import path_command_reporting as reporting
    cfg=report_run(tmp_path)
    result=reporting.report_conditional_panel(tmp_path,cfg)
    assert result['summary']['declared_episodes']==30
    assert set(result['cases'])==set(panel())
    assert set(result['cases']['left90_R2_2.6']['alphas'])==set(KEYS)
    assert result['preference']['qualified'] is None
    assert 'joint improvement' in result['preference']['role']
    assert result['reward_audit']['checked_policy_traces']==30
    assert result['reward_audit']['max_abs_tick_error']<1e-12
    assert (tmp_path/'report/ALL_XY.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    for name in panel():
        for suffix in ('_XY.png','_physical.png','_reward.png','_components.png','_command_chain.png'):
            assert (tmp_path/'report'/f'{name}{suffix}').exists()
    trace=tmp_path/'shared/final_best/alpha_0p25/left90_R2_2.6.npz'
    d=dict(np.load(trace));d['scored_tick_reward'][0]+=.1;np.savez_compressed(trace,**d)
    with pytest.raises(ValueError,match='reward'):
        reporting.report_conditional_panel(tmp_path,cfg)


def test_conditional_best_rejects_protocol_change_even_without_caller_hash(tmp_path):
    ck=tmp_path/'checkpoint.pt';ck.write_bytes(b'learner')
    actor=tmp_path/'actor.pkl';save_actor(actor)
    def candidate(update,cfg):
        evaluator=SavedEvaluator();evaluator.config=cfg
        out=tmp_path/f'validation/update_{update:04d}'
        p=evaluation.evaluate_conditional(evaluator,actor_parameters(),ALPHAS,out)
        source=dict(mode='shared_conditional_v1',source_update=update,checkpoint_sha256=None,actor_sha256=None)
        (out/'source.json').write_text(json.dumps(source))
        return selection.summarize_conditional(p,cfg),dict(mode='shared_conditional_v1',source_update=update,validation_dir=str(out))
    summary,identity=candidate(0,config());selection.persist_conditional_best(tmp_path,ck,actor,summary,identity)
    different=config();different['path_reward']['scale']*=2
    summary,identity=candidate(50,different)
    with pytest.raises(ValueError,match='protocol|identity'):
        selection.persist_conditional_best(tmp_path,ck,actor,summary,identity)


def test_conditional_source_cannot_relabel_learner_even_with_same_weight_hashes(tmp_path):
    ck=tmp_path/'checkpoint.pt';ck.write_bytes(b'learner')
    actor=tmp_path/'actor.pkl';save_actor(actor)
    out=tmp_path/'validation/update_0050'
    p=evaluation.evaluate_conditional(SavedEvaluator(),actor_parameters(),ALPHAS,out)
    source=dict(mode='other_policy',seed=7,source_update=50,checkpoint_sha256=None,actor_sha256=None)
    (out/'source.json').write_text(json.dumps(source))
    with pytest.raises(ValueError,match='source.*identity'):
        selection.persist_conditional_best(tmp_path,ck,actor,selection.summarize_conditional(p,config()),
            dict(mode='shared_conditional_v1',seed=42,source_update=50,validation_dir=str(out)))


def test_report_rejects_modified_final_physics_even_when_reward_is_rescored(tmp_path):
    from sttw_control import path_command_reporting as reporting
    cfg=report_run(tmp_path)
    trace=tmp_path/'shared/final_best/alpha_0p25/left90_R2_2.6.npz'
    d=dict(np.load(trace));d['peak_roll'][0]=.32
    d['scored_tick_reward']=reporting.reward_series(d,.25,cfg)[0]
    np.savez_compressed(trace,**d)
    with pytest.raises(ValueError,match='selected.*metrics|metrics.*selected'):
        reporting.report_conditional_panel(tmp_path,cfg)


def test_research_panel_pins_acceptance_identity(tmp_path):
    from test_path_dynamic_preference import _research_config
    cfg=_research_config()
    evaluator=SavedEvaluator();evaluator.config=cfg
    params=make_path_actor(cfg).init(jax.random.PRNGKey(9),j.zeros(351))
    evaluation.evaluate_conditional(evaluator,params,ALPHAS,tmp_path/'panel')
    meta=json.loads((tmp_path/'panel/panel_identity.json').read_text())
    from sttw_control.path_command_policy import config_key
    assert meta['acceptance_key']==config_key(selection.resolve_preference_acceptance(cfg))


def test_research_report_uses_research_pointer_and_documents_diagnostic_monotonicity(tmp_path,monkeypatch):
    import test_path_strong_selection as strong
    from test_path_dynamic_preference import _research_config
    from sttw_control.path_command_policy import config_key
    from sttw_control.path_command_reporting import report_conditional_panel
    cfg=_research_config();sha=config_key(selection.resolve_preference_acceptance(cfg))
    monkeypatch.setattr(strong,'config',lambda:cfg)
    persist=selection.persist_preference_best
    def research_candidate(root,checkpoint,actor,summary,identity):
        identity=dict(identity,acceptance_key=sha)
        source=Path(identity['validation_dir'])/'source.json'
        metadata=json.loads(source.read_text());metadata['acceptance_key']=sha
        source.write_text(json.dumps(metadata))
        return persist(root,checkpoint,actor,summary,identity)
    monkeypatch.setattr(selection,'persist_preference_best',research_candidate)
    cfg,best,_=strong.strong_saved_run(tmp_path)
    result=report_conditional_panel(tmp_path,cfg)
    assert result['model']['selection_schema']=='path_preference_research_v1'
    assert result['preference']['qualified_research_preference']
    assert result['preference']['acceptance_key']==sha
    text=(tmp_path/'report/INDEX.md').read_text()
    assert 'Adjacent-alpha monotonicity role: diagnostic' in text
    assert 'Legacy strict preference diagnostic:' in text
    assert 'Final 0.5 s hold:' in text


def test_last_receipt_is_unknown_when_missing_and_rebuilt_when_saved(tmp_path):
    from sttw_control.path_command_reporting import _conditional_last_receipt
    cfg=report_run(tmp_path);learner=tmp_path/'shared'
    best=json.loads((learner/'best_conditional_model.json').read_text())
    missing=_conditional_last_receipt(learner,cfg,best)
    assert missing['state']=='UNKNOWN' and missing['source_update'] is None
    (learner/'last_completed.json').write_text(json.dumps(dict(update=50,checkpoint=best['checkpoint'],checkpoint_sha256=best['checkpoint_sha256'])))
    last=_conditional_last_receipt(learner,cfg,best)
    assert last['state']=='verified_saved_panel' and last['source_update']==50
    assert last['summary']['qualified_task']
    source_file=Path(best['validation_dir'])/'source.json'
    data=json.loads(source_file.read_text());data['source_update']=49
    source_file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='last.*identity'):_conditional_last_receipt(learner,cfg,best)
