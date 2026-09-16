"""Real CPU physics/audit integration. MJX test is explicitly enabled in CI."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import numpy as np
import pytest
pytest.importorskip('flax')
pytest.importorskip('mujoco')
from sttw_control.env import RecoveryEnv, load_config
from sttw_control.evaluation import evaluate
from sttw_control.network import make_policy_identity
from sttw_control.observation import observation_fields
from sttw_control.tracking_diagnostics import audit_trace, rescore_trace


def config(short=False):
    c=load_config('learning/configs/geometric_reward_recovery.json')
    if short:
        c=replace(c,horizon_seconds=.025,random_events=None,
                  timed_reference=replace(c.timed_reference,fixed=((0.,2.1,0.),)))
    return c


def test_geometric_reset_has_alpha_not_timed_lag_and_fixed_curve():
    env=RecoveryEnv(config());s=env.reset(12);t=env.reset(13)
    assert s.obs.shape==(280,)
    fields=observation_fields(env.config.observation)
    assert 'speed_priority' in fields and 'longitudinal_error' not in fields
    assert s.reference_geometry.shape==(301,5)
    assert not np.array_equal(s.reference_geometry,t.reference_geometry)
    assert np.all(np.diff(np.asarray(s.reference_geometry)[:,0])>0)


def test_same_zero_residual_baseline_physics_for_all_alpha():
    env=RecoveryEnv(config(short=True));initial=env.reset(12)
    states=[env.set_priority(initial,a) for a in (0.,.5,1.)]
    table=np.asarray(initial.reference_geometry).copy()
    for _ in range(5):
        states=[env.step(s,np.zeros(2)) for s in states]
        for s in states:
            np.testing.assert_array_equal(s.reference_geometry,table)
            np.testing.assert_array_equal(s.data.qpos,states[0].data.qpos)
            np.testing.assert_array_equal(s.base,states[0].base)
            assert float(s.reward)==pytest.approx(sum(float(v) for v in s.tracking_components.values()),abs=1e-7)


def test_physics_trace_independent_projection_audit_and_baseline_rescore(tmp_path):
    env=RecoveryEnv(config(short=True))
    summary=evaluate(env,tmp_path/'baseline',seed=12,priority_alpha=0.)
    a=audit_trace(tmp_path/'baseline')
    b=rescore_trace(a,1.)
    assert summary['transitions']==5
    assert a['max_reference_error']<.002
    assert b['summary']['reference_mode'].startswith('fixed geometric')
    np.testing.assert_array_equal(a['trace']['qpos'],b['trace']['qpos'])
    np.testing.assert_array_equal(a['trace']['reference_geometry'],b['trace']['reference_geometry'])
    assert np.all(a['trace']['priority_alpha']==0)  # immutable source audit
    assert np.all(b['trace']['priority_alpha']==1)


def test_legacy_identity_defaults_are_removed_new_mode_is_not():
    c=load_config('learning/configs/timed_reference_recovery.json')
    full=asdict(c);old=json.loads(json.dumps(full))
    for k in ('mode','geometry_stride','projection_margin','extension_seconds'):old['timed_reference'].pop(k)
    for k in ('objective','deadline_penalty','overdue_rate'):old['tracking'].pop(k)
    assert make_policy_identity({},full,10)==make_policy_identity({},old,10)
    assert make_policy_identity({},asdict(config()),10)!=make_policy_identity({},full,10)


def test_invalid_mixed_semantics_rejected():
    c=config()
    with pytest.raises(ValueError):replace(c,observation=replace(c.observation,include_timed=True))
    with pytest.raises(ValueError):replace(c,tracking=replace(c.tracking,objective='legacy'))


@pytest.mark.skipif(os.environ.get('STTW_TEST_MJX_CPU')!='1',reason='explicit MJX CPU engineering opt-in')
def test_actual_mjx_geometric_reset_and_control_step():
    import jax
    import jax.numpy as jp
    env=RecoveryEnv(config(short=True),backend='mjx')
    s=jax.jit(env.reset)(jax.random.PRNGKey(12))
    n=jax.jit(env.step)(s,jp.zeros(2))
    jax.block_until_ready(n.obs)
    assert int(n.tick)==1 and np.isfinite(n.obs).all()
    assert float(n.reward)==pytest.approx(sum(float(v) for v in n.tracking_components.values()),abs=1e-6)
    np.testing.assert_array_equal(n.reference_geometry,s.reference_geometry)


def test_short_review_publishes_zip_and_reuses_baseline(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jp
    import zipfile
    from sttw_control.network import ResidualActor, save_policy
    from sttw_control.training import normalization
    from sttw_control.tracking_diagnostics import review_checkpoint
    import sttw_control.evaluation as evaluation
    c=config(short=True)
    c=replace(c,tracking=replace(c.tracking,start_seconds=0.,hold_seconds=.005,return_seconds=.005))
    env=RecoveryEnv(c)
    actor=ResidualActor()
    params=jax.tree.map(jp.zeros_like,actor.init(jax.random.PRNGKey(0),jp.zeros(env.observation_size)))
    training=tmp_path/'training';training.mkdir()
    checkpoint=training/'checkpoints'/'zero'
    mean,std=normalization(c)
    save_policy(checkpoint,params,mean,std,make_policy_identity(env.bundle.identity,asdict(c),10))
    (training/'declaration.json').write_text(json.dumps({'task':asdict(c)}))
    (training/'status.json').write_text(json.dumps({'complete':True,'best_reward_checkpoint':str(checkpoint)}))
    panel=tmp_path/'panel.json'
    panel.write_text(json.dumps({'seed':12,'alphas':[0.,1.],'scenarios':[{
        'name':'engineering_short','commands':[[0.,2.1,0.]],'event':{'start':0.,'duration':.005}}]}))
    called=[];original=evaluation.evaluate
    def counted(*args,**kwargs):
        called.append(kwargs.get('policy') is None)
        return original(*args,**kwargs)
    monkeypatch.setattr(evaluation,'evaluate',counted)
    review=tmp_path/'review'
    result=review_checkpoint(training,panel,review)
    assert called==[True,False,False] and len(result)==2
    assert json.loads((review/'status.json').read_text())['complete']
    with zipfile.ZipFile(tmp_path/'review.zip') as z:
        assert 'review/cross_scores.csv' in z.namelist()
        assert 'review/engineering_short/alpha_trajectories.png' in z.namelist()
    assert len((review/'cross_scores.csv').read_text().splitlines())==5
