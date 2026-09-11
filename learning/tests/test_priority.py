from dataclasses import asdict,replace
import numpy as np
import jax.numpy as jp
from sttw_control.priority import PriorityConfig,priority_weights


def test_priority_weights_and_risk_preserve_both_objectives():
    c=PriorityConfig()
    low=priority_weights(0.,0.,0.,0.,c);high=priority_weights(1.,0.,0.,0.,c)
    assert low[1]<high[1] and low[2]>high[2]
    neutral=priority_weights(.5,0.,0.,0.,c)
    np.testing.assert_allclose(neutral,[0.,1.,1.,1.])
    danger=priority_weights(.5,.6,.6,2.,c)
    assert danger[0]==1 and danger[3]>neutral[3]
    assert 0<danger[1]<neutral[1] and 0<danger[2]<neutral[2]


def test_priority_reset_identity_and_zero_action_cpu():
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.observation import ObservationConfig
    from sttw_control.network import make_policy_identity
    from sttw_control.training import normalization
    c=load_config('learning/configs/disturbance_learning.json')
    p=replace(c,priority=PriorityConfig(),observation=replace(c.observation,include_priority=True))
    a=RecoveryEnv(c);b=RecoveryEnv(p);x=a.reset(7);y=b.reset(7)
    assert y.obs.shape==(21,)
    assert normalization(p)[0].shape==y.obs.shape
    initial=np.asarray(y.obs).copy()
    for _ in range(3):
        x=a.step(x,jp.zeros(2));y=b.step(y,jp.zeros(2))
        np.testing.assert_array_equal(x.data.qpos,y.data.qpos)
    np.testing.assert_array_equal(b.reset(7).obs,initial)
    ident=make_policy_identity(b.bundle.identity,asdict(p),1)
    assert ident['observation_fields'][-2:]==['speed_priority','attitude_risk']


def test_alpha_intervention_preserves_physics_and_history(tmp_path):
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.evaluation import evaluate
    from sttw_control.network import ResidualActor,make_policy_identity,save_policy,load_policy
    from sttw_control.training import normalization
    import jax
    c=load_config('learning/configs/priority_conditioned_learning.json')
    c=replace(c,horizon_seconds=.01,random_events=None,observation=replace(c.observation,history_steps=3))
    e=RecoveryEnv(c);s=e.reset(8);t=e.set_priority(s,1.)
    np.testing.assert_array_equal(t.history.frames[:-1],s.history.frames[:-1])
    np.testing.assert_array_equal(t.data.qpos,s.data.qpos)
    assert t.priority_alpha==1 and t.history.frames[-1,-2]==1
    actor=ResidualActor();params=actor.init(jax.random.PRNGKey(1),s.obs)
    ident=make_policy_identity(e.bundle.identity,asdict(c),3);mean,std=normalization(c)
    save_policy(tmp_path/'policy',params,mean,std,ident)
    policy=load_policy(tmp_path/'policy',expected=ident)
    assert not np.allclose(policy(e.set_priority(s,0.).obs),policy(t.obs))
    evaluate(e,tmp_path/'trace',seed=8,priority_alpha=0.)
    tr=np.load(tmp_path/'trace/trace.npz')
    np.testing.assert_array_equal(tr['priority_alpha'],0.)
    assert np.all((tr['attitude_risk']>=0)&(tr['attitude_risk']<=1))


def test_conditioned_panel_roundtrips_json_checkpoint_and_fixed_alpha(tmp_path):
    import importlib.util,json,jax
    from pathlib import Path
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.network import ResidualActor,make_policy_identity,save_policy
    from sttw_control.training import normalization
    c=replace(load_config('learning/configs/priority_conditioned_learning.json'),horizon_seconds=.01,random_events=None)
    task=tmp_path/'task.json';task.write_text(json.dumps({**asdict(c),'priority':{}}))
    train=tmp_path/'training';train.mkdir();(train/'declaration.json').write_text(json.dumps({'task':asdict(c)}))
    e=RecoveryEnv(c);s=e.reset(1);params=ResidualActor().init(jax.random.PRNGKey(0),s.obs)
    mean,std=normalization(c);checkpoint=tmp_path/'checkpoint'
    save_policy(checkpoint,params,mean,std,make_policy_identity(e.bundle.identity,asdict(c),1))
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'seed':1,'start_seconds':0.,'duration_seconds':.005,'path_tolerance_m':.2,'extra_tolerance_m':.05,'heading_tolerance_rad':.15,'hold_seconds':.005,'cases':[{'name':'force','force':1.,'duration':.005}]}))
    spec=importlib.util.spec_from_file_location('panel_entry',Path('learning/cli/disturbance.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'panel';module.run(task,panel,train,checkpoint,out,priority_alpha=1.)
    assert json.loads((out/'status.json').read_text())['episodes']==4
    for policy in ('baseline','residual'):
        tr=np.load(out/'force'/policy/'trace.npz')
        np.testing.assert_array_equal(tr['priority_alpha'],1.)
        assert json.loads((out/'force'/policy/'summary.json').read_text())['sampled_mechanical_work']['available']
