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
    assert t.priority_alpha==1 and t.history.frames[-1,-1]==1
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
    task=tmp_path/'task.json';task.write_text(json.dumps(asdict(c)))
    train=tmp_path/'training';train.mkdir();(train/'declaration.json').write_text(json.dumps({'task':asdict(c)}))
    e=RecoveryEnv(c);s=e.reset(1);params=ResidualActor().init(jax.random.PRNGKey(0),s.obs)
    mean,std=normalization(c);checkpoint=tmp_path/'checkpoint'
    save_policy(checkpoint,params,mean,std,make_policy_identity(e.bundle.identity,asdict(c),c.observation.history_steps))
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'seed':1,'start_seconds':0.,'duration_seconds':.005,'path_tolerance_m':.2,'extra_tolerance_m':.05,'heading_tolerance_rad':.15,'hold_seconds':.005,'cases':[{'name':'force','force':1.,'duration':.005}]}))
    spec=importlib.util.spec_from_file_location('panel_entry',Path('learning/cli/disturbance.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'panel';module.run(task,panel,train,checkpoint,out,priority_alpha=1.)
    assert json.loads((out/'status.json').read_text())['episodes']==4
    for policy in ('baseline','residual'):
        tr=np.load(out/'force'/policy/'trace.npz')
        np.testing.assert_array_equal(tr['priority_alpha'],1.)
        assert json.loads((out/'force'/policy/'summary.json').read_text())['sampled_mechanical_work']['available']


def test_risk_free_history_keeps_past_alpha_commands_and_reset():
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.observation import observation_fields
    from sttw_control.training import normalization
    c=load_config('learning/configs/priority_conditioned_learning.json')
    assert c.observation.history_steps==10
    assert not c.observation.include_attitude_risk and not c.priority.risk_gate
    fields=observation_fields(c.observation)
    assert 'attitude_risk' not in fields
    e=RecoveryEnv(c);s=e.reset(8);alpha_index=fields.index('speed_priority')
    assert s.obs.shape==(200,)==normalization(c)[0].shape
    alphas=[];commands=[]
    for i in range(12):
        alpha=i/12
        s=e.set_priority(s,alpha)
        alphas.append(alpha)
        commands.append(np.asarray(s.actuator.previous).copy())
        np.testing.assert_allclose(s.history.frames[-1,alpha_index],alpha)
        s=e.step(s,jp.array([.2,-.1]))
        n=min(len(alphas),9)
        np.testing.assert_allclose(s.history.frames[-n-1:-1,alpha_index],alphas[-n:],atol=1e-7)
        indices=[fields.index('previous_steer_command'),fields.index('previous_rear_command')]
        np.testing.assert_allclose(np.asarray(s.history.frames[-n-1:-1])[:,indices],commands[-n:],atol=1e-7)
    s=e.reset(8)
    np.testing.assert_array_equal(s.history.frames[:-1],0)
    assert np.count_nonzero(s.history.mask)==1
    low=priority_weights(.3,0.,0.,0.,c.priority)
    high=priority_weights(.3,1.,1.,4.,c.priority)
    np.testing.assert_array_equal(low,high)
    assert high[0]==0 and high[3]==1


def test_new_defaults_preserve_old_checkpoint_identity():
    from sttw_control.env import TaskConfig
    from sttw_control.network import make_policy_identity
    from sttw_control.observation import ObservationConfig
    c=TaskConfig()
    current=asdict(c);legacy=asdict(c)
    legacy['observation'].pop('include_attitude_risk')
    assert make_policy_identity({},current,1)==make_policy_identity({},legacy,1)
    from sttw_control.path import CircleConfig
    c=replace(c,circle=CircleConfig(),priority=PriorityConfig(),observation=ObservationConfig(include_path=True,include_priority=True))
    current=asdict(c);legacy=asdict(c)
    legacy['observation'].pop('include_attitude_risk');legacy['priority'].pop('risk_gate');legacy['priority'].pop('speed_cost_scale');legacy['priority'].pop('path_cost_scale')
    assert make_policy_identity({},current,1)==make_policy_identity({},legacy,1)


def test_tracking_cost_scales_apply_before_alpha_without_changing_physics():
    from sttw_control.env import RecoveryEnv,load_config
    c=load_config('learning/configs/priority_conditioned_learning.json')
    e=RecoveryEnv(c);s=e.set_priority(e.reset(8),.5)
    scaled=RecoveryEnv(replace(c,priority=replace(c.priority,speed_cost_scale=2*c.priority.speed_cost_scale,path_cost_scale=2*c.priority.path_cost_scale)))
    no_tracking=RecoveryEnv(replace(c,speed_error_weight=0.,path_error_weight=0.,heading_error_weight=0.,path_excess_weight=0.))
    args=(s,s.measurement,s.actuator,jp.zeros(2),False,True,1.9,s.pose.at[0].add(.6))
    r=e._advance(*args).reward;r2=scaled._advance(*args).reward;r0=no_tracking._advance(*args).reward
    np.testing.assert_allclose(r0-r,2*(r0-r2),rtol=1e-5)
    a=e.step(s,jp.zeros(2));b=scaled.step(s,jp.zeros(2))
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)


def test_alive_reward_only_changes_nonterminal_reward_and_preserves_identity():
    import pytest
    from sttw_control.env import RecoveryEnv,TaskConfig,load_config
    from sttw_control.network import make_policy_identity
    c=load_config('learning/configs/priority_conditioned_learning.json')
    e=RecoveryEnv(c);s=e.set_priority(e.reset(8),.5)
    low=RecoveryEnv(replace(c,alive_reward_rate=1.))
    args=(s,s.measurement,s.actuator,jp.zeros(2),False,True,2.1,s.pose)
    np.testing.assert_allclose(e._advance(*args).reward-low._advance(*args).reward,4*c.controller.dt,rtol=1e-5)
    failed=list(args);failed[4]=True
    assert e._advance(*failed).reward == low._advance(*failed).reward == -c.failure_penalty
    for value in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):replace(c,alive_reward_rate=value)
    current=asdict(TaskConfig());legacy=dict(current);legacy.pop('alive_reward_rate')
    assert make_policy_identity({},current,1)==make_policy_identity({},legacy,1)
    changed=dict(current,alive_reward_rate=5.)
    assert make_policy_identity({},changed,1)!=make_policy_identity({},legacy,1)


def test_priority_reward_revision_has_symmetric_stronger_endpoints():
    from sttw_control.env import load_config
    c=load_config('learning/configs/priority_conditioned_learning.json')
    assert (c.speed_error_weight,c.path_error_weight,c.heading_error_weight,c.path_excess_weight)==(100.,20.,1.,80.)
    w=priority_weights(1.,0.,0.,0.,c.priority)
    np.testing.assert_allclose(w[1]/w[2],100.,rtol=1e-6)
    np.testing.assert_allclose(priority_weights(.5,0.,0.,0.,c.priority),[0.,.1,.1,1.])
    np.testing.assert_allclose(priority_weights(0.,0.,0.,0.,c.priority)[1:3],w[1:3][::-1])


def test_fixed_learning_roll_reference_preserves_ecbc_and_binds_identity():
    import pytest
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.network import make_policy_identity
    c=load_config('learning/configs/priority_conditioned_learning.json')
    fixed=RecoveryEnv(c)
    dynamic=RecoveryEnv(replace(c,learning_roll_reference=None))
    s=fixed.reset(8);d=dynamic.reset(8)
    assert abs(float(s.reference)-np.deg2rad(6.84))<1e-7
    np.testing.assert_array_equal(s.base,d.base)
    np.testing.assert_array_equal(s.controller.eso,d.controller.eso)
    for state in [s,s.replace(measurement=s.measurement.at[5].multiply(.8),pose=s.pose.at[0].add(.5))]:
        args=(state.controller,state.actuator,state.history,state.measurement,state.tick,state.pose,state.priority_alpha)
        f=fixed._prepare(*args);g=dynamic._prepare(*args)
        np.testing.assert_array_equal(f[3],g[3])
        assert abs(float(f[4])-np.deg2rad(6.84))<1e-7
    np.testing.assert_allclose(s.history.frames[-1,0],s.measurement[0]-c.learning_roll_reference,atol=1e-7)
    args=(s,s.measurement,s.actuator,jp.zeros(2),False,True,2.1,s.pose)
    f=fixed._advance(*args);g=dynamic._advance(*args)
    expected=-c.controller.dt*10*((s.measurement[0]-f.reference)**2-(s.measurement[0]-g.reference)**2)
    np.testing.assert_allclose(f.reward-g.reward,expected,atol=1e-6)
    f=fixed.step(s,jp.zeros(2));g=dynamic.step(d,jp.zeros(2))
    np.testing.assert_array_equal(f.data.qpos,g.data.qpos)
    current=asdict(dynamic.config);legacy=dict(current);legacy.pop('learning_roll_reference')
    assert make_policy_identity({},current,10)==make_policy_identity({},legacy,10)
    assert make_policy_identity({},asdict(c),10)!=make_policy_identity({},legacy,10)
    for value in [float('nan'),float('inf'),c.roll_failure]:
        with pytest.raises(ValueError):replace(c,learning_roll_reference=value)


def test_exponential_priority_has_equal_ratio_spacing_and_legacy_default():
    c=PriorityConfig(risk_gate=False,weight_schedule='exponential',tracking_weight_scale=.1)
    ws=np.asarray([priority_weights(a,0.,0.,0.,c)[1:3] for a in [0,.5,1]])
    np.testing.assert_allclose(ws,[[.01,1],[.1,.1],[1,.01]],rtol=1e-6)
    np.testing.assert_allclose(ws[1:,0]/ws[:-1,0],[10,10],rtol=1e-6)
    np.testing.assert_allclose(priority_weights(.5,0.,0.,0.,PriorityConfig())[1:3],[1,1])
    from sttw_control.network import make_policy_identity
    from sttw_control.env import load_config
    c=load_config('learning/configs/priority_conditioned_learning.json')
    legacy=asdict(c);legacy['priority'].update(weight_schedule='linear',tracking_weight_scale=1.,weight_base=10.)
    old={**legacy,'priority':dict(legacy['priority'])}
    for key in ['weight_schedule','tracking_weight_scale','weight_base']:old['priority'].pop(key)
    assert make_policy_identity({},legacy,10)==make_policy_identity({},old,10)
