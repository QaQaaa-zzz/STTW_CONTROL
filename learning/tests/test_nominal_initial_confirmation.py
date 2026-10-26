import copy
import json
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from sttw_control.nominal_initial_confirmation import initial_cases,verify_initial_feedback,authenticate_initial_models


def inputs():
    protocol=json.loads((Path(__file__).parents[1]/'configs/nominal_initial_confirmation.json').read_text())
    contexts={f'{base}__bank{index}':dict(nominal_profile_context=base,bank_index=index,
        world_direction=[0.,1.,0.],speed_m_s=speed) for base,speed in [('left',1.2),('right',1.5)] for index in [2,4]}
    previous=[dict(context=base,alpha=alpha,force_N=16.,pulse_onset_task_tick=tick*4)
        for base in ['left','right'] for alpha in [0.,.5,1.] for tick in [140,141,142]]
    return protocol,contexts,previous


def test_prescribed_real_starts_share_unseen_events_and_all_five_controls():
    protocol,contexts,previous=inputs();cases=initial_cases(protocol,contexts,previous)
    assert len(cases)==300 and len({r['case_id'] for r in cases})==300
    assert sum(r['force_N']==0 for r in cases)==60
    groups={};across_starts={}
    for r in cases:
        assert r['pulse_onset_task_tick']//4 not in [140,141,142]
        key=(r['context'],r['alpha'],r['force_N']);groups.setdefault(key,set()).add(r['role'])
        shared=(r['nominal_profile_context'],r['alpha'],r['force_N'])
        across_starts.setdefault(shared,set()).add(r['pulse_onset_task_tick'])
        assert r['feedback_kind']==(1 if r['role']=='feedback_nominal' else 3 if r['role']=='feedback_010' else 0)
        assert r['mode_index']==(0 if r['role']=='parent_plain' else 1)
    assert len(groups)==60 and all(len(x)==5 for x in groups.values())
    assert all(len(x)==1 for x in across_starts.values())
    assert cases==initial_cases(protocol,contexts,previous)


def test_do_not_silently_drop_an_initial_state_or_control_or_change_threshold():
    p,c,old=inputs()
    for key,value in [('bank_indices',[2]),('roles',p['roles'][:-1]),('goal_radius_m',.6)]:
        wrong=copy.deepcopy(p);wrong[key]=value
        with pytest.raises(ValueError):initial_cases(wrong,c,old)
    c.pop('right__bank4')
    with pytest.raises(ValueError):initial_cases(p,c,old)


def test_confirmation_audit_checks_current_progress_not_future_post_step(monkeypatch,tmp_path):
    import sttw_control.reference_task_feedback as feedback
    monkeypatch.setattr(feedback,'verify_feedback_log',lambda *args:None)
    file=tmp_path/'table.npz';np.savez(file,progress=[0,1],path_error=[0,0],heading_error=[0,0])
    raw=dict(active_tick=np.ones(8,bool),physical_tick=np.arange(8),
        nominal_progress_m=np.repeat([.1,.3],4),path_progress=np.arange(1,9)/10)
    case=dict(context='left',alpha=0.,feedback_kind=0)
    plan=dict(nominal_profiles={'left/a0':dict(file=str(file))},feedback_spec={},
        contexts={'left':dict(initial_progress_m=.1)})
    # At decision tick4 the current state is the preceding physical tick3=.4.
    with pytest.raises(AssertionError):verify_initial_feedback(raw,case,{},plan)
    raw['nominal_progress_m'][4:]=.4
    verify_initial_feedback(raw,case,{},plan)


def test_feedback_alias_authentication_uses_native_schema_without_weakening_identity(monkeypatch):
    import sttw_control.finite_value_evaluation_run as runner
    seen=[]
    monkeypatch.setattr(runner,'authenticate_models',lambda cfg,models:seen.append(set(models)))
    native=dict(actor='actor',actor_sha256='sha',checkpoint='checkpoint',checkpoint_sha256='cp_sha',
                fingerprint='identity',value_contract=dict(role='free_GAE'))
    models={role:copy.deepcopy(native) for role in ['parent_plain','parent_causal','free_GAE','feedback_010','feedback_nominal']}
    authenticate_initial_models({},models)
    assert seen==[{'parent_plain','parent_causal','free_GAE'}]
    models['feedback_nominal']['actor_sha256']='different'
    with pytest.raises(ValueError):authenticate_initial_models({},models)


def test_vmapped_feedback_selection_advances_physics_only_once(monkeypatch):
    import jax
    import jax.numpy as j
    import sttw_control.finite_value_evaluation as evaluation
    from sttw_control.nominal_initial_confirmation import make_initial_confirmation_step
    from sttw_control.reference_task_feedback import steering_feedback
    cfg=dict(residual_interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2),
        network=dict(context_fields=[dict(name='signed_cross_track_error_m',scale=.2),
            dict(name='sin_path_heading_error',scale=1.),dict(name='cos_path_heading_error',scale=1.)]))
    spec=dict(gains_rad=dict(feedback_005=.05,feedback_010=.1),path_scale_m=.5,heading_scale_rad=.3)
    observation=j.asarray([2.,np.sin(.2),np.cos(.2)])
    env=SimpleNamespace(config=cfg,cc=SimpleNamespace(dt=.005),observation=lambda *a:(observation,None,None))
    state=SimpleNamespace(physical=SimpleNamespace(failed=j.bool_(False)),fault=j.bool_(False),tick=j.int32(0),
        domain_exit=j.bool_(False),follower=SimpleNamespace(invalid=j.bool_(False)),projection=SimpleNamespace(progress=j.float32(.5)))
    carry=SimpleNamespace(state=state,clean_blocks=j.int32(16))
    class Actor:
        def apply(self,p,obs):return p
    def factory(env,actor,**kw):
        def advance(carry,params,*args):
            action=actor.apply(params,observation)
            return carry,dict(active_tick=j.ones(4,bool),actor_action_valid=j.ones(4,bool),
                issued_latent=j.tile(action,(4,1)),physical_sentinel=j.dot(action,action))
        return advance
    monkeypatch.setattr(evaluation,'make_evaluation_step',factory)
    one=make_initial_confirmation_step(env,Actor(),horizon_s=10.,spec=spec,nominal_spec=spec)
    params=dict(actor=j.asarray([[.2,.3]]*3),feedback_kind=j.asarray([0,3,1]),feedback_gain_rad=j.asarray([0.,.1,.1]),
        nominal_table=dict(progress=j.asarray([[0.,1.]]*3),path_error=j.asarray([[.1,.1]]*3),heading_error=j.zeros((3,2))))
    def advance(p):return one(carry,p,None,j.float32(1.2),None,j.int32(1))[1]
    graph=jax.make_jaxpr(jax.vmap(lambda p:advance(p)['physical_sentinel']))(params)
    assert sum(e.primitive.name=='dot_general' for e in graph.jaxpr.eqns)==1
    logs=jax.vmap(advance)(params)
    original=j.asarray([.2,.3])
    np.testing.assert_array_equal(logs['issued_latent'][0,0],original)
    constant=steering_feedback(original,.4,.2,.1,cfg,spec)[0]
    nominal=steering_feedback(original,.3,.2,.1,cfg,spec)[0]
    np.testing.assert_allclose(logs['issued_latent'][1,0],constant,atol=2e-6)
    np.testing.assert_allclose(logs['issued_latent'][2,0],nominal,atol=2e-6)
