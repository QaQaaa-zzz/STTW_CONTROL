import copy

import jax.numpy as j
import numpy as np
import pytest

from sttw_control.nominal_reference_feedback import (
    nominal_table,lookup_nominal,adjusted_errors,DeviationActor,verify_nominal_feedback)
from sttw_control.nominal_feedback_probe import probe_cases
from sttw_control.reference_task_feedback import steering_feedback

SPEC=dict(path_scale_m=.5,heading_scale_rad=.3)
CFG=dict(residual_interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2),
    network=dict(context_fields=[dict(name='signed_cross_track_error_m',scale=.2),
        dict(name='sin_path_heading_error',scale=1.),dict(name='cos_path_heading_error',scale=1.)]))


def table():
    progress=np.arange(1,2001)/100
    return nominal_table(progress,progress/10,np.zeros(2000),initial_progress=0.,initial_ey=0.,initial_eh=0.)


def test_profile_uses_true_initial_anchor_and_matches_original_observation_clip():
    t=table()
    assert t['progress'].shape==(2001,) and t['progress'][0]==0
    assert t['path_error'][-1]==1.
    ey,eh,covered=lookup_nominal(j.float32(5.),{k:j.asarray(v) for k,v in t.items()})
    np.testing.assert_allclose(ey,.5)
    assert bool(covered) and float(eh)==0
    assert not bool(lookup_nominal(j.float32(21.),t)[2])
    assert not bool(lookup_nominal(j.float32(-.1),t)[2])


def test_nonmonotone_or_incomplete_nominal_response_is_not_silently_repaired():
    p=np.arange(1,2001,dtype=float)
    p[100]=p[99]
    with pytest.raises(ValueError):
        nominal_table(p,p,p,initial_progress=0,initial_ey=0,initial_eh=0)


def test_heading_difference_wraps_at_pi_and_nominal_error_cancels_exactly():
    ey,eh=adjusted_errors(.7,-np.pi+.05,.7,np.pi-.05)
    assert float(ey)==0
    np.testing.assert_allclose(eh,.1,atol=5e-7)
    original=j.asarray([.1234,-.75])
    errors=adjusted_errors(.7,.5,.7,.5)
    out,requested,_=steering_feedback(original,*errors,.1,CFG,SPEC)
    np.testing.assert_array_equal(out,original)
    assert float(requested)==0


def test_actor_wrapper_preserves_native_speed_and_exact_zero_intervention():
    class Actor:
        def apply(self,params,obs):return params
    actor=DeviationActor(Actor(),CFG,SPEC)
    original=j.asarray([-.123456,14.])
    obs=j.asarray([2.,np.sin(.2),np.cos(.2)])
    result=actor.apply(dict(actor=original,reference_error=j.asarray([.4,.2]),effective_gain=0.),obs)
    np.testing.assert_array_equal(result,original)
    result=actor.apply(dict(actor=original,reference_error=j.zeros(2),effective_gain=.1),obs)
    assert result[0]==original[0]


def test_probe_keeps_both_variants_and_all_nominal_and_disturbed_events():
    protocol=dict(schema='nominal_deviation_and_existing_gate_probe_v1',roles=['feedback_nominal','feedback_gate'],
        feedback_gain_rad=.1,path_scale_m=.5,heading_scale_rad=.3,goal_radius_m=.5)
    source=[dict(case_id=f'context/a0/event{i}/free_GAE',role='free_GAE',mode_index=1,
        causal_support=True,alpha=0.,force_N=0. if i<6 else 16.,event_id=f'event{i}',pulse_onset_task_tick=700)
        for i in range(54)]
    result=probe_cases(source,protocol)
    assert len(result)==108 and sum(not r['force_N'] for r in result)==12
    assert len({r['case_id'] for r in result})==108
    assert {r['feedback_kind'] for r in result}=={1,2}
    assert all(r['mode_index']==1 and r['alpha']==0 and r['pulse_onset_task_tick']==700 for r in result)
    bad=copy.deepcopy(protocol);bad['roles'].pop()
    with pytest.raises(ValueError):probe_cases(source,bad)


def test_independent_audit_detects_illegal_feedback_and_profile_mismatch():
    t=table()
    raw=dict(active_tick=np.ones(2,bool),actor_action_valid=np.zeros(2,bool),issued_latent=np.zeros((2,2)),
        nominal_progress_m=np.array([0.,1.]),nominal_profile_covered=np.ones(2,bool),
        nominal_path_error_m=np.zeros(2),nominal_heading_error_rad=np.zeros(2),
        relay_active=np.zeros(2,bool),feedback_effective_gain_rad=np.zeros(2),
        feedback_path_error_m=np.zeros(2),feedback_heading_error_rad=np.zeros(2),
        feedback_requested_steer_rad=np.zeros(2),original_actor_latent=np.zeros((2,2)),feedback_actor_latent=np.zeros((2,2)))
    verify_nominal_feedback(raw,dict(feedback_kind=2,feedback_gain_rad=.1),CFG,SPEC,t)
    raw['nominal_path_error_m'][0]=.1
    with pytest.raises(AssertionError):verify_nominal_feedback(raw,dict(feedback_kind=2,feedback_gain_rad=.1),CFG,SPEC,t)
    raw['nominal_path_error_m'][0]=0;raw['issued_latent'][0,1]=.1
    with pytest.raises(AssertionError):verify_nominal_feedback(raw,dict(feedback_kind=2,feedback_gain_rad=.1),CFG,SPEC,t)
