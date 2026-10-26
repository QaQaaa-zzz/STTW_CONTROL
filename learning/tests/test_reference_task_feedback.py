import copy

import jax.numpy as j
import numpy as np
import pytest

from sttw_control.path_command_policy import map_latent
from sttw_control.finite_value_evaluation import confirmation_inventory,paired_report
from sttw_control.reference_task_feedback import (
    steering_feedback,context_geometry,feedback_cases,verify_feedback_log)

SPEC=dict(gains_rad=dict(feedback_005=.05,feedback_010=.1),path_scale_m=.5,heading_scale_rad=.3)
CFG=dict(residual_interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2),
         network=dict(context_fields=[dict(name='cos_path_heading_error',scale=1.),
             dict(name='signed_cross_track_error_m',scale=.2),dict(name='sin_path_heading_error',scale=1.)]))


def test_zero_feedback_is_exact_and_speed_latent_is_never_round_tripped():
    for original in [j.array([.1324,-.83]),j.array([-14.,14.])]:
        out,_,real=steering_feedback(original,0.,0.,.1,CFG,SPEC)
        np.testing.assert_array_equal(out,original)
        assert float(real)==0
        out,_,_=steering_feedback(original,.7,-.2,.1,CFG,SPEC)
        assert float(out[0])==float(original[0])


def test_signed_feedback_and_projection_obey_original_physical_authority():
    right,_,_=steering_feedback(j.zeros(2),.5,0.,.1,CFG,SPEC)
    left,_,_=steering_feedback(j.zeros(2),-.5,0.,.1,CFG,SPEC)
    assert float(right[1])<0<float(left[1])
    np.testing.assert_allclose(left,-right)
    heading,_,_=steering_feedback(j.zeros(2),0.,.3,.1,CFG,SPEC)
    assert float(heading[1])>0
    saturated,_,_=steering_feedback(j.array([-.17,10.]),-5.,2.,.1,CFG,SPEC)
    assert abs(float(map_latent(saturated,CFG)[1]))<=.2


def test_named_context_decoding_preserves_scaling_and_field_order():
    ey,eh=context_geometry(j.array([99.,99.,np.cos(.4),1.5,np.sin(.4)]),CFG)
    np.testing.assert_allclose(ey,.3,rtol=1e-6)
    np.testing.assert_allclose(eh,.4,rtol=1e-6)


def test_both_feedback_gains_reuse_every_event_without_changing_alpha():
    protocol=dict(seed=2631,alphas=[0.,.5,1.],forces_N=[0.,-16.,16.,-32.,32.],onset_upper_tick_range=[140,241],pulse_duration_s=.1)
    contexts={name:dict(speed_m_s=speed,world_direction=[0.,1.,0.]) for name,speed in [('left',1.2),('right',1.5)]}
    original=confirmation_inventory(protocol,contexts)
    result=feedback_cases(original,SPEC)
    assert len(result)==60 and len({r['case_id'] for r in result})==60
    baseline={(r['context'],r['alpha'],r['event_id']):r for r in original if r['role']=='parent_plain'}
    for row in result:
        b=baseline[row['context'],row['alpha'],row['event_id']]
        assert row['mode_index']==1 and row['causal_support']
        for key in ['world_direction','force_N','pulse_duration_s','pulse_onset_task_tick','requested_speed_m_s']:
            assert row[key]==b[key]
    wrong=copy.deepcopy(SPEC);wrong['gains_rad'].pop('feedback_010')
    with pytest.raises(ValueError):feedback_cases(original,wrong)


def test_feedback_audit_rejects_control_outside_original_permission():
    raw=dict(active_tick=np.ones(2,bool),actor_action_valid=np.zeros(2,bool),issued_latent=np.zeros((2,2)))
    verify_feedback_log(raw,{},CFG,SPEC)
    raw['issued_latent'][1,1]=.1
    with pytest.raises(AssertionError):verify_feedback_log(raw,{},CFG,SPEC)


def test_feedback_roles_get_matched_comparisons_against_supported_actor():
    rows=[]
    for role in ['parent_plain','free_GAE','feedback_005','feedback_010']:
        rows.append(dict(case_id=role,role=role,category='safe_task_failed',context='left',alpha=.5,event_id='event',force_N=16.,
                         scored_cost=1.,task_recovered=False))
    # Other pairwise metrics use these fields even for a simple synthetic row.
    for row in rows:
        row.update(task_path_rmse_m=1.,task_speed_rmse_m_s=1.,task_heading_rmse_rad=1.)
    got=paired_report(rows,roles=[r['role'] for r in rows])
    assert 'feedback_005_vs_free_GAE' in got['comparisons']
