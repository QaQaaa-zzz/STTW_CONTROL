import copy
import numpy as np
import pytest
import jax.numpy as j

from sttw_control.support_permission_audit import verify_disabled_support,verify_correction_chain
from sttw_control.path_command_policy import correction_tick


def fixture():
    cfg=dict(timing=dict(lower_dt_s=.005),residual_interface=dict(
        speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2,
        filter_tau_s=[.1,.08],correction_speed_down_m_s2=1.,correction_speed_up_m_s2=.5,
        correction_steer_rate_rad_s=.8,speed_reference_bounds_m_s=[1.,3.5],steer_reference_bound_rad=.65))
    raw=dict(active_tick=np.ones(8,bool),actor_action_valid=np.zeros(8,bool),
        physical_tick=np.arange(1020,1028),relay_profile=np.array([7]*4+[0]*4),
        relay_active=np.array([True]*4+[False]*4),support_permission_override=np.array([True]*4+[False]*4),
        nominal_reference=np.tile([1.1,.64],(8,1)),issued_latent=np.array([[np.arctanh(1-1e-6),0.]]*4+[[0.,0.]]*4,dtype=np.float32))
    filtered=j.array([-.1,.04]);offset=j.array([-.1,.04]);initial_f=np.array(filtered);initial_o=np.array(offset)
    records=[]
    for latent,nominal in zip(raw['issued_latent'],raw['nominal_reference']):
        filtered,offset,reference,flags=correction_tick(filtered,offset,j.asarray(latent),j.asarray(nominal),cfg)
        records.append((np.array(filtered),np.array(offset),np.array(reference),np.array(flags['target'])))
    for i,key in enumerate(['filtered_offset','correction_offset','reference','target_offset']):
        raw[key]=np.stack([row[i] for row in records])
    return cfg,raw,initial_f,initial_o


def test_independent_audit_accepts_real_correction_helper_and_rejects_tampering():
    cfg,raw,f,o=fixture()
    assert verify_disabled_support(raw,cfg,enabled=True)['override_lower_ticks']==4
    assert verify_correction_chain(raw,cfg,initial_filtered=f,initial_offsets=o)['correction_ticks']==8
    for field in ['support_permission_override','issued_latent','relay_active']:
        bad=copy.deepcopy(raw);bad[field][0]=0 if field!='issued_latent' else [2.,.1]
        with pytest.raises(AssertionError):verify_disabled_support(bad,cfg,enabled=True)
    for field in ['filtered_offset','correction_offset','reference','target_offset']:
        bad=copy.deepcopy(raw);bad[field][3,0]+=.01
        with pytest.raises(AssertionError):verify_correction_chain(bad,cfg,initial_filtered=f,initial_offsets=o)
    with pytest.raises(AssertionError):verify_correction_chain(raw,cfg,initial_filtered=np.zeros(2),initial_offsets=o)


def test_old_feedback_verifier_stays_strict_and_new_protocol_is_explicit():
    from sttw_control.reference_task_feedback import verify_feedback_log
    cfg,raw,_,_=fixture()
    with pytest.raises(AssertionError):verify_feedback_log(raw,{},cfg,{})
    verify_feedback_log(raw,{},cfg,{},disabled_action_audit=lambda saved:verify_disabled_support(saved,cfg,enabled=True))
    with pytest.raises(AssertionError):verify_disabled_support(raw,cfg,enabled=False)
