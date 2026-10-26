import copy
import json
from pathlib import Path

import numpy as np
import pytest

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
