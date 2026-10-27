import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.reference_bias_runtime import batch_size
from sttw_control.reference_preservation_confirmation import read_inputs,validate_inventory
from sttw_control.preference_best import atomic_json,digest


def test_64_requires_explicit_real_batch_permission():
    cases=[{}]*64;v=np.zeros((64,6))
    with pytest.raises(ValueError):batch_size(cases,v)
    assert batch_size(cases,v,(16,64))==64
    with pytest.raises(ValueError):batch_size(cases,np.zeros((64,5)),(16,64))


def test_frozen_actual_inventory_sharing_and_event_independence():
    parent=Path('runs/reference_bias_opportunity_v1')
    if not parent.exists():pytest.skip('server-only frozen research fixture unavailable')
    _,_,logical,physical=read_inputs(parent)
    assert validate_inventory(logical,physical)
    altered=copy.deepcopy(logical);altered[0]['case']['unannounced_control_permission']=True
    with pytest.raises(ValueError,match='semantics'):validate_inventory(altered,physical)
    changed=copy.deepcopy(logical);changed[0]['vector'][0]+=.01
    with pytest.raises(ValueError,match='vector'):validate_inventory(changed,physical)


def test_logical_expansion_keeps_feedback_identity_distinct_from_selected_candidate():
    from analyze_preservation_timecheck import expand_logical_rows
    audited=[dict(raw_sha256='same-physical-trace',role='full',source_role='feedback_010')]
    logical=[dict(physical_index=0,source_role='cem',case=dict(role='context_choice',source_role='feedback_010')),
             dict(physical_index=0,source_role='cem',case=dict(role='alpha_only',source_role='feedback_010'))]
    rows=expand_logical_rows(audited,logical)
    assert [r['role'] for r in rows]==['context_choice','alpha_only']
    assert all(r['source_role']=='feedback_010' and r['chosen_source_role']=='cem' for r in rows)
    assert rows[0]['raw_sha256']==rows[1]['raw_sha256']


@pytest.mark.parametrize('fail',[False,True])
def test_fixed_scheduler_and_partial_error_accounting(tmp_path,monkeypatch,fail):
    from sttw_control import reference_preservation_confirmation as campaign
    from sttw_control import reference_bias_runtime as runtime
    parent=Path('runs/reference_bias_opportunity_v1')
    if not parent.exists():pytest.skip('server-only fixed inventory unavailable')
    _,p,logical,physical=read_inputs(parent)
    atomic_json(tmp_path/'seal.json',{'fixture':'scheduler only'})
    atomic_json(tmp_path/'status.json',{'phase':'prepared'})
    atomic_json(tmp_path/'execution_binding.json',dict(ready=True,declaration_seal_sha256=digest(tmp_path/'seal.json'),source_sha256={},real_batch_sizes=[64,64,64,16,16,16]))
    monkeypatch.setattr(campaign,'read_inputs',lambda _: (tmp_path,p,logical,physical));calls=[]
    class Fake:
        def __init__(self,parent):pass
        def rollout(self,cases,vectors,directory,*,allowed_real_batch_sizes):
            calls.append(len(cases));assert allowed_real_batch_sizes==(16,64)
            directory.mkdir(parents=True)
            atomic_json(directory/'status.json',dict(active_lower_ticks=137 if fail else len(cases)*2000,allocated_lower_slots=len(cases)*2000,phase='error' if fail else 'completed'))
            if fail:raise RuntimeError('partial failure fixture; no implicit retry')
            return [dict(c,vector=v.tolist()) for c,v in zip(cases,vectors)]
    monkeypatch.setattr(runtime,'BiasRuntime',Fake)
    if fail:
        with pytest.raises(RuntimeError,match='partial failure'):campaign.execute(parent)
    else:campaign.execute(parent)
    state=json.loads((tmp_path/'status.json').read_text())
    if fail:
        assert calls==[64] and state['phase']=='error'
        assert state['active_lower_ticks']==137 and state['allocated_lower_slots']==128000
    else:
        assert calls==[64,64,64,16,16,16]
        assert state['completed_physical_cases']==240 and state['completed_logical_cases']==384
        assert state['active_lower_ticks']==state['allocated_lower_slots']==480000
