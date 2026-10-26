"""Entire fixed scheduler with fake scored backend; no physics validation claim."""
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.preference_best import atomic_json,digest
from sttw_control.reference_bias_campaign import execute


def fixture(root):
    out=root/'production';out.mkdir();atomic_json(root/'status.json',{'phase':'prepared'})
    events=[];searches=[]
    for i,context in enumerate(['c0','c1','c2','c3']):
        for a in [0.,.5,1.]:
            for method in ['cem','random']:searches.append(dict(context=context,alpha=a,method=method,seed=i+10,search_id=f'{context}/a{a}/{method}'))
            for force in [0,-16,16,-32,32]:events.append(dict(context=context,alpha=a,force_N=force,event_id=str(force),case_id=f'{context}/{a}/{force}/full'))
    atomic_json(root/'plan.json',{});atomic_json(root/'search_inventory.json',searches);atomic_json(root/'evaluation_events.json',events);atomic_json(out/'cached_baselines.json',[])
    c=dict(production_ready=True,source_sha256={},parent_plan_sha256=digest(root/'plan.json'),search_inventory_sha256=digest(root/'search_inventory.json'),event_inventory_sha256=digest(root/'evaluation_events.json'),cached_baselines_sha256=digest(out/'cached_baselines.json'))
    atomic_json(out/'contract.json',c);atomic_json(out/'seal.json',dict(contract_sha256=digest(out/'contract.json')))
    atomic_json(out/'status.json',dict(phase='prepared',completed_searches=0,evaluation_cases=0))
    return out


def test_full_scheduler_freezes_all_choices_before_events_and_charges120_without_padding(tmp_path,monkeypatch):
    import sttw_control.reference_bias_runtime as runtime
    out=fixture(tmp_path);sizes=[]
    class Fake:
        def __init__(self,root):pass
        def rollout(self,cases,vectors,directory):
            d=Path(directory);d.mkdir(parents=True);n=len(cases);sizes.append(n)
            if 'evaluation' in d.parts:assert (out/'selection_frozen.json').exists()
            else:assert all(c['force_N']==0 for c in cases)
            rows=[]
            for i,(c,v) in enumerate(zip(cases,vectors)):
                raw=d/f'raw_{i}.txt';raw.write_text(str(v.tolist()))
                rows.append(dict(c,vector=v.tolist(),objective=float(np.sum((v-.2)**2)+5),nominal_feasible=True,physical_outcome='survived',raw_file=str(raw),raw_sha256=digest(raw)))
            atomic_json(d/'status.json',dict(phase='completed',completed_cases=n,active_lower_ticks=n*2000,allocated_lower_slots=n*2000));return rows
    monkeypatch.setattr(runtime,'BiasRuntime',Fake);execute(tmp_path)
    s=json.loads((out/'status.json').read_text());assert s['phase']=='completed' and s['completed_searches']==24 and s['evaluation_cases']==120
    assert sizes[:96]==[16]*96 and sizes[96:]==[16]*7+[8]
    assert s['active_lower_ticks']==s['allocated_lower_slots']==3312000
    frozen=json.loads((out/'selection_frozen.json').read_text());assert frozen['final_event_cases_executed']==0
    assert frozen['search_cost']['active_lower_ticks']==3072000
    assert len(json.loads((out/'evaluation_results.json').read_text()))==120


def test_backend_error_is_terminal_and_partial_physics_remains_chargeable(tmp_path,monkeypatch):
    import sttw_control.reference_bias_runtime as runtime
    out=fixture(tmp_path);calls=[]
    class Failing:
        def __init__(self,root):pass
        def rollout(self,cases,vectors,directory):
            d=Path(directory);d.mkdir(parents=True);calls.append(1)
            atomic_json(d/'status.json',dict(phase='running',active_lower_ticks=123,allocated_lower_slots=3200))
            raise RuntimeError('fake unknown engineering outcome')
    monkeypatch.setattr(runtime,'BiasRuntime',Failing)
    with pytest.raises(RuntimeError,match='unknown'):execute(tmp_path)
    s=json.loads((out/'status.json').read_text());assert s['phase']=='error' and s['active_lower_ticks']==123 and s['allocated_lower_slots']==3200
    assert len(calls)==1 and not (out/'selected_references.json').exists()
