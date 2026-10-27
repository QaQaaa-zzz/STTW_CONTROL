"""Scenario identity, physical-vs-task outcomes and equal rollout budgets."""
import copy
import numpy as np
import pytest
from sttw_control.reference_scenario_search import aggregate, search, validate_scenarios, evaluate_generation


def scenarios():
    return [dict(context='declared_route',alpha=0.,case_id=f'route/F{f}',event_id=f'F{f}',force_N=f,
                 pulse_onset_task_tick=600,pulse_duration_s=.1,world_direction=[0.,1.,0.])
            for f in (0.,-32.,32.)]


def rows_for(vectors, *, mode='quadratic'):
    rows=[]
    for v in vectors:
        for s in scenarios():
            # A deterministic toy outcome, never presented as physical evidence.
            alive=(s['force_N']==0 or (v[0]>.1 if s['force_N']<0 else v[0]<.6))
            cost=10.+float(np.sum((v-.2)**2))
            rows.append(dict(s,vector=list(v),physical_outcome='survived' if alive else 'contact',
                             goals={'1.0':{'goal_arrived':True}},scored_cost=cost))
    return rows


def test_physical_and_arrival_counts_separate_and_nominal_cost_not_success_veto():
    v=np.zeros((1,6),np.float32);r=rows_for(v)
    got=aggregate(v,r,scenarios(),nominal_cost_limit=1)[0]
    assert not got.nominal_feasible
    assert got.survived==1 and got.arrived==2
    assert r[0]['goals']['1.0']['goal_arrived']  # Cost allowance is only search feasibility.
    better=copy.deepcopy(r);better[1]['physical_outcome']='survived';better[1]['scored_cost']=500
    a=aggregate(v,r,scenarios(),nominal_cost_limit=20)[0]
    b=aggregate(v,better,scenarios(),nominal_cost_limit=20)[0]
    assert b.key<a.key  # Explicit lexicographic ordering, not hidden reward scaling.


@pytest.mark.parametrize('corruption',['unknown','missing','order','alpha','vector','cost','label'])
def test_unknown_or_misaligned_result_never_becomes_an_optimization_negative(corruption):
    v=np.zeros((2,6),np.float32);r=rows_for(v)
    if corruption=='unknown':r[2]['physical_outcome']='unknown'
    if corruption=='missing':r.pop()
    if corruption=='order':r[1],r[2]=r[2],r[1]
    if corruption=='alpha':r[1]['alpha']=1.
    if corruption=='vector':r[1]['vector'][0]=.2
    if corruption=='cost':r[1]['scored_cost']=np.nan
    if corruption=='label':r[1]['goals']['1.0']['goal_arrived']=None
    with pytest.raises(ValueError):aggregate(v,r,scenarios(),nominal_cost_limit=20)


def test_search_equal_full_rectangles_first_generation_and_fixed_original_limit():
    calls={};outputs={}
    for method in ('cem','random'):
        calls[method]=[]
        def evaluate(v):
            calls[method].append(v.copy())
            return rows_for(v)
        outputs[method]=search(evaluate,scenarios(),seed=17,method=method)
        got=outputs[method]
        assert got['charged_rollout_calls']==192 and got['scored_candidate_processes']==64
        assert len(calls[method])==4
        np.testing.assert_array_equal(calls[method][0][0],np.zeros(6))
        assert got['nominal_cost_limit']==pytest.approx(10.24*1.1)
        for i in range(1,4):
            np.testing.assert_array_equal(calls[method][i][0],got['history'][i-1]['incumbent'])
        keys=[(-h['best']['survived'],-h['best']['arrived'],h['best']['mean_disturbed_cost']) for h in got['history']]
        assert all(b<=a for a,b in zip(keys,keys[1:]))
    np.testing.assert_array_equal(calls['cem'][0],calls['random'][0])
    assert not np.array_equal(calls['cem'][1][1:],calls['random'][1][1:])
    # No evaluated event may select its own deployment vector.
    assert np.asarray(outputs['cem']['vector']).shape==(6,)


def test_bad_nominal_baseline_or_duplicate_zero_stops_without_replacement():
    def bad(v):
        r=rows_for(v);r[0]['physical_outcome']='contact';return r
    with pytest.raises(ValueError,match='incumbent'):
        search(bad,scenarios(),seed=1,method='cem')
    s=scenarios();s[1]['force_N']=0.
    with pytest.raises(ValueError,match='nominal'):validate_scenarios(s)
    s=scenarios();s[1]['event_id']=s[2]['event_id']
    with pytest.raises(ValueError,match='identities'):validate_scenarios(s)


def test_real_runtime_adapter_crosses_batches_without_changing_process_or_event(tmp_path):
    import json
    from sttw_control.preference_best import atomic_json
    events=scenarios()
    for force in (-16.,16.):
        events.append(dict(events[0],case_id=f'route/F{force}',force_N=force,event_id=f'F{force}'))
    vectors=np.zeros((16,6),np.float32)
    vectors[:,0]=np.linspace(0,.5,16)
    calls=[]
    class Runtime:
        def rollout(self,cases,vs,directory,*,allowed_real_batch_sizes):
            assert allowed_real_batch_sizes==(16,64)
            directory.mkdir()
            calls.append((copy.deepcopy(cases),vs.copy()))
            out=[]
            for c,v in zip(cases,vs):
                assert c['force_N']==events[c['scenario_index']]['force_N']
                np.testing.assert_array_equal(v,vectors[c['candidate_index']])
                out.append(dict(c,vector=v.tolist(),physical_outcome='survived',
                                goals={'1.0':{'goal_arrived':True}},scored_cost=10.))
            atomic_json(directory/'status.json',dict(phase='completed',completed_cases=len(cases),
                         active_lower_ticks=len(cases)*1999,allocated_lower_slots=len(cases)*2000))
            return out
    got=evaluate_generation(Runtime(),events,vectors,tmp_path/'generation')
    assert [len(c) for c,v in calls]==[64,16]
    assert len(got)==80 and got[64]['candidate_index']==12 and got[64]['scenario_index']==4
    receipt=json.loads((tmp_path/'generation/receipt.json').read_text())
    assert receipt['actual_lower_ticks']==80*1999
    assert receipt['allocated_lower_slots']==160000 and receipt['padding']==0
    assert events[0]['case_id']=='route/F0.0'  # No mutation of declaration.


def test_runtime_adapter_requires_full_fresh_generation_and_preserves_partial_costs(tmp_path):
    from sttw_control.preference_best import atomic_json
    events=scenarios()
    for s in events:s['case_id']=s['event_id']
    class BrokenRuntime:
        def rollout(self,cases,vs,directory,**kwargs):
            directory.mkdir()
            atomic_json(directory/'status.json',dict(phase='error',active_lower_ticks=137,
                                                     allocated_lower_slots=3200))
            raise RuntimeError('physical engineering fault')
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError):
        evaluate_generation(BrokenRuntime(),events,np.zeros((16,6)),out)
    assert (out/'batch_00/status.json').exists() and not (out/'receipt.json').exists()
    with pytest.raises(FileExistsError):
        evaluate_generation(BrokenRuntime(),events,np.zeros((16,6)),out)
