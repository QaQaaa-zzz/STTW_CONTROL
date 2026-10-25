"""Execution bookkeeping must preserve real ordering, sealed inputs and costs."""
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.preference_best import atomic_json, digest


def test_preparation_copies_sealed_inputs_and_refuses_drift(tmp_path):
    from sttw_control.survival_endpoint_run import prepare_run
    source=tmp_path/'protocol';source.mkdir()
    atomic_json(source/'cases.json',[{'case_id':'actual-case'}])
    atomic_json(source/'common_environment_config.json',{'timing':{'lower_dt_s':.005}})
    atomic_json(source/'nominal_geometry.json',{'routes':[]})
    origin=tmp_path/'origin';origin.write_text('immutable')
    plan=dict(training_run=str(tmp_path/'training'),declared_actual_episodes=1,maximum_lower_ticks=2000,
              source_sha256={str(origin):digest(origin)},horizon_s=10.,lower_dt_s=.005,seed=42)
    for name,key in [('cases.json','cases_sha256'),('common_environment_config.json','common_config_sha256'),
                     ('nominal_geometry.json','nominal_geometry_sha256')]:plan[key]=digest(source/name)
    atomic_json(source/'plan.json',plan)
    out=tmp_path/'evaluation';prepare_run(source,out)
    assert json.loads((out/'status.json').read_text())['phase']=='prepared'
    assert (out/'cases.json').read_bytes()==(source/'cases.json').read_bytes()
    assert json.loads((out/'plan.json').read_text())['optimizer_updates']==0
    origin.write_text('changed')
    with pytest.raises(ValueError,match='changed'):prepare_run(source,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_chunk_flattening_keeps_each_case_lower_time_and_decisions_aligned():
    from sttw_control.survival_endpoint_run import flatten_chunk, stack_case_chunks
    # scan, environment, lower-tick. This ordering is easy to corrupt silently.
    physical={'time':np.arange(2*3*4).reshape(2,3,4),
              'xy':np.arange(2*3*4*2).reshape(2,3,4,2)}
    decisions={'valid':np.array([[1,0,1],[0,1,1]],bool)}
    d,p=flatten_chunk(decisions,physical)
    np.testing.assert_array_equal(p['time'][:,1],[4,5,6,7,16,17,18,19])
    d1,p1=stack_case_chunks([(d,p),(d,p)],1)
    np.testing.assert_array_equal(p1['time'],[4,5,6,7,16,17,18,19]*2)
    np.testing.assert_array_equal(d1['valid'],[False,True,False,True])
    assert p1['xy'].shape==(16,2)


def test_pairing_rejects_missing_cases_and_reports_gains_and_losses():
    from sttw_control.survival_endpoint_run import paired_outcomes
    rows=[]
    for i,(a,b) in enumerate([('contact','survived'),('survived','contact')]):
        for role,outcome in [('parent',a),('task_only',b)]:
            rows.append(dict(panel='physical_deterministic',context_index=i,alpha=.5,replicate=0,
                             role=role,outcome=outcome))
    result=paired_outcomes(rows,'parent','task_only')
    assert result==dict(reference='parent',candidate='task_only',paired=2,gain=1,loss=1,both_safe=0,both_contact=0,unknown=0)
    with pytest.raises(ValueError,match='unmatched'):paired_outcomes(rows[:-1],'parent','task_only')
