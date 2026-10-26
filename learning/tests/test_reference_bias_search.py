import numpy as np
import pytest
from sttw_control.reference_bias_runtime import batch_size
from sttw_control.reference_bias_search import batch_slices,freeze_selected,evaluation_schedule


def test_final120_are_seven16_plus_eight_real_lanes_without_padding():
    slices=batch_slices(120)
    assert [s.stop-s.start for s in slices]==[16]*7+[8]
    assert [i for s in slices for i in range(s.start,s.stop)]==list(range(120))
    assert batch_size([{}]*8,np.zeros((8,6)))==8
    with pytest.raises(ValueError):batch_size([{}]*8,np.zeros((16,6)))
    with pytest.raises(ValueError):batch_slices(121)


def test_selection_requires_actual_matching_nominal_witness_and_rejects_future_force():
    s=dict(context='ctx',alpha=.5,method='cem',search_id='s',seed=1)
    row=dict(context='ctx',alpha=.5,force_N=0,physical_outcome='survived',nominal_feasible=True,objective=2.,vector=[0.]*6,case_id='one',raw_file='one.npz',raw_sha256='sha')
    rows=[dict(row,case_id=str(i)) for i in range(64)]
    q=freeze_selected(s,dict(vector=np.zeros(6),cost=2.),rows)
    assert q['nominal_case_id']=='0'
    with pytest.raises(ValueError):freeze_selected(s,dict(vector=np.ones(6),cost=2.),rows)
    rows[0]['force_N']=16
    with pytest.raises(ValueError,match='nominal'):freeze_selected(s,dict(vector=np.zeros(6),cost=2.),rows)


def test_all_force_events_use_same_preselected_vector_and_original_alpha():
    events=[];decisions=[]
    for ctx in ['c0','c1','c2','c3']:
        for alpha in [0.,.5,1.]:
            for force in [0,-16,16,-32,32]:events.append(dict(context=ctx,alpha=alpha,event_id=str(force),case_id=f'{ctx}/{alpha}/{force}/full',force_N=force,pulse_onset_task_tick=800+force))
            for method in ['cem','random']:decisions.append(dict(context=ctx,alpha=alpha,method=method,vector=[alpha]*6,nominal_case_id=f'{ctx}/{alpha}/{method}',nominal_raw_sha256='sha'))
    cases,vectors=evaluation_schedule(events,decisions)
    assert len(cases)==120 and len({c['case_id'] for c in cases})==120
    for c,v in zip(cases,vectors):np.testing.assert_array_equal(v,[c['alpha']]*6)
    assert {c['force_N'] for c in cases}=={-32,-16,0,16,32}
