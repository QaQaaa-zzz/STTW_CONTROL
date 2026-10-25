import numpy as np
import pytest
from sttw_control.anticipatory_analysis import cost_accounting, paired_changes, load_analysis_cases


def test_costs_ignore_padding_and_keep_terminal_replacement_separate():
    d=dict(active_tick=np.array([1,1,0],bool),
        effective_costs=np.array([[1.,2.],[3.,4.],[np.nan,np.nan]]),
        tick_reward=np.array([-.03,-.07,np.nan]),
        scored_tick_reward=np.array([0.,-10.,np.nan]))
    r=cost_accounting(d,scale=2.,dt=.005)
    assert r['running_cost']==pytest.approx(.1)
    assert r['scored_cost']==10.
    assert r['terminal_replacement_delta']==pytest.approx(9.9)
    d['tick_reward'][0]=-.1
    with pytest.raises(ValueError):cost_accounting(d,scale=2.,dt=.005)


def test_pairing_respects_alpha_and_event_and_preserves_losses():
    rows=[dict(case_id='p0',role='parent',alpha=0.,event_id='zero',category='safe_task',scored_cost=1.),
          dict(case_id='p5',role='parent',alpha=.5,event_id='zero',category='safe_task_failed',scored_cost=4.),
          dict(case_id='s0',role='speed_gate',alpha=0.,event_id='zero',category='contact',scored_cost=8.),
          dict(case_id='s5',role='speed_gate',alpha=.5,event_id='zero',category='safe_task',scored_cost=2.)]
    result=paired_changes(rows)
    assert result[0]['physical_change']==-1 and result[0]['joint_task_change']==-1
    assert result[1]['physical_change']==0 and result[1]['joint_task_change']==1
    assert result[1]['scored_cost_change']==-2.


def test_inherited_cases_are_not_counted_as_new_and_duplicates_rejected(tmp_path):
    import json
    stage=tmp_path/'endpoints';stage.mkdir()
    (stage/'results.json').write_text(json.dumps([dict(case_id='new')]))
    (stage/'case_index.json').write_text(json.dumps([dict(case_id='new',file='new.npz')]))
    (tmp_path/'reuse_cases.json').write_text(json.dumps([dict(case=dict(case_id='old'),index=dict(case_id='old',file='/external/old.npz'))]))
    plan=dict(stage_counts=dict(endpoints=1),inherited_cases=1,logical_cases=2)
    rows,index=load_analysis_cases(tmp_path,'endpoints',plan)
    assert len(rows)==2 and len(index)==2 and index[1]['file']=='/external/old.npz'
    (tmp_path/'reuse_cases.json').write_text(json.dumps([dict(case=dict(case_id='new'),index=dict(case_id='new',file='new.npz'))]))
    with pytest.raises(ValueError):load_analysis_cases(tmp_path,'endpoints',plan)


def test_csv_keeps_late_metadata_from_inherited_cases(tmp_path):
    import csv
    from sttw_control.anticipatory_analysis import save_csv
    path=tmp_path/'mixed.csv'
    save_csv(path,[dict(case_id='new',cost=3.),dict(case_id='inherited',cost=4.,sequence_kind='sequence',sequence_index=26)])
    with path.open() as f:
        reader=csv.DictReader(f);rows=list(reader)
        assert reader.fieldnames==['case_id','cost','sequence_kind','sequence_index']
    assert rows[0]['sequence_kind']=='' and rows[1]['sequence_kind']=='sequence' and rows[1]['sequence_index']=='26'
