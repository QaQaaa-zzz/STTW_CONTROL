from copy import deepcopy
import numpy as np
import pytest
from test_path_task_selection import CFG, panel
from sttw_control.path_command_selection import summarize_task, task_score_key, persist_task_best, task_pointer_name

def cfg():
    c=deepcopy(CFG);c['task_selection_schema']='path_safety_task_v1';return c

def test_safety_severity_duration_and_corridor_order(tmp_path):
    a=panel();b=panel();name='left90_R2_2.6'
    a[name]['peak_roll'][0]=.31;b[name]['peak_roll'][:20]=.305
    sa=summarize_task(a,0,cfg());sb=summarize_task(b,0,cfg())
    assert task_score_key(sb)<task_score_key(sa)
    assert sb['roll_exceedance_seconds']==pytest.approx(.1)
    b[name]['peak_roll'][:]=.1;b[name]['path_cross_track'][0]=.401
    s=summarize_task(b,0,cfg());assert not s['qualified_task'] and s['corridor_failures']==1
    assert len(task_score_key(s))==11
    assert task_pointer_name(cfg())=='best_safety_task_model.json'
    ck=tmp_path/'c';ck.write_bytes(b'checkpoint');act=tmp_path/'a';act.write_bytes(b'actor')
    (tmp_path/'best_task_model.json').write_text('historical')
    persist_task_best(tmp_path,ck,act,s,{'schema':s['schema'],'alpha':0})
    assert (tmp_path/'best_task_model.json').read_text()=='historical'
    assert (tmp_path/'best_safety_task_model.json').exists()

def test_safe_roll_has_zero_excess_and_old_version_unchanged():
    a=panel();b=panel()
    for d in b.values():d['peak_roll'][:]=.28
    assert task_score_key(summarize_task(a,0,cfg()))==task_score_key(summarize_task(b,0,cfg()))
    assert summarize_task(a,0,CFG)['selection_schema']=='path_task_best_v2'
    c=cfg();c['task_selection_schema']='unknown'
    with pytest.raises(ValueError,match='schema'):summarize_task(a,0,c)

def test_preference_equal_bin_mse_and_full_turn_coverage():
    from sttw_control.path_command_selection import qualify_preference
    a=panel();b=panel();paths={}
    for name,d in a.items():
        paths[name]={'s':np.array([0.,3.,4.]),'curvature':np.array([0.,1.,1.]),'turn_end':4.}
        d['path_progress']=np.linspace(0,20,4000)
        b[name]['path_progress']=d['path_progress'].copy()
        mask=(d['path_progress']>=3)&(d['path_progress']<=4)
        d['actual_forward_speed'][mask]=2.09;d['path_cross_track'][mask]=.01
        b[name]['actual_forward_speed'][mask]=2.01;b[name]['path_cross_track'][mask]=.05
    result=qualify_preference(a,b,paths,cfg());assert result['qualified']
    b['left90_R2_2.6']['path_progress'][:]=20
    result=qualify_preference(a,b,paths,cfg());assert not result['qualified']
    assert result['cases']['left90_R2_2.6']['status']=='incomplete_turn_bins'
    a['left90_R2_2.6']['peak_roll'][0]=.31
    assert qualify_preference(a,b,paths,cfg())['status']=='common_task_failed'
