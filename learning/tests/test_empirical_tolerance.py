import pytest
from sttw_control.empirical_tolerance import summarize_direction

def test_nonmonotone_success_does_not_extend_prefix():
 r=summarize_direction([(0,True),(1,True),(3,False),(6,True)])
 assert r['last_prefix_grid_success_N']==1
 assert r['first_grid_failure_N']==3
 assert r['success_after_failure_N']==[6]
 assert r['continuous_margin_N'] is None
 assert r['status']=='SAMPLED_TRANSITION_NONMONOTONE'

def test_all_pass_is_censored_not_exact_boundary():
 r=summarize_direction([(6,True),(0,True),(1,True)])
 assert r['status']=='RIGHT_CENSORED'
 assert r['last_prefix_grid_success_N']==6
 assert r['first_grid_failure_N'] is None

def test_incomplete_does_not_become_failure_or_extend_prefix():
 r=summarize_direction([(0,True),(1,None),(3,True),(6,False)])
 assert r['last_prefix_grid_success_N']==0
 assert r['first_grid_failure_N']==6
 assert r['unknown_amplitudes_N']==[1]
 assert r['status']=='INCOMPLETE_GRID'

def test_nominal_fail_not_positive_tolerance():
 r=summarize_direction([(0,False),(1,True)])
 assert r['status']=='NOMINAL_FAILURE'
 assert r['last_prefix_grid_success_N'] is None

def test_no_nominal_is_unknown():
 assert summarize_direction([(1,True)])['status']=='UNKNOWN_NOMINAL'

@pytest.mark.parametrize('rows', [[(0,True),(1,True),(1,False)],[(0,True),(-1,True)],[(0,True),(float('nan'),False)],[(0,True),(1,1)]])
def test_invalid_records_rejected(rows):
 with pytest.raises(ValueError):summarize_direction(rows)

def test_zero_only_is_untested_direction():
 assert summarize_direction([(0,True)])['status']=='UNTESTED_DIRECTION'
