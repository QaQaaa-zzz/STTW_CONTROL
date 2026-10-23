import pytest
from sttw_control.preference_length_evaluation import paired_counts

def row(force,success):
 return dict(route='left',speed=2.3,alpha=0.,force_N=force,onset_s=2.,task_recovered=success)

def test_pairing_uses_case_identity_not_row_order():
 a=[row(-10,False),row(10,True)];b=[row(10,False),row(-10,True)]
 assert paired_counts(a,b)==dict(n=2,baseline=1,candidate=1,gains=1,losses=1)

@pytest.mark.parametrize('other',[[row(-10,True),row(-10,False)],[row(20,True),row(-10,False)]])
def test_duplicate_or_different_cases_rejected(other):
 with pytest.raises(ValueError):paired_counts([row(-10,False),row(10,True)],other)
