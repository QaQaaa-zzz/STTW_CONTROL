import pytest
from sttw_control.empirical_capability import directional_grid_summary


def test_larger_success_does_not_fill_a_smaller_failure():
    r=directional_grid_summary([0,16,32,48,64],['survived','survived','contact','survived','contact'])
    assert r['all_pass_grid_prefix_N']==16
    assert r['largest_observed_pass_N']==48
    assert r['nonmonotone_observed']
    assert r['first_nonpass_outcome']=='contact'
    assert r['first_nonpass_N']==32
    assert not r['right_censored']


def test_unknown_stops_prefix_without_becoming_contact():
    r=directional_grid_summary([0,16,32,48],['survived','survived','unknown','survived'])
    assert r['all_pass_grid_prefix_N']==16
    assert r['first_nonpass_outcome']=='unknown'
    assert r['first_observed_contact_N'] is None
    assert not r['nonmonotone_observed']
    assert not r['right_censored']


def test_all_observed_passes_are_right_censored():
    r=directional_grid_summary([0,16,32],['survived']*3)
    assert r['all_pass_grid_prefix_N']==32 and r['right_censored']
    assert r['first_nonpass_N'] is None


@pytest.mark.parametrize('zero',['unknown','contact'])
def test_no_safe_zero_does_not_create_zero_force_capacity(zero):
    r=directional_grid_summary([0,16], [zero,'survived'])
    assert r['all_pass_grid_prefix_N'] is None
    assert r['first_nonpass_N']==0
    assert not r['zero_force_survived']


@pytest.mark.parametrize('forces,outcomes',[
    ([16,32],['survived']*2),
    ([0,32,16],['survived']*3),
    ([0,16,16],['survived']*3),
    ([0,float('nan')],['survived']*2),
    ([0,16],['survived','task_failed']),
    ([0],[]),
])
def test_rejects_inconsistent_grid(forces,outcomes):
    with pytest.raises(ValueError):directional_grid_summary(forces,outcomes)
