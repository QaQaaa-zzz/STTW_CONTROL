import pytest

from sttw_control.evaluation_clock import physical_origin,physical_pulse


def test_prefix_clock_moves_absolute_force_without_changing_task_event():
    event=dict(pulse_onset_task_tick=600,pulse_duration_s=.1,force_N=-16.,world_direction=[0.,1.,0.])
    assert physical_origin({})==700
    assert physical_pulse(event,{})==[6.5,6.6,-0.,-16.,-0.]
    new=physical_pulse(event,{'initial_physical_tick':1020})
    assert new==[8.1,8.2,-0.,-16.,-0.]
    assert new[0]-1020*.005==pytest.approx(3.)
    assert event['pulse_onset_task_tick']==600


@pytest.mark.parametrize('tick',[True,699,701,1020.0,-1])
def test_invalid_origin_is_rejected(tick):
    with pytest.raises(ValueError):physical_origin({'initial_physical_tick':tick})
