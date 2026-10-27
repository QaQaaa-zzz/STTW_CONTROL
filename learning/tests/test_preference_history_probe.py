import numpy as np
import pytest
from sttw_control.preference_history_probe import pulse_directions


def test_declared_world_direction_is_shared_despite_different_capture_yaw():
    vector = [0., 1., 0.]
    actual = pulse_directions([{'world_direction': vector}] * 2, [0., .3])
    np.testing.assert_array_equal(actual, [vector, vector])
    legacy = pulse_directions([{}, {}], [0., .3])
    np.testing.assert_allclose(legacy[1], [-np.sin(.3), np.cos(.3), 0.])


@pytest.mark.parametrize('vector', [[0., 0., 0.], [0., 2., 0.], [0., 0., 1.], [np.nan, 1., 0.]])
def test_invalid_direction_cannot_silently_change_force_magnitude(vector):
    with pytest.raises(ValueError):
        pulse_directions([{'world_direction': vector}], [0.])
