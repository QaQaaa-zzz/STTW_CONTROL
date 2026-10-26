import numpy as np
import pytest

from review_stage_preforce import FIELDS, sample_prefix


def test_preforce_measurement_excludes_force_and_rejects_contaminated_clock():
    raw = {name: np.zeros((160, 2)) if name in ['reference', 'nominal_reference', 'correction_offset']
           else np.zeros(160) for name in FIELDS}
    raw.update(active_tick=np.ones(160, bool), physical_tick=np.arange(160),
               applied_force_impulse=np.zeros((160, 3)))
    raw['phi'][120:] = 999.
    raw['reference'][120:, 1] = 999.
    raw['applied_force_impulse'][120:140, 0] = .1
    result = sample_prefix(raw, 120)
    assert result['samples'] == 100
    assert result['tv_sample_intervals'] == 99
    assert result['tv_sample_span_s'] == .495
    assert result['sample_task_tick'] == 119
    assert result['first_measured_force_task_tick'] == 120
    assert result['prewindow_peak_abs_roll_rad'] == 0
    assert result['prewindow_reference_steer_tv_rad'] == 0
    raw['applied_force_impulse'][119, 0] = .1
    with pytest.raises(ValueError, match='contaminates'):
        sample_prefix(raw, 120)
