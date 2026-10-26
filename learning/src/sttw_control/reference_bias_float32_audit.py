"""Allow float32 interpolation roundoff; all remaining original audits persist."""
import numpy as np
from .reference_bias_optimization import verify_bias_log as original_verify


def verify_bias_log(raw, *, positions, targets, **kwargs):
    positions, targets = np.asarray(positions), np.asarray(targets)
    progress = np.asarray(raw['reference_bias_progress'])
    k = np.clip(np.searchsorted(positions, progress, side='right')-1, 0, 3)
    u = np.clip((progress-positions[k])/(positions[k+1]-positions[k]), 0, 1)
    expected = targets[k] + (u*u*(3-2*u))[:, None]*(targets[k+1]-targets[k])
    active = np.asarray(raw['active_tick'], bool)
    # Physical units: one micrometre/second and one microradian. This changes
    # only the independent diagnostic, never a target, reward, goal or action.
    np.testing.assert_allclose(raw['reference_bias_target'][active], expected[active],
                               atol=1e-6, rtol=1e-6)
    # Delegate unchanged native/action/zero/goal/feedback audits. Only the field
    # already checked above is substituted in this temporary audit dictionary;
    # original saved arrays and every physically applied action stay untouched.
    original_verify(dict(raw, reference_bias_target=expected),
                    positions=positions, targets=targets, **kwargs)
    return float(np.max(abs(raw['reference_bias_target'][active]-expected[active])))
