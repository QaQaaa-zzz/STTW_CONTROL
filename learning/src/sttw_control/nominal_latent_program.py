"""Extract an observed nominal neural program without inventing domain-exit outputs."""
import numpy as np


def extract_nominal_program(log, *, lower_ticks=800, upper_stride=4, dt_s=.005):
    active = np.asarray(log['active_tick'], bool)
    if not np.array_equal(active, np.arange(len(active)) < active.sum()):
        raise ValueError('active samples must form one observed prefix')
    time = np.asarray(log['time'])[active]
    actions = np.asarray(log['original_policy_latent'])[active]
    valid = np.asarray(log['upper_active'], bool)[active]
    held = np.asarray(log['fallback_active'], bool)[active]
    if actions.shape != (len(time), 2) or valid.shape != time.shape or held.shape != time.shape:
        raise ValueError('aligned original bare latent and permission logs required')
    if not np.allclose(time, np.arange(len(time))*dt_s, atol=3e-6, rtol=0):
        raise ValueError('original suffix clock is not aligned')
    reasons = []
    if len(time) != lower_ticks:
        reasons.append('incomplete_physical_observation')
    if not valid.all() or held.any():
        reasons.append('nominal_actor_permission_not_continuous')
    if not np.isfinite(actions).all():
        reasons.append('missing_nominal_bare_latents_no_fill_allowed')
    receipt = dict(eligible=not reasons, reasons=reasons, observed_lower_ticks=len(time),
                   actor_valid_lower_ticks=int(valid.sum()), held_lower_ticks=int(held.sum()),
                   finite_latent_values=int(np.isfinite(actions).sum()),
                   expected_lower_ticks=lower_ticks, upper_stride=upper_stride,
                   source='original_policy_latent before governor/filter; never held final reference')
    if reasons:
        return None, receipt
    if lower_ticks % upper_stride:
        raise ValueError('complete upper publication blocks required')
    blocks = actions.reshape(-1, upper_stride, 2)
    if not np.array_equal(blocks, np.repeat(blocks[:, :1], upper_stride, axis=1)):
        raise ValueError('bare output changed within a logged upper publication block')
    receipt['upper_publications'] = len(blocks)
    return blocks[:, 0].copy(), receipt
