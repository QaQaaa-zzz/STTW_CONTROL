"""V3 direct reference policy: pure JAX action and observation operations.

The caller publishes the raw command before building an observation, and advances
the recovery clock once after the corresponding 5 ms physical interval.
"""
from __future__ import annotations

from flax import linen as nn, struct
import jax.numpy as jp
import numpy as np
import torch
from torch import nn as tnn
from .controller import _system


@struct.dataclass
class CorrectionState:
    offsets: object


@struct.dataclass
class History:
    frames: object
    mask: object


def initial_correction(dtype=jp.float32):
    return CorrectionState(jp.zeros(2, dtype=dtype))


def map_latent(z, spec):
    """Map unbounded Gaussian latent to target [speed, steer] corrections."""
    a = jp.tanh(jp.asarray(z))
    c = spec['action']
    return jp.stack((a[..., 0] * jp.where(a[..., 0] >= 0,
                    c['speed_positive_scale_m_s'], c['speed_negative_scale_m_s']),
                    a[..., 1] * c['steer_scale_rad']), axis=-1)


def correction_tick(offsets, z, raw, spec):
    """Apply one 5 ms correction tick; back calculate offsets after reference clip."""
    c = spec['action']; dt = spec['plant']['control_dt_s']
    old = jp.asarray(offsets.offsets if isinstance(offsets, CorrectionState) else offsets)
    raw = jp.asarray(raw); target = map_latent(z, spec)
    lower = jp.asarray([-c['correction_speed_decrease_rate_m_s2']*dt,
                        -c['correction_steer_rate_rad_s']*dt])
    upper = jp.asarray([c['correction_speed_increase_rate_m_s2']*dt,
                        c['correction_steer_rate_rad_s']*dt])
    increment = jp.clip(target-old, lower, upper)
    before_clip = raw + old + increment
    governed = jp.clip(before_clip,
        jp.asarray([c['speed_reference_min_m_s'], -c['steer_reference_abs_max_rad']]),
        jp.asarray([c['speed_reference_max_m_s'], c['steer_reference_abs_max_rad']]))
    new_offsets = governed-raw
    flags = dict(rate_clipped=jp.any(increment != target-old, axis=-1),
                 reference_clipped=jp.any(governed != before_clip, axis=-1),
                 policy_fault=~jp.all(jp.isfinite(z), axis=-1))
    return new_offsets, governed, flags


def q_ratio(speed, cc):
    """Use the controller's own equilibrium ratio, including its floors."""
    _, _, _, ratio, _ = _system(speed, jp.zeros(3), cc)
    return ratio


def raw_context(raw, rates, settle_clock, cc, spec):
    """Return current chi/g and the clock to store after this physical interval."""
    raw = jp.asarray(raw); rates = jp.asarray(rates)
    r = spec['reward']; dt = spec['plant']['control_dt_s']
    phi_raw = raw[..., 1]/q_ratio(raw[..., 0], cc)
    chi = jp.clip((jp.abs(phi_raw)-r['conflict_phi_start_rad'])/
                  (r['conflict_phi_full_rad']-r['conflict_phi_start_rad']), 0., 1.)
    eligible = ((jp.abs(raw[..., 1]) <= r['recovery_raw_steer_abs_max_rad']) &
                (jp.abs(phi_raw) <= r['recovery_raw_phi_abs_max_rad']) &
                (jp.abs(rates[..., 0]) <= r['recovery_raw_speed_rate_abs_max_m_s2']) &
                (jp.abs(rates[..., 1]) <= r['recovery_raw_steer_rate_abs_max_rad_s']))
    g = jp.where(eligible, jp.clip((settle_clock-r['recovery_settle_s'])/
                  r['recovery_ramp_s'], 0., 1.), 0.)
    next_clock = jp.where(eligible,
        jp.minimum(r['recovery_settle_s']+r['recovery_ramp_s'], settle_clock+dt), 0.)
    return chi, g, eligible, next_clock


def initial_history(spec, dtype=jp.float32):
    n = spec['network']['history_frames']
    return History(jp.zeros((n, 20), dtype=dtype), jp.zeros(n, dtype=dtype))


def make_frame(*, measurement, forward_speed, previous_governed,
               previous_final_command, previous_bounded_residual,
               raw, raw_rates, eso_equilibrium_shift, cc):
    """Build the 20 raw fields from values available before the next Actor action.

    measurement is [roll, roll_rate, steer, steer_rate, yaw_rate, rear_rate,
    front_rate], with wheel shaft rates using the existing forward sign convention.
    """
    m = jp.asarray(measurement); raw = jp.asarray(raw); rates = jp.asarray(raw_rates)
    return jp.stack((m[0], m[1], m[2], m[3], forward_speed, m[4],
        m[5]*.1, m[6]*.1, raw[0], raw[1], rates[0], rates[1],
        previous_governed[0], previous_governed[1], previous_final_command[0],
        previous_final_command[1], previous_bounded_residual[0],
        previous_bounded_residual[1], eso_equilibrium_shift,
        raw[1]/q_ratio(raw[0], cc)))


def push_history(history, frame):
    return History(jp.concatenate((history.frames[1:], jp.asarray(frame)[None]), axis=0),
                   jp.concatenate((history.mask[1:], jp.ones_like(history.mask[:1])), axis=0))


def assemble_observation(history, alpha, epsi, chi, g, settle_clock, offsets, spec):
    """Return (345-vector, clipped fraction, policy fault); no value is sanitized."""
    n = spec['network']; scales = jp.asarray([f['scale'] for f in n['frame_fields']])
    context = jp.stack((alpha, epsi, jp.sin(epsi), jp.cos(epsi), chi, g,
                        settle_clock, offsets[0], offsets[1]))
    context_scales = jp.asarray([f['scale'] for f in n['context_fields']])
    normalized = jp.concatenate(((history.frames/scales).reshape(-1),
                                  history.mask, context/context_scales))
    # Masks and boolean context fields are already in {0,1}; clip only finite
    # normalized numeric values. A NaN remains a NaN and signals policy_fault.
    clip = n['finite_input_clip_abs']
    obs = jp.clip(normalized, -clip, clip)
    finite = jp.isfinite(normalized)
    fraction = jp.sum((jp.abs(normalized) > clip) & finite)/normalized.size
    return obs, fraction, ~jp.all(finite)


class DirectCommandActor(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = obs
        for width in (128, 128, 64):
            x = nn.Dense(width)(x)
            # Match torch.nn.functional.elu for export parity.
            x = jp.where(x > 0, x, jp.exp(x)-1.)
        return nn.Dense(2)(x)


def initialize_policy(policy):
    """Initialize RSL ActorCritic with the V3 zero-mean and orthogonal contract."""
    actors = [m for m in policy.actor.modules() if isinstance(m, tnn.Linear)]
    critics = [m for m in policy.critic.modules() if isinstance(m, tnn.Linear)]
    if [(m.in_features, m.out_features) for m in actors] != [
        (345, 128), (128, 128), (128, 64), (64, 2)]:
        raise ValueError('RSL actor architecture differs from V3 contract')
    if [(m.in_features, m.out_features) for m in critics] != [
        (346, 128), (128, 128), (128, 64), (64, 1)]:
        raise ValueError('RSL critic architecture differs from V3 contract')
    with torch.no_grad():
        for layer in actors[:-1] + critics[:-1]:
            tnn.init.orthogonal_(layer.weight, gain=2.**.5)
            tnn.init.zeros_(layer.bias)
        tnn.init.zeros_(actors[-1].weight); tnn.init.zeros_(actors[-1].bias)
        tnn.init.orthogonal_(critics[-1].weight, gain=1.)
        tnn.init.zeros_(critics[-1].bias)
        if not hasattr(policy, 'log_std') or tuple(policy.log_std.shape) != (2,):
            raise ValueError('RSL policy must have a two-dimensional log_std')
        policy.log_std.fill_(float(np.log(.2)))
    return policy


def export_actor(policy):
    """Export RSL actor mean weights as Flax parameters, without action tanh."""
    layers = [m for m in policy.actor.modules() if isinstance(m, tnn.Linear)]
    if [(m.in_features, m.out_features) for m in layers] != [
        (345, 128), (128, 128), (128, 64), (64, 2)]:
        raise ValueError('RSL actor architecture differs from V3 contract')
    return {'params': {f'Dense_{i}': {
        'kernel': jp.asarray(m.weight.detach().cpu().numpy().T),
        'bias': jp.asarray(m.bias.detach().cpu().numpy())}
        for i, m in enumerate(layers)}}
