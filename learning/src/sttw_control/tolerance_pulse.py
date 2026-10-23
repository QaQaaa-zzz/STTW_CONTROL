"""Actual-pulse curriculum primitives; reserve guides sampling, not reward."""
import jax
import jax.numpy as j
from flax import struct

FORCES = j.array([-8., -6., -4., -2., 2., 4., 6., 8.])


def force_probabilities(plus, minus, valid, *, guided):
    """50% uniform support, 50% near directional reserve; invalid -> uniform."""
    uniform = j.ones(8) / 8
    usable = valid & j.isfinite(plus) & j.isfinite(minus) & (plus >= 0) & (minus >= 0)
    reserve = j.where(FORCES > 0, plus, minus)
    reserve = j.clip(j.where(usable, reserve, 0.), 0., 8.)
    weights = j.exp(-j.abs(j.abs(FORCES) - reserve) / j.maximum(1., reserve))
    mixed = .5 * uniform + .5 * weights / j.sum(weights)
    return j.where(usable & guided, mixed, uniform), (~usable) & guided


@struct.dataclass
class PulseState:
    enabled: object
    onset_tick: object
    key: object
    fired: object
    pulse: object
    force: object
    fallback: object


def schedule(seed, env_id, episode):
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), env_id), episode)
    enabled_key, onset_key, amplitude_key = jax.random.split(key, 3)
    # Control-aligned onset, uniform on [.8, 3.5]s at .02s resolution.
    return PulseState(jax.random.bernoulli(enabled_key, .5),
                      4 * jax.random.randint(onset_key, (), 40, 176),
                      amplitude_key, j.bool_(False), j.zeros(5), j.asarray(0.), j.bool_(False))


def activate(state, *, tick, absolute_time, heading, plus, minus, valid, guided):
    """Choose once at onset; freeze world direction for the whole 0.1s pulse."""
    def fire(s):
        probs, fallback = force_probabilities(plus, minus, valid, guided=guided)
        index = jax.random.categorical(s.key, j.log(probs))
        force = FORCES[index]
        vector = force * j.array([-j.sin(heading), j.cos(heading), 0.])
        pulse = j.concatenate([j.array([absolute_time, absolute_time + .1]), vector])
        return s.replace(fired=j.bool_(True), pulse=pulse, force=force, fallback=fallback)
    return jax.lax.cond(state.enabled & ~state.fired & (tick >= state.onset_tick), fire, lambda s:s, state)
