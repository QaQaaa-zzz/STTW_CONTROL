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


@struct.dataclass
class PulseEpisode:
    core: object
    pulse_state: object
    impulse: object

    @property
    def state(self): return self.core.state
    @property
    def route(self): return self.core.route
    @property
    def episode(self): return self.core.episode
    @property
    def env_id(self): return self.core.env_id


class PulseBatch:
    """Wrap existing task accounting; only add external-force episode state."""
    def __init__(self, base, *, guided, pulse_seed, reserve_config):
        self.base = base
        self.guided = guided
        self.pulse_seed = pulse_seed
        self.reserve_config = reserve_config
        if abs(base.env.cc.dt - .005) > 1e-9:
            raise ValueError('pulse schedule requires original .005s lower period')

    def reset(self, eid, episode):
        return PulseEpisode(self.base.reset(eid, episode),
                            schedule(self.pulse_seed, eid, episode), j.zeros(3))

    def observation(self, ep):
        return self.base.observation(ep.core)

    def advance_one(self, ep, z, *, auto_reset=True):
        from .tolerance_shaping import measurement_reserve
        from .path_command_statistics import merge_statistics, reset_statistics
        st = ep.state
        def onset(p):
            m, pose, _ = self.base.env.physics.observe(st.physical.data)
            reserve = measurement_reserve(m, st.physical.controller, self.base.env.cc,
                remaining=self.base.horizon-st.tick*self.base.env.cc.dt,
                config=self.reserve_config)
            return activate(p, tick=st.tick,
                absolute_time=st.physical.physical_tick*self.base.env.cc.dt,
                heading=pose[2], plus=reserve['plus_N'], minus=reserve['minus_N'],
                valid=reserve['valid'], guided=self.guided)
        p = ep.pulse_state
        p = jax.lax.cond(p.enabled & ~p.fired & (st.tick >= p.onset_tick), onset, lambda p:p, p)
        core, _, reward, flags, final, stats = self.base.advance_one(
            ep.core, z, auto_reset=False, force_pulse=p.pulse)
        ended = ep.replace(core=core, pulse_state=p,
                           impulse=ep.impulse+stats['applied_force_impulse'])
        # Export episode totals before reset; early termination retains partial impulse.
        stats['pulse'] = dict(fired_now=p.fired & ~ep.pulse_state.fired,
            force=p.force, fallback_now=p.fallback & ~ep.pulse_state.fired,
            completed=flags['done'], enabled=p.enabled,
            completed_impulse=j.where(flags['done'], ended.impulse, j.zeros(3)),
            completed_expected_impulse=j.where(flags['done'] & p.fired, .1*p.pulse[2:], j.zeros(3)))
        if auto_reset:
            nxt = jax.lax.cond(flags['done'] & ~flags['engineering_fault'],
                lambda _:self.reset(ep.env_id, ep.episode+1), lambda _:ended, None)
            stats['diagnostics'] = merge_statistics(stats['diagnostics'],
                reset_statistics(nxt.route, flags['done'] & ~flags['engineering_fault']))
            obs = self.observation(nxt)
        else:
            nxt, obs = ended, final
        return nxt, obs, reward, flags, final, stats
