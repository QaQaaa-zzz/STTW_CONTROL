"""Scalar reference for the PROPOSED STTW priority_return_v2 reward.

This is not an environment patch or a validated controller. q is supplied by
an independently audited causal phase state machine. No MuJoCo/PPO training is
performed here. The NumPy/JAX implementation in the repository must match it.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Sample:
    alpha: float
    speed_error: float       # post-action true body-forward speed - active external request
    lateral_error: float     # signed geometric error to immutable original path
    heading_error: float     # wrapped radians, original path tangent
    roll: float = 0.0
    roll_rate: float = 0.0
    action: tuple[float, float] = (0.0, 0.0)
    previous_action: tuple[float, float] = (0.0, 0.0)
    relaxation: float = 1.0   # q: 1 maneuver, 0 final strict bands
    dt: float = 0.005
    failed: bool = False
    missed_deadline_now: bool = False
    terminal_incomplete_now: bool = False
    task_penalty_already_paid: bool = False


def huber(z: float) -> float:
    z = abs(float(z))
    return z*z if z <= 1.0 else 2.0*z-1.0


def reward_terms(s: Sample) -> dict[str, object]:
    """Return unscaled cost rates and signed reward components.

    Failure=200 is an initial engineering setting, not a proof that early
    termination is suboptimal for arbitrary unbounded states. The handoff
    requires an episode-level termination-escape audit before long runs.
    """
    if s.alpha not in (0.0, 0.5, 1.0):
        raise ValueError('alpha must be exactly 0, 0.5, or 1')
    if len(s.action) != 2 or len(s.previous_action) != 2:
        raise ValueError('actions must contain steer and rear channels')
    numbers = [s.speed_error, s.lateral_error, s.heading_error, s.roll,
               s.roll_rate, s.relaxation, s.dt, *s.action, *s.previous_action]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError('nonfinite input: production environment must terminate safely')
    if not 0 <= s.relaxation <= 1 or s.dt <= 0:
        raise ValueError('invalid relaxation or dt')
    if any(abs(v) > 1 for v in (*s.action, *s.previous_action)):
        raise ValueError('normalized residual actions must be within [-1,1]')
    wp, wv = {0.0: (8.0, 0.1), 0.5: (4.0, 4.0), 1.0: (0.1, 8.0)}[s.alpha]
    ev, ey, ep = s.speed_error, s.lateral_error, s.heading_error
    under, over = max(-ev, 0.0), max(ev, 0.0)
    bu = .05 + .45*(1-s.alpha)*s.relaxation
    by = .10 + .30*s.alpha*s.relaxation
    bp = .15 + .20*s.alpha*s.relaxation
    raw = {
        'path': wp*(huber(ey/.10)+.3*huber(ep/.15)),
        'speed': wv*huber(ev/.05),
        'under_budget': 2*huber(max(under-bu, 0)/.05),
        'over_budget': 2*huber(max(over-.05, 0)/.05),
        'path_budget': 2*huber(max(abs(ey)-by, 0)/.10),
        'heading_budget': .6*huber(max(abs(ep)-bp, 0)/.15),
        'roll_excess': 20*huber(max(abs(s.roll)-.30, 0)/.10),
        'roll_rate': .2*huber(s.roll_rate/1.0),
        'action': .01*sum(x*x for x in s.action),
        'action_delta': .02*sum((x-y)**2 for x,y in zip(s.action,s.previous_action)),
    }
    # Do not put every cost through a common saturating denominator.
    parts = {k: -.1*s.dt*v for k,v in raw.items()}
    task_penalty = ((s.missed_deadline_now or s.terminal_incomplete_now)
                    and not s.task_penalty_already_paid)
    parts['task_incomplete'] = -20.0 if task_penalty else 0.0
    parts['failure'] = 0.0
    if s.failed:
        parts = {k: 0.0 for k in parts}
        parts['failure'] = -200.0
    return {'raw_costs': raw, 'reward_parts': parts, 'reward': sum(parts.values()),
            'bands': {'underspeed': bu, 'overspeed': .05, 'lateral': by, 'heading': bp}}


def self_test() -> dict[str, object]:
    """Algebraic preference tests; fabricated errors are NOT feasible trajectories."""
    samples = {'path_candidate': (-.35, .04),
               'balanced_candidate': (-.10, .10),
               'speed_candidate': (-.02, .30)}
    ranking = {}
    for alpha in (0.0, .5, 1.0):
        costs = {name: sum(reward_terms(Sample(alpha, ev, ey, 0))['raw_costs'].values())
                 for name,(ev,ey) in samples.items()}
        ranking[str(alpha)] = costs
    assert min(ranking['0.0'], key=ranking['0.0'].get) == 'path_candidate'
    assert min(ranking['0.5'], key=ranking['0.5'].get) == 'balanced_candidate'
    assert min(ranking['1.0'], key=ranking['1.0'].get) == 'speed_candidate'
    for alpha in (0.0, .5, 1.0):
        last = -1.0
        for ey in (0, .05, .1, .2, .4, 1.0, 2.0):
            c = sum(reward_terms(Sample(alpha, -.1, ey, 0))['raw_costs'].values())
            assert c > last
            last = c
        assert reward_terms(Sample(alpha, 0, 0, 0))['reward'] == 0
        assert reward_terms(Sample(alpha, 0, 0, 0, failed=True))['reward'] == -200
        b = reward_terms(Sample(alpha, 0, 0, 0, relaxation=0))['bands']
        assert b == {'underspeed': .05, 'overspeed': .05, 'lateral': .10, 'heading': .15}
    first = reward_terms(Sample(1., 0, 0, 0, missed_deadline_now=True))['reward']
    repeated = reward_terms(Sample(1., 0, 0, 0, terminal_incomplete_now=True,
                                  task_penalty_already_paid=True))['reward']
    assert first == -20 and repeated == 0
    one = reward_terms(Sample(.5, -.1, .2, .1, dt=.02))['reward']
    four = 4*reward_terms(Sample(.5, -.1, .2, .1, dt=.005))['reward']
    assert math.isclose(one, four, rel_tol=1e-12)
    return {'algebra_tests_passed': True, 'candidate_costs': ranking,
            'scope': 'reward arithmetic only; no dynamics, feasibility, or RL success claim'}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), indent=2))
