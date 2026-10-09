"""Reference checks for V5.1 reward changes; no vehicle simulation.

All inputs are post-physics measurements. e_heading = reference_yaw - actual_yaw.
The returned quantities are nonnegative cost rates, NOT actuator commands.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path


def huber(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("nonfinite loss input")
    a = abs(x)
    return x * x if a <= 1 else 2 * a - 1


@dataclass(frozen=True)
class Result:
    primary_excess_raw: float
    primary_excess_effective: float
    yaw_recovery_raw: float
    yaw_recovery_effective: float
    yaw_error_rate_target: float


def reward_changes(*, alpha: int, ev: float, ed: float, e_heading: float,
                   yaw_rate: float, raw_yaw_rate: float, chi: float, g: float) -> Result:
    if alpha not in (0, 1):
        raise ValueError("alpha must be 0 or 1")
    if not all(math.isfinite(v) for v in (ev, ed, e_heading, yaw_rate, raw_yaw_rate, chi, g)):
        raise ValueError("nonfinite measurement")
    if not (0 <= chi <= 1 and 0 <= g <= 1):
        raise ValueError("chi and g must be in [0,1]")
    # Extra primary penalty is intentionally only active during raw-command conflict,
    # not during heading recovery. Existing V5 physical tracking costs remain.
    steer_excess = huber(max(abs(ed) - 0.03, 0.0) / 0.05)
    speed_excess = huber(max(abs(ev) - 0.05, 0.0) / 0.10)
    p = 40.0 * chi * (1 - g) * ((1 - alpha) * steer_excess + alpha * speed_excess)
    # This replaces V5 yaw_damping (does NOT supplement it). It shapes how rapidly
    # heading debt should contract without providing an action to the controller.
    error_rate_target = max(-0.4, min(0.4, 0.8 * e_heading))
    y = 2.0 * g * huber(((yaw_rate - raw_yaw_rate) - error_rate_target) / 0.20)
    return Result(p, min(p, 400.0), y, min(y, 20.0), error_rate_target)


def self_test() -> dict:
    # Exact tolerances: no oversized punishment for already good primary tracking.
    common = dict(e_heading=0.0, yaw_rate=0.0, raw_yaw_rate=0.0, chi=1., g=0.)
    a = reward_changes(alpha=0, ev=-0.4, ed=0.08, **common)
    b = reward_changes(alpha=1, ev=-0.15, ed=-0.10, **common)
    assert abs(a.primary_excess_raw - 40) < 1e-10
    assert abs(b.primary_excess_raw - 40) < 1e-10
    assert reward_changes(alpha=0, ev=-0.4, ed=0.02, **common).primary_excess_raw == 0
    assert reward_changes(alpha=1, ev=-0.03, ed=-0.10, **common).primary_excess_raw == 0
    # Recovery must not lock the wheel to the raw zero-steering request.
    r = reward_changes(alpha=0, ev=0, ed=.08, e_heading=.8,
                       yaw_rate=.4, raw_yaw_rate=0, chi=0, g=1)
    assert r.primary_excess_raw == 0 and r.yaw_recovery_raw == 0
    stationary = reward_changes(alpha=0, ev=0, ed=0, e_heading=.8,
                                yaw_rate=0, raw_yaw_rate=0, chi=0, g=1)
    wrong = reward_changes(alpha=0, ev=0, ed=-.08, e_heading=.8,
                          yaw_rate=-.4, raw_yaw_rate=0, chi=0, g=1)
    assert stationary.yaw_recovery_raw == 6
    assert wrong.yaw_recovery_raw == 14
    for sign in (-1, 1):
        rr = reward_changes(alpha=1, ev=0, ed=0, e_heading=sign*.8,
                            yaw_rate=sign*.4, raw_yaw_rate=0, chi=0, g=1)
        assert rr.yaw_recovery_raw == 0
    caps_v5 = [100, 100, 300, 900, 40, 50, 50, 10, 10, 20, 20, 40, 40]
    assert sum(caps_v5) == 1680
    assert sum(caps_v5) + 400 == 2080
    ev_examples = [{"abs_speed_error": x,
                    "old_primary_cost": 8*huber(x/.1),
                    "added_cost": reward_changes(alpha=1, ev=-x, ed=0, **common).primary_excess_raw}
                   for x in (.03, .05, .10, .15, .20, .30)]
    ed_examples = [{"abs_steer_error": x,
                    "old_primary_cost": 8*huber(x/.05),
                    "added_cost": reward_changes(alpha=0, ev=0, ed=x, **common).primary_excess_raw}
                   for x in (.02, .03, .05, .08, .10, .15)]
    return {"checks_passed": True, "vehicle_simulation_run": False,
            "speed_examples": ev_examples, "steer_examples": ed_examples,
            "recovery_cost_stationary": stationary.yaw_recovery_raw,
            "recovery_cost_wrong_direction": wrong.yaw_recovery_raw,
            "cost_rate_bound": 2080.,
            "total_policy_transitions_two_endpoints": 2*512*128*100,
            "total_control_ticks_upper_bound": 2*512*128*100*4}


if __name__ == '__main__':
    result = self_test()
    p = Path(__file__).with_name('math_checks.json')
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
