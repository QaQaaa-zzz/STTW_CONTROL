"""Algebraic reflection check of the actual deterministic control interface.

Synthetic inputs are NOT reachable-state evidence. Learned policies and MuJoCo
meshes are not reflected or tested here. No simulation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np

from sttw_control.controller import ControllerConfig, ControllerState, initial_controller
from sttw_control.actuator import ActuatorConfig, ActuatorState
from sttw_control.closed_loop_kernel import controls


def audit(reference, output):
    manifest = json.loads((reference / 'bank3/manifest.json').read_text())
    declaration = Path(manifest['plan']['source_declaration'])
    source = json.loads(declaration.read_text())['source']
    cc, ac = ControllerConfig(**source['controller']), ActuatorConfig(**source['actuator'])
    assert cc.fixed_roll_reference is None
    run = jax.jit(lambda c, a, m, raw, goal, eso, lower:
                  controls(c, a, m, raw, goal, eso, False, cc, ac, lower_action=lower))
    rng = np.random.default_rng(20261024)
    sign_m = j.array([-1., -1., -1., -1., -1., 1.])
    sign_command, sign_action = j.array([1., -1.]), j.array([-1., 1.])
    errors, saturated = [], 0
    for speed in (.8, 1.5, 1.8, 2.55):
        for steer in (0., .2, .79, .82):
            for eso in (False, True):
                cs = initial_controller(cc).replace(eso=j.array(rng.normal(0, .1, 4)),
                                                     disturbance=j.asarray(.3))
                actuator = ActuatorState(j.tile(j.array([.2, 15.]), (ac.delay_steps + 1, 1)), j.array([.2, 15.]))
                m = j.array([.12, -.3, steer, .4, .2, speed / .1])
                raw, goal = j.array([speed, .15]), j.array([speed + .15, .11])
                lower = j.array([.7, -.2])
                left = run(cs, actuator, m, raw, goal, eso, lower)
                mirrored_cs = ControllerState(-cs.eso, -cs.disturbance, cs.gains)
                mirrored_actuator = ActuatorState(actuator.pending * sign_action, actuator.previous * sign_action)
                right = run(mirrored_cs, mirrored_actuator, m * sign_m,
                            raw * sign_command, goal * sign_command, eso, lower * sign_action)
                expected_cs = ControllerState(-left[0].eso, -left[0].disturbance, left[0].gains)
                expected_actuator = ActuatorState(left[1].pending * sign_action, left[1].previous * sign_action)
                for actual, expected in zip(jax.tree.leaves(right[:2]), jax.tree.leaves((expected_cs, expected_actuator))):
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                    errors.append(float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
                for key, value in left[2].items():
                    expected = value if value.dtype == j.bool_ else value * sign_action
                    np.testing.assert_allclose(right[2][key], expected, atol=1e-6, rtol=1e-6)
                saturated += bool(left[2]['final_command_clipped'])
    files = [Path(__file__), declaration]
    base = Path(__file__).resolve().parents[4]
    files += [base / 'learning/src/sttw_control' / name for name in ('controller.py', 'actuator.py', 'closed_loop_kernel.py')]
    output.mkdir(exist_ok=False)
    (output / 'receipt.json').write_text(json.dumps(dict(
        synthetic_cases=32, cases_with_final_command_clipped=saturated,
        max_state_reflection_error=max(errors), new_physical_ticks=0, fits=0,
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        conclusion='Actual ECBC/ESO, command composition and actuator numerical interface admit this reflection on these synthetic inputs. Not proof of plant/policy/complete-history equivariance or mirrored recovery.'), indent=2) + '\n')
    print(output / 'receipt.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.reference.resolve(), args.output.resolve())
