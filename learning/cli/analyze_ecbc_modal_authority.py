"""Algebra-only ECBC modal/gain audit; never imports a simulator or policy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(plan_file, output):
    plan = json.loads(plan_file.read_text())
    c = SimpleNamespace(**plan['initial_source']['controller'])
    source = Path(__file__).resolve().parents[1] / 'src/sttw_control/controller.py'
    # Run only the two existing coefficient functions with NumPy float64.
    # This checks algebra/source semantics, not production JAX float32 replay.
    tree = ast.parse(source.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ('_system', '_floor_magnitude')]
    assert len(body) == 2
    namespace = {'jp': np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    system = namespace['_system']
    inertia = c.mass * c.cg_height ** 2 + .043929 * c.mass / 5.434
    a4 = c.mass * c.gravity * c.cg_height / inertia
    lam = np.sqrt(a4)
    h, x, q = c.cg_height, c.cg_forward, c.cg_forward * c.trail * c.gravity
    middle = 2 * h * q + a4 * x * x * h * h
    roots_v2 = np.sort(np.roots([h * h, -middle, q * q]))
    roots = np.sqrt(roots_v2)
    previous = system(1., np.zeros(3), c)[-1]

    def values(v):
        a1, a2, a4, ratio, gains = system(float(v), previous, c)
        A = np.array([[0., 0., 0.], [0., 0., 1.], [a2, a4, 0.]])
        B = np.array([1., 0., a1])
        controllability = np.column_stack([B, A @ B, A @ A @ B])
        D = a4 * a1 * a1 - a2 * a2
        np.testing.assert_allclose(np.linalg.det(controllability), D, atol=1e-8, rtol=1e-10)
        return dict(speed_m_s=float(v), effective_speed_m_s=max(float(v), c.minimum_speed),
                    a1=float(a1), a2=float(a2), a4=float(a4), ratio=float(ratio),
                    determinant_s_minus4=float(D), gain_update=bool(abs(D) > 5.),
                    unstable_input_s_minus1=float(a1 + a2 / lam),
                    stable_input_s_minus1=float(a1 - a2 / lam),
                    gains=gains.tolist(),
                    pbh_unstable_rank=int(np.linalg.matrix_rank(np.column_stack([lam * np.eye(3) - A, B]))),
                    pbh_stable_rank=int(np.linalg.matrix_rank(np.column_stack([-lam * np.eye(3) - A, B]))),
                    controllability_rank=int(np.linalg.matrix_rank(controllability)),
                    ideal_no_eso_no_local_no_actuator_closed_poles=[
                        [float(z.real), float(z.imag)] for z in np.linalg.eigvals(A - np.outer(B, gains))])

    critical = float(roots[-1])
    def boundary(target):
        lo, hi = critical - .1, critical + .1
        for _ in range(70):
            mid = (lo + hi) / 2
            if values(mid)['determinant_s_minus4'] > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    guard = [boundary(5.), boundary(-5.)]
    speeds = np.unique(np.r_[np.linspace(c.minimum_speed, 3.5, 601), critical, guard])
    rows = [values(v) for v in speeds]
    root_row = values(critical)
    assert root_row['pbh_unstable_rank'] == 3 and root_row['pbh_stable_rank'] == 2
    assert root_row['controllability_rank'] == 2 and not root_row['gain_update']
    assert roots[0] < c.minimum_speed
    assert abs(root_row['unstable_input_s_minus1']) > 1.
    output.mkdir(parents=True, exist_ok=False)
    result = dict(controller=vars(c), inertia=inertia, lambda_s_minus1=float(lam),
        unclamped_formula_root_speeds_m_s=roots.tolist(), actual_high_root=root_row,
        historical_previous_gain_example_speed_m_s=1., previous_gains=previous.tolist(),
        high_root_gain_hold_interval_m_s=guard, rows=rows,
        source_sha256=digest(source), plan_sha256=digest(plan_file),
        new_physics_steps=0, new_training_steps=0,
        scope='NumPy float64 source algebra. Fixed-speed three-state model; '
              'not the ECBC/ESO/local300/actuator/plant trajectory or a safety region.')
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    axes[0].plot(speeds, [r['determinant_s_minus4'] for r in rows], color='#333333')
    axes[0].set_ylabel('det C (1/s^4)'); axes[0].set_yscale('symlog', linthresh=5)
    axes[1].plot(speeds, [r['unstable_input_s_minus1'] for r in rows], color='#D55E00', label='unstable +lambda')
    axes[1].plot(speeds, [r['stable_input_s_minus1'] for r in rows], '--', color='#0072B2', label='stable -lambda')
    axes[1].set_ylabel('Modal input (1/s)'); axes[1].legend()
    for k, ax in enumerate(axes[2:]):
        ax.plot(speeds, [r['gains'][k] for r in rows], color='#009E73')
        ax.set_ylabel(['K_delta (1/s)', 'K_roll (1/s)', 'K_rollrate (1)'][k])
        ax.set_yscale('symlog', linthresh=10)
    for ax in axes:
        ax.axvspan(*guard, color='#777777', alpha=.15)
        ax.axvline(critical, linestyle=':', color='black', linewidth=.8)
        ax.axhline(0, color='#777777', linewidth=.5); ax.grid(alpha=.15)
    axes[-1].set_xlabel('Coefficient-model speed (m/s)')
    fig.suptitle('ECBC algebra only: stable-mode loss is not loss of stabilizability\n'
                 'Grey = existing gain hold; previous gains from 1 m/s, not an actual speed trajectory\n'
                 'No ESO/local300/actuator/physics simulation; no disturbance threshold')
    fig.savefig(output / 'modal_and_gain.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(output=str(output), root_m_s=critical, gain_hold=guard,
                          unstable_input=root_row['unstable_input_s_minus1'], physics_steps=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.plan.resolve(), args.output.resolve())
