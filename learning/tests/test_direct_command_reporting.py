"""Small trace contracts for the V3 review reporter (run by central meter)."""

import json
from pathlib import Path

import numpy as np

from sttw_control.direct_command_reporting import (
    _main_claims, analyze_trace, generate_report, rescore_baseline_alpha1,
)


SPEC = json.loads(
    (Path(__file__).parents[1] / "configs" / "STTW_Direct_Command_V3.json").read_text()
)


def _trace(n, *, failure=False):
    t = np.arange(n) * .005
    raw = np.tile([2.3, 0.], (n, 1))
    zeros = np.zeros(n)
    flags = np.zeros(n, dtype=bool)
    if failure:
        flags[-1] = True
    return dict(
        time=t, active_tick=np.ones(n, dtype=bool), limited_command=raw,
        actual_forward_speed=np.full(n, 2.3), actual_delta=zeros.copy(),
        phi=zeros.copy(), peak_roll=zeros.copy(), e_psi_unwrapped=zeros.copy(),
        scored_tick_reward=np.full(n, -.001), physical_failure=flags,
        policy_fault=np.zeros(n, dtype=bool), finite_task_end=np.full(n, n == 3200 and not failure),
        cap_fraction=zeros.copy(), chi=zeros.copy(), g=zeros.copy(),
        actual_normalized_residual=np.tile([.2, -.3], (n, 1)),
        applied_residual=np.tile([.5, -1.0], (n, 1)),
        raw_cost_speed=np.full(n, 2.), effective_cost_speed=np.full(n, 1.),
    )


def test_complete_trace_has_last_two_second_window_and_recovery():
    metrics = analyze_trace(_trace(3200), SPEC, "main")
    assert metrics["state"] == "complete"
    assert metrics["last_two_seconds"]["speed_rmse_m_s"] == 0
    assert metrics["recovery"]["final_hold_met"] is True
    assert metrics["reward_sum"] == np.sum(np.full(3200, -.001))
    assert np.isclose(metrics["residuals"]["realized_final_command_change"]["front_peak_abs_rad_s"], .3)
    assert np.isclose(metrics["residuals"]["realized_final_command_change"]["rear_peak_abs_rad_s"], 3.)
    assert metrics["residuals"]["bounded_additive_request"]["front_peak_abs_rad_s"] == .5


def test_failure_invalidates_final_hold_and_truncated_window():
    metrics = analyze_trace(_trace(700, failure=True), SPEC, "main")
    assert metrics["state"] == "physical_failure"
    assert metrics["main_conflict"] is None
    assert metrics["recovery"]["final_hold_met"] is False


def test_missing_methods_remain_missing_and_report_is_explicit(tmp_path):
    root = tmp_path / "run"
    path = root / "review" / "main"
    path.mkdir(parents=True)
    np.savez_compressed(path / "B0.npz", **_trace(12))
    report = generate_report(root, SPEC)
    assert report["cases"]["main"]["methods"]["B0"]["state"] == "partial"
    assert report["cases"]["main"]["methods"]["pi_alpha0"]["state"] == "missing"
    assert report["claims"]["main_preference"] == "unavailable"
    assert "| pi_alpha0 | missing |" in (root / "review" / "REPORT.md").read_text()
    assert (path / "B0_steps.csv").exists()
    assert (root / "review" / "main_rewards.png").exists()
    assert (root / "review" / "main_rewards.pdf").exists()
    assert (root / "review" / "main_B0_components.png").exists()
    assert (root / "review" / "main_B0_components.pdf").exists()


def _declared_gate_example():
    t = np.arange(3200) * .005
    speed = np.full(3200, 2.3)
    # Exactly 0.2 s meets the drop gate, while the 2 s mean drop is only 0.02.
    speed[500:540] = 2.1
    trace = dict(time=t, limited_command=np.tile([2.3, 0.], (3200, 1)),
                 actual_forward_speed=speed, actual_delta=np.zeros(3200),
                 e_psi_unwrapped=np.zeros(3200))
    methods = {}
    for method, speed_rmse, steer_rmse, final_hold in (
            ("B0", .3, .1, False),
            ("pi_alpha0", .2, .04, False),
            ("pi_alpha1", .1, .08, False)):
        methods[method] = dict(state="complete", physical_failure=False,
                               peak_roll_abs_rad=.25,
                               main_conflict=dict(speed_rmse_m_s=speed_rmse,
                                                  steer_rmse_rad=steer_rmse,
                                                  actual_abs_steer_mean_rad=.2 if method == "pi_alpha1" else .25,
                                                  raw_abs_steer_mean_rad=.25),
                               recovery=dict(first_recovery_s=5., final_hold_met=final_hold))
    return methods, {"pi_alpha0": trace}


def test_declared_recovery_and_sustained_drop_do_not_require_extra_final_or_mean_gates():
    methods, traces = _declared_gate_example()
    claim = _main_claims(methods, traces, SPEC)
    assert claim["alpha0_speed_drop_sustained"] is True
    assert claim["alpha0_speed_drop_mean_m_s"] < .15
    assert claim["qualifying_recovery_both_policies"] is True
    assert claim["main_preference"] == "criteria_met"


def test_uninformative_baseline_does_not_earn_positive_preference_claim():
    methods, traces = _declared_gate_example()
    methods["B0"]["main_conflict"]["speed_rmse_m_s"] = .05
    methods["B0"]["main_conflict"]["steer_rmse_rad"] = .02
    assert _main_claims(methods, traces, SPEC)["main_preference"] == "not_informative_case"


def test_one_policy_without_qualifying_recovery_fails_gate():
    methods, traces = _declared_gate_example()
    methods["pi_alpha1"]["recovery"]["first_recovery_s"] = None
    claim = _main_claims(methods, traces, SPEC)
    assert claim["qualifying_recovery_both_policies"] is False
    assert claim["main_preference"] == "criteria_not_met"


def _scorable_baseline(n=12, failure=False):
    x = _trace(n, failure=failure)
    x["limited_command"][:, 0] = 2.3
    x["actual_forward_speed"][:] = 2.2
    x["actual_delta"][:] = .1
    x["chi"][:] = 1.
    x["g"][:] = 0.
    # alpha0: speed 0.8*H(-1)=0.8, steer 8*H(2)=24.
    # alpha1: speed 8, steer 0.8*H(2)=2.4.
    for component in ("speed", "steer", "heading", "roll", "roll_rate",
                      "overspeed", "low_speed", "correction", "command_change"):
        x[f"raw_cost_{component}"] = np.full(n, {"speed": .8, "steer": 24.}.get(component, 0.))
        x[f"effective_cost_{component}"] = x[f"raw_cost_{component}"].copy()
    x["raw_cost"] = np.full(n, 24.8)
    x["effective_cost"] = np.full(n, 24.8)
    x["scored_tick_reward"] = np.full(n, -.1 * .005 * 24.8)
    if failure:
        fail_tick = n - 1
        policy_step = fail_tick // 4
        remaining = 800 - policy_step
        terminal = -5 - .1 * .02 * 100 * (1 - .997 ** remaining) / (1 - .997)
        x["scored_tick_reward"][policy_step * 4:] = 0.
        x["scored_tick_reward"][fail_tick] = terminal
        x["failure_cost"] = np.zeros(n)
        x["failure_cost"][fail_tick] = -terminal
    return x


def test_baseline_alpha1_rescore_changes_weights_without_changing_physics():
    x = _scorable_baseline()
    rescored = rescore_baseline_alpha1(x, SPEC)
    assert np.isclose(rescored["raw_cost_speed"][0], 8.)
    assert np.isclose(rescored["raw_cost_steer"][0], 2.4)
    assert np.isclose(rescored["raw_cost"][0], 10.4)
    assert np.isclose(rescored["scored_tick_reward"][0], -.1 * .005 * 10.4)
    assert np.array_equal(rescored["actual_forward_speed"], x["actual_forward_speed"])


def test_baseline_rescore_preserves_whole_failed_policy_interval_replacement():
    x = _scorable_baseline(n=8, failure=True)
    rescored = rescore_baseline_alpha1(x, SPEC)
    np.testing.assert_array_equal(rescored["scored_tick_reward"][4:7], [0., 0., 0.])
    assert np.isclose(rescored["scored_tick_reward"][7], x["scored_tick_reward"][7])


def test_baseline_rescore_rejects_missing_or_inconsistent_components():
    x = _scorable_baseline()
    del x["raw_cost_steer"]
    with np.testing.assert_raises_regex(ValueError, "raw_cost_steer"):
        rescore_baseline_alpha1(x, SPEC)


def test_report_persists_audited_alpha1_baseline_from_same_physical_trace(tmp_path):
    case_dir = tmp_path / "review" / "main"
    case_dir.mkdir(parents=True)
    np.savez_compressed(case_dir / "B0.npz", **_scorable_baseline())
    report = generate_report(tmp_path, SPEC)
    assert report["cases"]["main"]["baseline_alpha1_rescore"]["state"] == "audited"
    with np.load(case_dir / "B0_rescored_alpha1.npz") as derived:
        assert np.isclose(derived["raw_cost"][0], 10.4)
        assert np.array_equal(derived["actual_forward_speed"], _scorable_baseline()["actual_forward_speed"])
    assert (case_dir / "B0_rescored_alpha1_steps.csv").exists()


def test_clip_fraction_ignores_float32_roundoff_but_counts_real_limit_crossing():
    x = _trace(10)
    x["requested_residual"] = np.tile([.3, -3.], (10, 1)).astype(np.float32)
    x["applied_residual"] = np.tile([.30000004, -3.], (10, 1)).astype(np.float32)
    x["residual_clipped"] = np.ones(10, dtype=bool)
    x["u_prelimit"] = np.tile([.7, 23.], (10, 1)).astype(np.float32)
    x["final_command"] = np.tile([.70000005, 23.], (10, 1)).astype(np.float32)
    x["final_command_clipped"] = np.ones(10, dtype=bool)
    metrics = analyze_trace(x, SPEC, "main")
    assert metrics["clip_fraction"]["residual_clipped"] == 0.
    assert metrics["clip_fraction"]["logged_residual_clipped"] == 1.
    assert metrics["clip_fraction"]["final_command_clipped"] == 0.
    assert metrics["clip_fraction"]["logged_final_command_clipped"] == 1.
    x["requested_residual"][0, 0] = 2.
    x["u_prelimit"][0, 0] = 4.
    x["final_command"][0, 0] = 3.
    metrics = analyze_trace(x, SPEC, "main")
    assert np.isclose(metrics["clip_fraction"]["residual_clipped"], .1)
    assert np.isclose(metrics["clip_fraction"]["final_command_clipped"], .1)
