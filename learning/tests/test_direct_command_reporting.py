"""Small trace contracts for the V3 review reporter (run by central meter)."""

import json
from pathlib import Path

import numpy as np

from sttw_control.direct_command_reporting import analyze_trace, generate_report


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
        actual_forward_speed=np.full(n, 2.3), actual_delta=zeros,
        phi=zeros, peak_roll=zeros, e_psi_unwrapped=zeros,
        scored_tick_reward=np.full(n, -.001), physical_failure=flags,
        policy_fault=np.zeros(n, dtype=bool), finite_task_end=np.full(n, n == 3200 and not failure),
        cap_fraction=zeros, chi=zeros, g=zeros,
    )


def test_complete_trace_has_last_two_second_window_and_recovery():
    metrics = analyze_trace(_trace(3200), SPEC, "main")
    assert metrics["state"] == "complete"
    assert metrics["last_two_seconds"]["speed_rmse_m_s"] == 0
    assert metrics["recovery"]["final_hold_met"] is True
    assert metrics["reward_sum"] == np.sum(np.full(3200, -.001))


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
