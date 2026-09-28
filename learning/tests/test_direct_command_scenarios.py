"""Contract checks for V3 commands; intentionally left for the shared test budget."""

import json
from pathlib import Path

import jax.numpy as jp
import numpy as np

from sttw_control.direct_command_scenarios import publish_command, review_rows, schedule


SPEC = json.loads(
    (Path(__file__).parents[1] / "configs" / "STTW_Direct_Command_V3.json").read_text()
)


def test_schedule_is_fixed_shape_deterministic_and_folds_episode_identity():
    rows, family, slew = schedule(7, 2, 2.3, SPEC)
    again = schedule(7, 2, 2.3, SPEC)
    assert rows.shape == (16, 3)
    assert slew.shape == (2,)
    assert int(family) in (0, 1, 2)
    np.testing.assert_array_equal(rows, again[0])
    np.testing.assert_array_equal(slew, again[2])
    assert not np.array_equal(rows, schedule(7, 3, 2.3, SPEC)[0])
    assert np.all(np.asarray(rows)[np.asarray(rows)[:, 0] == 99, 1:] == 0)
    assert np.all(np.asarray(rows)[np.asarray(rows)[:, 0] < 99, 0] < 16)
    assert 0.3 <= float(slew[0]) <= 0.8
    assert 0.15 <= float(slew[1]) <= 0.45


def test_publish_uses_current_target_and_exact_boundary_slew():
    rows = jp.array([[0, 2.6, 0], [1.5, 2.6, .25], [4.5, 2.6, 0]] + [[99, 0, 0]] * 13)
    raw = jp.array([2.3, 0.])
    issued, rates, target = publish_command(raw, rows, 0, jp.array([.5, .3]), .005)
    np.testing.assert_allclose(target, [2.6, 0], atol=1e-7)
    np.testing.assert_allclose(issued, [2.3025, 0], atol=1e-6)
    np.testing.assert_allclose(rates, [.5, 0], atol=1e-5)
    issued, rates, target = publish_command(jp.array([2.6, 0]), rows, 300, jp.array([.5, .3]), .005)
    np.testing.assert_allclose(target, [2.6, .25], atol=1e-7)
    np.testing.assert_allclose(issued, [2.6, .0015], atol=1e-6)
    np.testing.assert_allclose(rates, [0, .3], atol=1e-5)
    _, _, target = publish_command(jp.array([2.6, .25]), rows, 900, jp.array([.5, .3]), .005)
    np.testing.assert_allclose(target, [2.6, 0], atol=1e-7)


def test_review_rows_freeze_main_and_random_without_global_rng():
    main, random, slew = review_rows(SPEC)
    np.testing.assert_allclose(main[:3], [[0, 2.6, 0], [1.5, 2.6, .25], [4.5, 2.6, 0]])
    np.testing.assert_allclose(slew, [.5, .3])
    assert random.shape == (16, 3)
    assert random[0, 0] == 0 and random[0, 1] == 2.3 and random[0, 2] == 0
    assert np.any(random[:, 0] == 8.)
    assert np.array_equal(random, review_rows(SPEC)[1])
