"""CPU parity against the independently imported, pinned curriculum400 V3 source.

These integration checks require the original local reference artifacts. They
skip explicitly when those external artifacts are unavailable; no reference
formula is copied into the implementation or test.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control import path_command_reward as current


REFERENCE_ROOT = Path('/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1')
REFERENCE_SOURCE = REFERENCE_ROOT / 'learning/src/sttw_control/path_command_reward.py'
REFERENCE_MANIFEST = REFERENCE_ROOT / 'runs/path_distilled_v3_curriculum_400_20261021/training/manifest.json'
SOURCE_SHA256 = 'a1ee37554d1d4cca491b2696f70b41dc9852dc826a8bbf1317014bd371f57926'
MANIFEST_SHA256 = '14e645f95a64b8790beba142d8e17921fab7984cc45f7fdda58b15cd3cdb698f'


@pytest.fixture(scope='module')
def reference():
    for path, expected in ((REFERENCE_SOURCE, SOURCE_SHA256),
                           (REFERENCE_MANIFEST, MANIFEST_SHA256)):
        if not path.is_file():
            pytest.skip(f'external V3 parity artifact unavailable: {path}')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    spec = importlib.util.spec_from_file_location('independent_curriculum400_v3_reward', REFERENCE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(REFERENCE_MANIFEST.read_text())['config']
    assert config['path_reward']['reward_version'] == 'path_persistent_preference_v3'
    assert not current.tolerance_enabled(config)
    assert module is not current
    assert Path(module.__file__).resolve() != Path(current.__file__).resolve()
    return module, config


def assert_cost_parity(expected, actual):
    for field in ('raw', 'effective', 'capped', 'endpoint_raw',
                  'endpoint_effective', 'endpoint_capped'):
        assert set(expected[field]) == set(actual[field]), field
        for key in expected[field]:
            np.testing.assert_array_equal(actual[field][key], expected[field][key],
                                          err_msg=f'{field}.{key}')
    np.testing.assert_array_equal(actual['reward'], expected['reward'])
    assert np.isfinite(np.asarray(actual['reward'])).all()


@pytest.mark.parametrize('alpha', [0., .01, .25, .5, .75, .99, 1.])
def test_v3_random_and_extreme_costs_match_original(reference, alpha):
    module, config = reference
    rng = np.random.default_rng(20261022)
    count = 144
    data = dict(alpha=np.full(count, alpha), chi=rng.uniform(0., 1., count),
                ev=rng.normal(0., 1., count), ey=rng.normal(0., 1., count),
                eh=rng.normal(0., 1., count), peak_roll=rng.uniform(0., .7, count),
                roll_rate=rng.normal(0., 3., count), speed=rng.uniform(-2., 8., count),
                offsets=rng.normal(0., 2., (count, 2)),
                offset_rate=rng.normal(0., 20., (count, 2)))
    # Zero errors, signed extremes, and exact Huber/primary-band boundaries.
    extremes = [0., -1e5, 1e5, -.1, .1, -.05, .05, -.2, .2]
    for key in ('ev', 'ey', 'eh', 'roll_rate', 'speed'):
        data[key][:len(extremes)] = extremes
    data['chi'][:12] = np.tile([0., .25, .5, 1.], 3)
    data['peak_roll'][:9] = [0., .26, .3, .34, .7, 1e5, .259, .299, .301]
    data['offsets'][:3] = [[0., 0.], [-1e5, 1e5], [1e5, -1e5]]
    data['offset_rate'][:3] = data['offsets'][:3]
    data = {key: j.asarray(value, dtype=j.float32) for key, value in data.items()}
    evaluate = lambda function: jax.vmap(lambda row: function(**row, config=config))(data)
    expected, actual = evaluate(module.costs), evaluate(current.costs)
    assert_cost_parity(expected, actual)
    # Exercise both sides of every component cap, including endpoint flags.
    for key, flags in actual['capped'].items():
        assert np.asarray(flags).any(), key
        assert not np.asarray(flags).all(), key


@pytest.mark.parametrize('remaining', [0, 1, 2, 64, 512, 1000])
def test_v3_failure_reward_matches_original(reference, remaining):
    module, config = reference
    np.testing.assert_array_equal(current.failure_reward(remaining, config),
                                  module.failure_reward(remaining, config))


def test_v3_diagnostic_difference_is_zero_only(reference):
    module, config = reference
    values = dict(alpha=.5, chi=.7, ev=.2, ey=-.3, eh=.1, peak_roll=.32,
                  roll_rate=-1., speed=2., offsets=j.array([.1, -.1]),
                  offset_rate=j.array([.2, -.2]), config=config)
    expected, actual = module.costs(**values), current.costs(**values)
    assert set(expected) - set(actual) == {'diagnostic_subcosts'}
    assert not (set(actual) - set(expected))
    for value in expected['diagnostic_subcosts'].values():
        np.testing.assert_array_equal(value, 0.)
    assert_cost_parity(expected, actual)
