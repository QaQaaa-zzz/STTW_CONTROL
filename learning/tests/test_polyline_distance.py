import numpy as np

from sttw_control.polyline_distance import PolylineDistance


def test_exact_distance_with_distant_midpoints_and_repeated_vertices():
    # The nearest segment can have a distant midpoint: fixed nearest-k is unsafe.
    route = np.column_stack((np.linspace(0, 1, 100), np.ones(100)))
    route = np.concatenate((route, [[100, 100], [-100, 100], [-100, 0], [100, 0], [100, 0]]))
    points = np.array([[0., .1], [.5, 1.], [150., 0.], [0., 80.]])
    np.testing.assert_allclose(PolylineDistance(route)(points), [.1, 0., 50., 20.], atol=1e-12)


def test_exact_distance_agrees_with_all_segments():
    rng = np.random.default_rng(42)
    route = np.cumsum(rng.normal(size=(100, 2)), axis=0)
    points = rng.normal(size=(100, 2)) * 20
    segments = np.diff(route, axis=0)
    delta = points[:, None] - route[:-1]
    u = np.clip(np.sum(delta * segments, axis=-1) / np.sum(segments ** 2, axis=-1), 0, 1)
    expected = np.sqrt(np.sum((delta - u[..., None] * segments) ** 2, axis=-1).min(-1))
    np.testing.assert_allclose(PolylineDistance(route)(points), expected, atol=1e-12)
