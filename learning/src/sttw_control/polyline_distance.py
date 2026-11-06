"""Exact point-to-polyline distance using a midpoint spatial index."""
import numpy as np
from scipy.spatial import cKDTree


class PolylineDistance:
    def __init__(self, route):
        segments = np.diff(route, axis=0)
        length2 = np.sum(segments * segments, axis=-1)
        keep = length2 > 0
        self.starts = route[:-1][keep]
        self.segments = segments[keep]
        self.length2 = length2[keep]
        self.tree = cKDTree(self.starts + self.segments / 2)
        self.max_half_length = np.sqrt(self.length2.max()) / 2

    def __call__(self, xy):
        flat = xy.reshape(-1, 2)
        result = np.empty(len(flat))
        count = len(self.starts)
        for begin in range(0, len(flat), 4096):
            points = flat[begin:begin + 4096]
            pending = np.arange(len(points))
            k = min(16, count)
            while len(pending):
                centers, ids = self.tree.query(points[pending], k=list(range(1, k + 1)))
                delta = points[pending, None, :] - self.starts[ids]
                u = np.clip(np.sum(delta * self.segments[ids], axis=-1) / self.length2[ids], 0, 1)
                best = np.sqrt(np.sum((delta - u[..., None] * self.segments[ids]) ** 2, axis=-1).min(-1))
                # Every unqueried segment is at least d(center)-half_length away.
                # Grow the exact query until none can improve the incumbent.
                settled = np.ones(len(pending), dtype=bool) if k == count else centers[:, -1] - self.max_half_length > best + 1e-12
                result[begin + pending[settled]] = best[settled]
                pending = pending[~settled]
                k = min(2 * k, count)
        return result.reshape(xy.shape[:-1])
