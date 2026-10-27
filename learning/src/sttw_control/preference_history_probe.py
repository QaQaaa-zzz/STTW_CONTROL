"""Host-side direction contract for paired complete-state interventions."""
import numpy as np


def pulse_directions(cases, headings):
    """Use declared unit world vectors, or preserve the historical local rule."""
    headings = np.asarray(headings, dtype=float)
    if headings.shape != (len(cases),) or not np.isfinite(headings).all():
        raise ValueError('one finite capture heading required per case')
    directions = np.column_stack((-np.sin(headings), np.cos(headings), np.zeros(len(cases))))
    for i, row in enumerate(cases):
        if 'world_direction' in row:
            vector = np.asarray(row['world_direction'], dtype=float)
            if (vector.shape != (3,) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1., atol=1e-6)
                    or abs(vector[2]) > 1e-8):
                raise ValueError('world_direction must be a horizontal unit vector')
            directions[i] = vector
    return directions
