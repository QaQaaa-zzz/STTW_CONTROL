"""Optional TRAIN-only pre-action observation records for recovery teaching."""
import numpy as np


def teaching_enabled(declaration):
    spec = declaration.get('teaching_records')
    if spec is None:
        return False
    if spec != {'role': 'TRAIN', 'observation_dim': 351}:
        raise ValueError('teaching records require explicit TRAIN role and 351D actor observations')
    if not declaration.get('recovery_intervention'):
        raise ValueError('teaching records require declared matched intervention')
    return True


def extract_teaching(data, upper_dt=.02):
    """Select first active physical tick of each decision; never shift to next obs."""
    time = np.asarray(data['time'])
    select = np.isclose(time / upper_dt, np.rint(time / upper_dt), atol=1e-4, rtol=0)
    select &= np.asarray(data['active_tick'], dtype=bool)
    result = {'time': time[select]}
    for source, target, width in (
        ('teaching_observation', 'observations', 351),
        ('teaching_target', 'targets', 2),
        ('teaching_parent', 'parent_means', 2),
    ):
        values = np.asarray(data[source])
        if values.shape != (len(time), width):
            raise ValueError('invalid teaching record shape: ' + source)
        result[target] = values[select]
        if not np.isfinite(result[target]).all():
            raise ValueError('nonfinite active teaching record: ' + source)
    if len(np.unique(np.rint(result['time']/upper_dt))) != len(result['time']):
        raise ValueError('duplicate policy decision')
    return result
