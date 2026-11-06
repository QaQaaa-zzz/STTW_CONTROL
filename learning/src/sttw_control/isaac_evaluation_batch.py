"""Lane packing and ordinary six-lane outputs for batched fixed-case evaluation."""
from copy import deepcopy
import numpy as np

ALPHAS = [0., .25, .5, .75, 1., 0.]


def batch_settings(settings, cases):
    result = deepcopy(settings)
    paths = list(dict.fromkeys(case['terrain_path'] for case in cases))
    result.update(num_envs=6*len(cases), trace_lanes=6*len(cases),
                  terrains={str(i): path for i, path in enumerate(paths)},
                  lane_terrain_ids=[paths.index(case['terrain_path']) for case in cases for _ in ALPHAS],
                  lane_alphas=ALPHAS*len(cases),
                  lane_speeds=[case['user_speed_m_s'] for case in cases for _ in ALPHAS],
                  alphas=ALPHAS*len(cases), zero_middle=([False]*5+[True])*len(cases))
    return result


def split_case(arrays, contract, case, case_index):
    """Preserve policy, control and optional physics-contact sampling axes."""
    lanes = slice(case_index*6, (case_index+1)*6)
    result = {}
    for key, value in arrays.items():
        axis = 2 if key in ('lower_trace', 'wheel_contact_points', 'wheel_contact_forces') else 1
        result[key] = np.take(value, range(lanes.start, lanes.stop), axis=axis)
    output = deepcopy(contract)
    output['batch_source'] = dict(output=case['batch_output'], lane_start=lanes.start, lane_stop=lanes.stop)
    settings = output['settings']
    for key in ('lane_terrain_ids', 'lane_alphas', 'lane_speeds'):
        settings.pop(key, None)
    settings.update(num_envs=6, trace_lanes=6, alphas=ALPHAS,
                    zero_middle=[False]*5+[True], user_speed_m_s=case['user_speed_m_s'],
                    terrains={'evaluation': case['terrain_path']})
    for key in ('origin', 'terrain'):
        output[key] = output[key][lanes]
    for key in ('randomization_readback', 'delay_steps'):
        if key in output:
            output[key] = {name: value[lanes] for name, value in output[key].items()}
    if 'lab_physics' in output:
        meta = output['lab_physics']
        meta['num_envs'] = 6
        for key in ('env_origins', 'masses', 'materials'):
            meta[key] = meta[key][lanes]
        meta['terrain_layout']['clone_origins'] = meta['terrain_layout']['clone_origins'][lanes]
    return result, output
