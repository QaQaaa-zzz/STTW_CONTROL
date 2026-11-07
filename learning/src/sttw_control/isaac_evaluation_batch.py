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
    if cases[0].get('randomization') is not None:
        groups={}
        for i,case in enumerate(cases):
            cfg=case['randomization']
            fixed=('friction_range','mass_scale','base_com_m','latency_s','servo_gain_scale','steer_zero_offset_rad')
            if any(cfg[k][0]!=cfg[k][1] for k in fixed) or any(cfg['measurement_noise']) or any(any(v) for v in cfg['push_velocity_range'].values()):
                raise ValueError('Mixed-case batches require frozen parameter values and no observation noise or pushes')
            group=groups.setdefault(case['seed'],dict(config=deepcopy(cfg),env_ids=[]))
            group['env_ids'].extend(range(6*i,6*i+6))
        result['fixed_randomization_groups']=list(groups.values())
        result['randomization']=deepcopy(cases[0]['randomization'])
        result['randomization']['latency_s']=[0,max(g['config']['latency_s'][1] for g in groups.values())]
        result['auto_reset_horizon']=False
    return result


def split_case(arrays, contract, case, case_index):
    """Preserve policy, control and optional physics-contact sampling axes."""
    count=case.get('lane_count',6)
    start=case.get('lane_start',case_index*count)
    lanes = slice(start,start+count)
    result = {}
    for key, value in arrays.items():
        axis = 2 if key in ('lower_trace', 'wheel_contact_points', 'wheel_contact_forces', 'executed_commands', 'body_poses') else 1
        result[key] = np.take(value, range(lanes.start, lanes.stop), axis=axis)
    output = deepcopy(contract)
    output['batch_source'] = dict(output=case['batch_output'], lane_start=lanes.start, lane_stop=lanes.stop)
    settings = output['settings']
    for key in ('lane_terrain_ids', 'lane_alphas', 'lane_speeds', 'fixed_randomization_groups'):
        settings.pop(key, None)
    settings.update(num_envs=count, trace_lanes=count, alphas=case.get('alphas',ALPHAS[:count]),
                    zero_middle=([False]*5+[True])[:count], user_speed_m_s=case['user_speed_m_s'],
                    terrains={'evaluation': case['terrain_path']})
    if case.get('randomization') is not None:
        settings.update(seed=case['seed'],randomization=deepcopy(case['randomization']))
    for key in ('origin', 'terrain'):
        output[key] = output[key][lanes]
    for key in ('randomization_readback', 'delay_steps'):
        if key in output:
            output[key] = {name: value[lanes] for name, value in output[key].items()}
    if 'lab_physics' in output:
        meta = output['lab_physics']
        meta['num_envs'] = count
        for key in ('env_origins', 'masses', 'materials'):
            meta[key] = meta[key][lanes]
        meta['terrain_layout']['clone_origins'] = meta['terrain_layout']['clone_origins'][lanes]
    return result, output
