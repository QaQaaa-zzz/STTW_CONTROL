"""Lane packing and ordinary six-lane outputs for batched fixed-case evaluation."""
from copy import deepcopy
import numpy as np

ALPHAS = [0., .25, .5, .75, 1., 0.]


def batch_settings(settings, cases):
    result = deepcopy(settings)
    paths = list(dict.fromkeys(case['terrain_path'] for case in cases))
    offset=0
    for case in cases:
        case['lane_start']=offset
        case['lane_count']=len(case.get('alphas',ALPHAS))
        offset+=case['lane_count']
    result.update(num_envs=offset, trace_lanes=offset,
                  terrains={str(i):path for i,path in enumerate(paths)},
                  lane_terrain_ids=[paths.index(case['terrain_path']) for case in cases for _ in range(case['lane_count'])],
                  lane_alphas=[a for case in cases for a in case.get('alphas',ALPHAS)],
                  lane_speeds=[case['user_speed_m_s'] for case in cases for _ in range(case['lane_count'])],
                  alphas=[a for case in cases for a in case.get('alphas',ALPHAS)],
                  zero_middle=[z for case in cases for z in case.get('zero_middle',[False]*5+[True])])
    if 'zero_reference_channels' in cases[0]:
        result['zero_reference_channels']=[mask for case in cases for mask in case['zero_reference_channels']]
    if 'evaluation_pulse' in cases[0]:
        result['evaluation_pulse']=deepcopy(cases[0]['evaluation_pulse'])
        result['evaluation_pulse']['world_force_N']=[case['evaluation_pulse']['world_force_N'] for case in cases for _ in range(case['lane_count'])]
    if cases[0].get('randomization') is not None:
        groups={}
        for i,case in enumerate(cases):
            cfg=case['randomization']
            fixed=('friction_range','mass_scale','base_com_m','latency_s','servo_gain_scale','steer_zero_offset_rad')
            if any(cfg[k][0]!=cfg[k][1] for k in fixed) or any(cfg['measurement_noise']) or any(any(v) for v in cfg['push_velocity_range'].values()):
                raise ValueError('Mixed-case batches require frozen parameter values and no observation noise or pushes')
            group=groups.setdefault(case['seed'],dict(config=deepcopy(cfg),env_ids=[]))
            group['env_ids'].extend(range(case['lane_start'],case['lane_start']+case['lane_count']))
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
                    zero_middle=case.get('zero_middle',([False]*5+[True])[:count]), user_speed_m_s=case['user_speed_m_s'],
                    terrains={'evaluation': case['terrain_path']})
    for key in ('evaluation_pulse','zero_reference_channels','case_labels'):
        if key in case:settings[key]=deepcopy(case[key])
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
