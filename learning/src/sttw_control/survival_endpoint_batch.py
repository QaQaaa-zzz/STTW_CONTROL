"""Shape-homogeneous endpoint groups and dynamic-parameter collection kernels."""
import math


def simulation_alpha(case):
    value=case['alpha']
    if value is None and case['inference_mode']=='zero':return 0.
    if value is None or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('legal simulation alpha required; only exact zero has an alpha alias')
    return float(value)


def partition_cases(cases, shape_by_route):
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('duplicate actual case IDs')
    groups={}
    for index,c in enumerate(cases):
        geometry='nominal' if c['panel'].startswith('nominal_') else 'physical'
        mode=c['inference_mode'];floor=c['speed_reference_floor_m_s']
        shape=shape_by_route[(geometry,c['route'])]
        key=(geometry,mode,floor,shape)
        if key not in groups:
            groups[key]=dict(geometry=geometry,inference_mode=mode,speed_reference_floor_m_s=floor,
                             path_shape=shape,case_indices=[])
        groups[key]['case_indices'].append(index)
    return [dict(group_id=f'group_{i:03d}',**g) for i,g in enumerate(groups.values())]


def make_endpoint_chunk(step, *, steps_per_chunk=50, zero_correction=False):
    """Weights/std, routes, requests and pulses are data, never model closures."""
    import jax
    if not isinstance(steps_per_chunk,int) or steps_per_chunk<=0:
        raise ValueError('positive integer chunk length required')
    axes=(0,None if zero_correction else 0,0,0,0,0)
    @jax.jit
    def chunk(carry,parameters,log_std,path,speed,pulse):
        return jax.lax.scan(lambda st,_:jax.vmap(step,in_axes=axes)(
            st,parameters,log_std,path,speed,pulse),carry,None,length=steps_per_chunk)
    return chunk
