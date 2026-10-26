"""Explicit physical origin for task-relative evaluation events.

The original evaluations default to physical tick700. Real forward prefix
contexts must declare their later origin rather than firing pulses in the past.
"""


def physical_origin(context):
    tick=context.get('initial_physical_tick',700)
    if type(tick) is not int or tick<700 or (tick-700)%4:
        raise ValueError('physical origin must be a declared post700 upper-step boundary')
    return tick


def physical_pulse(case, context, dt=.005):
    if dt!=.005:raise ValueError('original lower timestep required')
    tick=case['pulse_onset_task_tick']
    if type(tick) is not int or tick<0:
        raise ValueError('nonnegative task-relative onset required')
    start=(physical_origin(context)+tick)*dt
    return [start,start+case['pulse_duration_s'],
            *[case['force_N']*d for d in case['world_direction']]]
