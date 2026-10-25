"""Prepare a fixed DEV case inventory; does not execute or authorize physics."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source):
    cases = json.loads((source / 'cases.json').read_text())
    group = json.loads((source / 'groups.json').read_text())[0]
    slots = { (cases[i]['role'], cases[i]['alpha']): slot
              for slot, i in enumerate(group['case_indices'])
              if cases[i]['route'] == 'right90_R1p1' }
    parent = next(c for c in cases if c['route'] == 'right90_R1p1'
                  and c['role'] == 'parent' and c['alpha'] == 0.)
    direction = parent['world_direction']
    assert abs(sum(x*x for x in direction) - 1.) < 1e-10
    roles = [('parent', 'parent', 0, 1.),
             ('task_only', 'task_only', 0, 1.),
             ('physical_credit', 'physical_credit', 0, 1.),
             ('floor', 'parent', 0, 1.5),
             ('speed_gate', 'parent', 1, 1.),
             ('steer_gate', 'parent', 2, 1.),
             ('joint_gate', 'parent', 3, 1.)]
    combinations = [(name, model, profile, floor, alpha, False)
                    for name, model, profile, floor in roles for alpha in [0., .5]]
    combinations += [('zero_middle', 'parent', 0, 1., None, True),
                     ('causal_eso', 'parent', 4, 1., 0., False)]
    events = [('zero', 0., 420)] + [
        (f't{tick}_F{force:+g}', force, tick)
        for tick in [420, 520] for force in [-32., 32.]]
    rows = []
    for name, model, profile, floor, alpha, zero in combinations:
        for event, force, tick in events:
            rows.append(dict(case_id=f'{name}/alpha{alpha}/{event}', role=name,
                actor_role=model, alpha=alpha, state_alpha=0. if alpha is None else alpha,
                source_batch_slot=slots[(model, 0. if alpha is None else alpha)],
                profile=profile, reference_speed_floor_m_s=floor, zero_mode=zero,
                event_id=event, force_N=force, pulse_onset_task_tick=tick,
                pulse_duration_s=.1, world_direction=direction,
                route='right90_R1p1', requested_speed_m_s=1.5,
                horizon_s=10., stage='nominal' if force == 0. else 'disturbed'))
    assert len(rows) == 80 and len({r['case_id'] for r in rows}) == 80
    assert sum(r['stage'] == 'nominal' for r in rows) == 16
    assert sum(r['stage'] == 'disturbed' for r in rows) == 64
    initial = source / group['group_id'] / 'initial.pkl'
    return dict(project='STTW_CONTROL', execution_status='prepared_not_executed',
        question='Does progress-gated legal reference coordination improve physical outcomes at acceptable original-alpha task cost?',
        source_run=str(source.resolve()), source_initial_state=str(initial.resolve()),
        source_initial_state_sha256=sha(initial),
        source_cases_sha256=sha(source/'cases.json'),
        source_groups_sha256=sha(source/'groups.json'),
        maximum_active_lower_ticks=160000, optimizer_updates=0,
        nominal_cases=16, disturbed_cases=64, cases=rows,
        alias=dict(causal_eso_alpha_0p5='parent_alpha_0p5'),
        staging='Run nominal first and report original-alpha costs before disturbance stage; do not silently delete rejected candidates or enlarge the library.',
        deployment_information='No actual future event input to actor or reference gate. Pulse fields are integrator inputs only.',
        inference_limit='One related initial context, two alpha values, deterministic DEV inventory. No independent statistical replication.',
        attribution_limit='Opportunity screen only. Pure-task matched candidate selection and curvature-rule comparisons remain necessary before claiming an anti-disturbance selection mechanism.',
        selection='One candidate per observable context across designed events, separately from eventwise privileged oracle.')


if __name__ == '__main__':
    source = Path('runs/live_path_rejoin_pilot_v3')
    target = Path(__file__).with_name('case_inventory.json')
    inventory = build(source)
    target.write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(cases=len(inventory['cases']), nominal=16, disturbed=64,
                          physics_steps=0, output=str(target), sha256=sha(target))))
