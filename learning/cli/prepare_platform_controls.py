"""Prepare the paired locked-H75 approach and landing controls on the native step."""
import argparse
import json
from pathlib import Path

import numpy as np


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare(template, initial_bank, out):
    cfg = json.loads(template.read_text())
    hand = cfg['ground_handoff']
    base_world = json.loads(Path(cfg['world_map']).read_text())
    front = base_world['terrain']['front_x_m']
    grid = np.linspace(-5, 25, 1501)
    references = []
    for sign in (1, 0, -1):
        def bump(start, end):
            return np.sin(np.pi * np.clip((grid - start) / (end - start), 0, 1)) ** 2
        y = sign * .30 * bump(front - 5.0, front - .50) + .12 * bump(front + 2., front + 3.75)
        distance = np.r_[0., np.cumsum(np.hypot(np.diff(grid), np.diff(y)))]
        arc = np.arange(0., 30., .025)
        xy = np.stack([np.interp(arc, distance, axis) for axis in (grid, y)], axis=1)
        heading = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
        references.append((xy, heading, np.gradient(heading, arc)))
    route_file = out / 'routes.npz'
    np.savez_compressed(route_file, s=arc, **dict(zip(('xy', 'heading', 'curvature'), map(np.stack, zip(*references)))))
    template_variant = hand['handoff_variants'][0]
    variants = [dict(template_variant, bank_row=i, root_variant=i, requested_speed_m_s=2.,
                     approach_speed_m_s=2., continuation_speed_m_s=2., activation_offset_m=-1.1)
                for i in range(8)]
    common = dict(schema=cfg['schema'], role='paired platform DEV, nominal speed 2 m/s',
                  shared_adapters=True, native_apex_signal=True, contact_substep_telemetry=True,
                  audit_actuator_owner=True, ground_physics={'integrator':'RK4'},
                  planned_activation_offset_m=-1.1, approach_initial_bank=str(initial_bank),
                  evaluation_lanes=[[route, 'h75', root] for root in range(8) for route in range(3)])
    for test in (1, 2, 3):
        gap = .55 if test == 3 else 0.
        surface = 'level_gap' if test == 3 else 'native_step'
        terrain = dict(surface=surface, front_x_m=front, width_m=gap, far_bank_height_m=0. if gap else .16)
        goal = dict(base_world['goal'], x_min_m=front+2.1, x_max_m=front+3.85)
        world = dict(map_id=surface, bounds_xy_m=base_world['bounds_xy_m'], terrain=terrain, goal=goal,
                     forward_obstacle=dict(base_world['forward_obstacle'], obstacle_id=surface,
                                           activation_offset_m=-1.1, jump_signal_offsets_m=[-1.1,.4]))
        world_path = out / ('world_' + surface + '.json')
        write(world_path, world)
        setup = dict(hand, approach=dict(hand['approach'], speed_m_s=2.), start_vx_m_s=2.,
                     continuation=dict(hand['continuation'], speed_m_s=2.), reference_path=str(route_file),
                     handoff_variants=variants, retain_rb_after_entry=test==1, goal=goal)
        write(out / f'test{test}.json', dict(common, task_evaluation=True, world_map=str(world_path), ground_handoff=setup))
        if test == 2:
            speed_variants = [dict(v, requested_speed_m_s=speed, approach_speed_m_s=speed)
                              for speed in (1.9,2.1) for v in variants]
            lanes = [[r, 'h75', i] for i in range(len(speed_variants)) for r in range(3)]
            write(out / 'test2_speeds.json', dict(common, task_evaluation=True, world_map=str(world_path),
                  evaluation_lanes=lanes, ground_handoff=dict(setup,handoff_variants=speed_variants),
                  role='Entry speed sensitivity; continuation remains 2.0 m/s, 96 episodes, 8 s each.'))
    write(out / 'approach_contract.json', dict(tests=[1,2,3], episodes_each=48, horizon_s=8.,
          bank_rows=list(range(8)), requested_speed_m_s=2., entry_offset_m=-1.1,
          ground_integrator='RK4 1 ms, matched to Test 0 RK4 1 ms', route_order=['left','straight','right'],
          comparisons='1 RB retains authority; 2 state-triggered handback; 3 same handback on .55m level gap',
          goal='0.5s in the declared platform window, 2.1 to 3.85m after front', training_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('template', 'bank', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    prepare(a.template, a.bank.resolve(), a.output.resolve())
