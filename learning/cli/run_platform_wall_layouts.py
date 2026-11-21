"""Frozen 15 ms geometry transfer on three wall layouts and 30 held-out initial states."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from prepare_blind_platform import project, sha
from report_blind_platform import summarize
from report_platform_wall_layouts import report
from run_blind_platform import execute
from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import save


def freeze(parent, out):
    original = parent / 'test_freeze_manifest.json'
    plan = read(original)
    roots = list(range(1041,1071))
    with np.load(plan['bank']) as bank, np.load(Path(plan['wide'])/'initial_bank.npz') as dev:
        states = np.column_stack((bank['qpos'], bank['qvel']))
        seen = {row.tobytes() for row in states[plan['roots']]}
        seen.update(row.tobytes() for row in np.column_stack((dev['qpos'],dev['qvel'])))
        new = {row.tobytes() for row in states[roots]}
        if seen & new or len(new) != len(roots):
            raise ValueError('Repeated development or previous test state')
    source_map = Path(plan['maps']['test_a']['path'])
    world = read(source_map)
    xmin,xmax,ymin,ymax = world['terrain']['forbidden_rectangles'][0]
    hx,hy = (xmax-xmin)/2,(ymax-ymin)/2
    center_x = (xmin+xmax)/2
    layouts = {'config_l':[(center_x,.20)], 'config_r':[(center_x,-.25)],
               'config_double':[(12.8,.20),(14.5,-.25)]}
    out.mkdir(parents=True,exist_ok=True)
    maps = {}
    for layout,centers in layouts.items():
        world = read(source_map)
        world['map_id'] = layout
        walls = [[x-hx,x+hx,y-hy,y+hy] for x,y in centers]
        world['terrain']['forbidden_rectangles'] = walls
        world['terrain']['rough_source']['forbidden_rectangles'] = walls
        target = out/(layout+'.json')
        save(target,world)
        maps[layout] = dict(path=str(target),sha256=sha(target),walls=walls)
    pairs = list(product(roots,plan['forces_N']))
    clean = {k:plan[k] for k in ('wide','bank','bank_sha256','weights','predictor','ground_model','margin','latency_ticks',
                                  'timings_ms','clock','alpha','speed','forces_N','section_x','root_bootstrap_seed')}
    clean.update(role='T2 geometry transfer; locked unseen initial states',maps=maps,roots=roots,
                 parent_manifest=str(original),parent_sha256=sha(original),base_map=str(source_map),
                 assignment_batches=[pairs],scope='3 layouts x 2 skills x 2970 physical lanes. Four planners and nine physical oracle branches; 30 paired root clusters.')
    save(out/'test_freeze_manifest.json',clean)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    if not (args.output/'test_freeze_manifest.json').exists(): freeze(args.parent,args.output)
    if not all((args.output/layout/'rb_run1_prior.csv').exists() for layout in read(args.output/'test_freeze_manifest.json')['maps']):
        project(args.output, args.parent/'test_a')
    execute(args.output)
    summarize(args.output)
    report(args.output)
