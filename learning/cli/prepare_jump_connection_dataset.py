"""Preassign whole upstream roots before expanding matched jump candidates."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--template-manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=20261106)
a = p.parse_args()
source = json.loads(a.template_manifest.read_text())
origin = source['cases'][0]['origin_id']
templates = [c for c in source['cases'] if c['origin_id'] == origin]
rng = np.random.default_rng(a.seed)
offsets = rng.uniform([-0.08, -2.0], [0.08, 2.0], size=(120, 2))
order = rng.permutation(120)
splits = {int(i): name for name, ids in (
    ('train', order[:72]), ('validation_tuning', order[72:84]),
    ('validation_calibration', order[84:96]), ('test', order[96:])) for i in ids}
a.output.mkdir(parents=True, exist_ok=False)
roots, cases = [], []
physics_lane_steps = 0
for i, (lateral, yaw) in enumerate(offsets):
    root = f'rb_connection_seed{a.seed}_root{i:03d}'
    split = splits[i]
    roots.append(dict(origin_id=root, split=split, initial_offset=dict(lateral_m=float(lateral), yaw_deg=float(yaw))))
    for template in templates:
        config = json.loads(Path(template['config']).read_text())
        case_id = template['case_id'].replace(origin, root, 1)
        base_x=config['start_x']+config['initial_offset']['lateral_m']
        base_heading=config['start_heading_rad']-math.radians(config['initial_offset']['yaw_deg'])
        physics_lane_steps += round(config['seconds']/config['physics_dt'])
        config.update(origin_id=root, data_split=split,
            role=f'Preassigned {split} root; all descendants stay in this split; frozen policies',
            initial_offset=roots[-1]['initial_offset'], start_x=base_x-float(lateral),
            start_heading_rad=base_heading+math.radians(float(yaw)), output=str(a.output/case_id))
        path = a.output/(case_id+'.json')
        path.write_text(json.dumps(config, indent=2)+'\n')
        cases.append(dict(template, case_id=case_id, config=str(path), origin_id=root, split=split))
contract = dict(seed=a.seed, status='prepared_not_started', roots=roots,
    distribution='Independent uniform initial lateral [-.08,.08] m and heading [-2,2] deg around the existing upstream start; prescribed simulation distribution, not measured real-world traffic',
    source_snapshot='All start from the declared common source snapshot, rigidly translated/rotated; no synthetic exit states',
    inheritance='All distances, candidates, midpoints, future disturbances and skill variants inherit the root split',
    upstream_failures='Retain in full-task denominator; do not substitute roots or invent exit labels',
    test_use='Do not inspect test outcomes for design or fitting; lock methods, normalization, support and calibration before final test',
    skill_updates=0, task_groups=len(roots)*3, candidate_trajectories=len(cases),
    physics_lane_steps_max=physics_lane_steps)
(a.output/'root_split.json').write_text(json.dumps(contract, indent=2)+'\n')
for split in ('train','validation_tuning','validation_calibration','test'):
    selected=[c for c in cases if c['split']==split]
    manifest=dict(role=f'Whole-root {split}; frozen jump connection candidates',
        simulation_python=source['simulation_python'], analysis_python=source['analysis_python'],
        root_split=str(a.output/'root_split.json'), cases=selected, status='prepared_not_started')
    (a.output/(split+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k: contract[k] for k in ('seed','status','task_groups','candidate_trajectories','physics_lane_steps_max')}))
