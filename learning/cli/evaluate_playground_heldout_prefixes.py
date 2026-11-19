"""Test the two frozen approaches on predeclared new roots and a new layout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import run, save

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('discovery',type=Path)
p.add_argument('output',type=Path)
p.add_argument('gpu_lock',type=Path)
a = p.parse_args()
out = a.output.resolve()
out.mkdir()
(out/'gpu.lock').symlink_to(a.gpu_lock.resolve())
cfg = json.loads((a.discovery/'screen.json').read_text())
world = json.loads(Path(cfg['world_map']).read_text())
seeds = [19031,19032]
roots = [np.random.default_rng(s).uniform([-.05,-.03,-.015],[.05,.03,.015]).tolist() for s in seeds]
layout = dict(world, map_id='heldout_reversal_055_19033',terrain=dict(world['terrain'],seed=19033,
              forbidden_rectangles=[[13.45,13.65,-.02,.10]]))
save(out/'plan.json',dict(role='Locked transfer test, no tuning',maps=[world,layout],root_seeds=seeds,
     root_y_yaw_roll=roots,source=str(a.discovery),prefixes=[dict(bend=.98,offset=-1.91),dict(bend=1.,offset=-1.95)],
     sectors_m=[12.7,13.6,15.0],budget='Two neutral starts and two ten-second 84-lane batches. Eight unique prefix tests. All other lanes are fixed batch companions. No retry of failed prefixes.'))
h = cfg['ground_handoff']
neutral = Path(cfg['initial_snapshot_path']).parent.parent/'neutral.npz'
records = []
for m in (world,layout):
    group = out/m['map_id']
    group.mkdir()
    save(group/'map.json',m)
    variants = []
    for lane, variant in enumerate(h['handoff_variants']):
        group_id = int(lane>=42)
        y,yaw,roll = roots[group_id]
        variants.append(dict(variant,start_y_m=y,start_yaw_rad=yaw,start_roll_rad=roll,root_variant=seeds[group_id],data_split='TEST'))
    initial = {k:v for k,v in cfg.items() if k not in ('initial_snapshot_path','oracle_initial_speed_m_s','oracle_initial_alpha','oracle_branch_reference')}
    initial.update(world_map=str(group/'map.json'),snapshot_loaded_far=False,verify_active_jump=False,
                   ground_handoff=dict(h,reference_path=str(neutral),handoff_variants=variants))
    run(initial,group/'neutral','source',.005,out)
    full = dict(cfg,world_map=str(group/'map.json'),initial_snapshot_path=str(group/'neutral/source_initial_snapshot.pkl'),
                ground_handoff=dict(h,handoff_variants=variants),snapshot_loaded_far=False)
    cases = run(full,group/'screen','source',10,out)
    audits = json.loads((group/'screen/jump_validation.json').read_text())['cases']
    for lane in (12,17,54,59):
        r,j = cases[lane],audits[lane]
        assert j['case']==r['case_dir']
        land = [t for t in (j['solid_bank_landing_s'],j['both_far_after_flight_s']) if t is not None]
        hand = r['handoff_s']
        records.append(dict(map=m['map_id'],root=r['variant']['root_variant'],lane=lane,prefix='A' if lane%42==12 else 'B',
                            flight=any(e['bank_to_bank_flight'] or e['solid_bank_rear_launch'] for e in j['events']),
                            landing_s=min(land) if land else None,handoff_s=hand,qualified=bool(land and hand is not None and min(land)<=hand+1e-5),
                            termination=r['termination'],end_s=r['duration_s']))
    with (out/'heldout_prefixes.csv').open('w') as f:
        writer=csv.DictWriter(f,records[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
