"""Replicate the frozen approach commands on reserved bank rows and two physical layouts."""
import argparse
from pathlib import Path

import numpy as np

from run_platform_task_oracle import branches, capture, read
from run_playground_wide_gap_oracle import save


def replication(plan_path):
    plan=read(plan_path)
    out=plan_path.resolve().parent
    baseline=Path(plan['baseline'])
    cfg=read(baseline/'source_capture.json')
    for key in ('initial_snapshot_path','oracle_branch_reference','snapshot_first_locked'):
        cfg.pop(key)
    ground=cfg['ground_handoff']
    ground['handoff_variants']=[dict(ground['handoff_variants'][0],bank_row=r,root_variant=r) for r in plan['rows']]
    options=plan['approaches']
    cfg['evaluation_lanes']=[[r,'h75',i] for i in range(len(plan['rows'])) for r in range(len(options))]
    profiles=[]
    for option in options:
        with np.load(Path(option['run'])/'approaches.npz') as ref:
            arc=ref['s']
            profiles.append([ref[k][option['route']] for k in ('xy','heading','curvature')])
    route=out/'frozen_profiles.npz'
    np.savez_compressed(route,s=arc,**dict(zip(('xy','heading','curvature'),map(np.stack,zip(*profiles)))))
    ground['reference_path']=str(route)
    for name,mapfile in plan['maps'].items():
        folder=out/name
        folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            continue
        lock=folder/'gpu.lock'
        if not lock.exists():
            lock.symlink_to((baseline/'gpu.lock').resolve())
        cfg['world_map']=mapfile
        save(folder/'screen.json',cfg)
        capture(folder,folder,baseline/'plan.json')
        budget=read(folder/'plan.json')
        budget.update(approaches=options,approach_labels=[r['name'] for r in options],role=plan['role'],contracts=plan['contracts'])
        save(folder/'plan.json',budget)
        branches(folder)
        save(out/(name+'_complete.json'),read(folder/'complete.json'))
    save(out/'complete.json',{name:read(out/(name+'_complete.json')) for name in plan['maps']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan',type=Path)
    replication(p.parse_args().plan)
