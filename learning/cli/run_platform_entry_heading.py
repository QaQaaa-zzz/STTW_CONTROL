"""Test signed approach tangent changes from the same physical roots and frozen neutral snapshots."""
import argparse
from pathlib import Path

import numpy as np

from prepare_platform_alignment import shape_entry_heading
from run_platform_task_oracle import branches, read
from run_playground_wide_gap_oracle import run, save


def experiment(baseline, out, slope):
    old = read(baseline/'plan.json')
    terrain = read(Path(old['physical_map']))['terrain']
    first = read(baseline/(old['skills'][0]+'_capture.json'))
    entry = terrain['front_x_m']+first['planned_activation_offset_m']
    with np.load(baseline/'approaches.npz') as ref:
        routes = shape_entry_heading(dict(ref),entry,slope)
    np.savez_compressed(out/'approaches.npz',**routes)
    plan = dict(old, baseline=str(baseline), entry_x_m=entry, entry_tangent_delta=slope,
                role='paired DEV intervention on the same eight bank rows and frozen map',
                entry='signed local change in reference tangent, zero lateral change at the physical entry x',
                budget='32 approach prefixes, up to 96 physical departures and 576 contract scores, zero optimizer updates')
    save(out/'plan.json',plan)
    for skill in plan['skills']:
        cfg=read(baseline/(skill+'_capture.json'))
        cfg.update(oracle_branch_reference=str(out/'approaches.npz'),
                   ground_handoff=dict(cfg['ground_handoff'],reference_path=str(out/'approaches.npz')))
        run(cfg,out/(skill+'_capture'),skill,plan['horizon_s'],out)
    branches(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--slope',type=float,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'gpu.lock').symlink_to((a.baseline/'gpu.lock').resolve())
    experiment(a.baseline.resolve(),a.output.resolve(),a.slope)
