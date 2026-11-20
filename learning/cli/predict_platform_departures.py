"""Evaluate frozen ground predictions for every physically executed departure on the native step."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from run_playground_task_reversal import write_rows


def physical_raster(model_path, out):
    model = mujoco.MjModel.from_binary_path(str(model_path))
    model.geom_group[:] = 5
    for name in ('floor','step','flat_lip_start','approach_rough','platform_rough'):
        model.geom_group[model.geom(name).id] = 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    spacing = .025
    xs,ys = np.arange(-5,25.01,spacing),np.arange(-4,4.01,spacing)
    height = np.zeros((len(ys),len(xs)),dtype=np.float32)
    support = np.zeros_like(height)
    origin,down = np.array([0.,0.,2.]),np.array([0.,0.,-1.])
    group = np.array([1,0,0,0,0,0],dtype=np.uint8)
    gid = np.zeros(1,dtype=np.int32)
    for i,y in enumerate(ys):
        origin[1] = y
        for j,x in enumerate(xs):
            origin[0] = x
            dist = mujoco.mj_ray(model,data,origin,down,group,1,-1,gid)
            if dist>=0:
                height[i,j] = 2.-dist
                support[i,j] = 1
    np.savez_compressed(out,height=height,support=support,bounds=[-5.,25.,-4.,4.],spacing=spacing)


def evaluate(raw, model, calibration, out):
    plan=json.loads((raw/'plan.json').read_text())
    out.mkdir(exist_ok=True,parents=True)
    raster=out/'terrain.npz'
    physical_raster(raw/(plan['skills'][0]+'_capture')/'terrain.mjb', raster)
    with (raw/'prefixes.csv').open() as f:
        prefix={(r['skill'],r['lane']):r['prefix'] for r in csv.DictReader(f)}
    with (raw/'oracle_task_reversal.csv').open() as f:
        rows=list(csv.DictReader(f))
    summary=[]
    for skill in plan['skills']:
        cases=[dict(r,prefix=prefix[skill,r['lane']]) for r in rows if r['skill']==skill]
        if not cases:
            continue
        folder=out/skill
        folder.mkdir(exist_ok=True)
        (folder/'plan.json').write_text(json.dumps(plan,indent=2))
        matrix=folder/'oracle_task_reversal.csv'
        write_rows(matrix,cases)
        subprocess.run([sys.executable,'learning/cli/evaluate_task_reversal_predictions.py',str(matrix),
            str(model),str(calibration),str(folder),'--terrain-raster',str(raster),'--save-forecast'],check=True)
        with (folder/'offline_false_accepts.csv').open() as f:
            decisions=list(csv.DictReader(f))
        for method in ('kinematic','phase2'):
            for margin in ('bare','shared_dev'):
                cell=[r for r in decisions if (r['method'],r['margin'])==(method,margin)]
                poses=[r for r in cell if r['bin']=='near' and r['task']=='left']
                summary.append(dict(skill=skill,method=method,margin=margin,contracts=len(cell),
                    observed_pass=sum(r['actual_pass']=='True' for r in cell),
                    false_accept=sum(r['false_pass']=='True' for r in cell),
                    false_reject=sum(r['false_reject']=='True' for r in cell),physical_branches=len(poses),
                    observed_collision=sum(r['actual_collision']=='True' for r in poses),
                    missed_collision=sum(r['actual_collision']=='True' and r['predicted_collision_free']=='True' for r in poses)))
    if summary:
        write_rows(out/'comparison.csv',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','model','calibration','out'):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    evaluate(a.raw,a.model,a.calibration,a.out)
