"""Pair approach and downstream commands on the original successful GPU lane."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from run_playground_wide_gap_oracle import DIRECTIONS, departures, routes, run, save, write_matrix


def evaluate(screen, source, output, lane):
    original = json.loads((screen/'batch.json').read_text())
    h = original['ground_handoff']
    original_bank = np.load(original['oracle_branch_reference'])
    full_bank = np.load(json.loads(source.read_text())['ground_handoff']['reference_path'])
    # Recover the nine-route bank indices of the unchanged companion vehicles.
    route_ids = [next(i for i in (0,3,6) if np.array_equal(xy,full_bank['xy'][i]))
                 for xy in original_bank['xy']]
    plans = [(2,1.8,-1.95),(1,1.8,-1.85),(1,1.8,-1.95),(2,1.8,-1.85),
             (2,1.8,-2.0),(2,1.9,-1.95),(0,1.8,-1.95),(2,1.8,-1.94)]
    save(output/'plan.json',dict(map=original['world_map'],lane=lane,
         initial_snapshot=original['initial_snapshot_path'],frozen_skill='source',
         plans=[(DIRECTIONS[u],v,x) for u,v,x in plans],
         budget='8 approach replays; up to 16 landing branches, 10s total per trajectory. Locked ground, alpha 0. No training. All other GPU lanes retain original commands.'))
    matrix=[]
    for up,speed,offset in plans:
        ids=route_ids.copy()
        ids[lane]=3*up
        name=f'{DIRECTIONS[up]}_{speed:.1f}_x{offset:.2f}'
        parent=output/name
        ref=routes(full_bank,ids,output/(name+'_routes.npz'))
        speeds=original['oracle_initial_speed_m_s'].copy()
        speeds[lane]=speed
        variants=h['handoff_variants'].copy()
        variants[lane]=dict(variants[lane],activation_offset_m=offset)
        common=dict(original,ground_handoff=dict(h,handoff_variants=variants,reference_path=ref),
                    oracle_branch_reference=ref,planned_activation_offset_m=offset)
        rows=run(dict(common,oracle_initial_speed_m_s=speeds,snapshot_loaded_far=True),
                 parent,'source',10,output)
        if not matrix:
            comparison={}
            case=rows[lane]['case_dir']
            with np.load(parent/case/'trajectory.npz') as a, np.load(screen/'batch'/case/'trajectory.npz') as b:
                for k in ('qpos','qvel','rb_state','controller_state','applied_targets','trace'):
                    comparison[k]=bool(np.array_equal(a[k],b[k],equal_nan=True))
            save(output/'reference_replay.json',dict(case=case,equal=comparison,
                 termination=rows[lane]['termination']))
        del common['oracle_initial_speed_m_s']
        common['verify_active_jump']=False
        model=mujoco.MjModel.from_binary_path(str(parent/'terrain.mjb'))
        new=departures(rows,parent,'source',common,full_bank,model,Path(original['initial_snapshot_path']),
                       'wide_oracle_055',up,speed,output,[lane],route_ids=ids)
        for r in new:r['batch_lane']=lane
        matrix.extend(new)
        write_matrix(output,matrix)
    return matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('screen',type=Path)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--lane',type=int,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    out.mkdir()
    (out/'gpu.lock').symlink_to((a.screen.parent/'gpu.lock').resolve(strict=True))
    try:
        rows=evaluate(a.screen.resolve(),a.source,out,a.lane)
        status=f'完成 {len(rows)} 个配对单元，成功 {sum(r["full_success"] for r in rows)}'
    except Exception as e:
        status=f'异常 {e}'
        raise
    finally:
        save(out/'terminal.json',dict(status=status))
        subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章 P2 原车道配对复验','--message',status,'--receipt',str(out/'bark.json')])


if __name__=='__main__':
    main()
