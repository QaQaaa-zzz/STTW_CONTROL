"""Run a finite, offline approach/departure matrix using real branch snapshots."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np

from sttw_control.playground_actual_clearance import obstacle_aabb_bounds

DIRECTIONS = ('straight', 'left', 'right')


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def run(config, case, skill, seconds, output):
    save(case.parent/(case.name+'.json'), config)
    cmd = [sys.executable,'learning/cli/evaluate_playground_handoff.py','--config',str(case.parent/(case.name+'.json')),
           '--output',str(case),'--skills',skill,'--modes','h75','--routes','0','1',
           '--physics-substeps','5','--seconds',str(seconds)]
    with open(output/'gpu.lock','a') as lock, (case.parent/(case.name+'.log')).open('w') as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start=time.time()
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        save(output/'live.json',dict(stage=str(case.relative_to(output)),pid=proc.pid,start=start))
        code=proc.wait()
    save(case.parent/(case.name+'.exit.json'),dict(code=code,seconds=time.time()-start))
    if code:
        raise RuntimeError(f'{case.name} exited {code}, see {log.name}')
    return json.loads((case/'status.json').read_text())['results']


def routes(bank, ids, path):
    np.savez_compressed(path,s=bank['s'],**{k:bank[k][ids] for k in ('xy','heading','curvature')})
    return str(path)


def write_matrix(output, rows):
    with (output/'oracle_matrix.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def departures(originals,case,skill,cfg,bank,model,root,world_id,up,speed,output,selected_lanes):
    group=case.parent
    h=cfg['ground_handoff']
    results=[]
    audit=json.loads((case/'jump_validation.json').read_text())['cases']
    for lane,(orig,jump) in enumerate(zip(originals,audit,strict=True)):
        if lane not in selected_lanes:
            continue
        assert orig['case_dir']==jump['case']
        prefix=np.load(case/orig['case_dir']/'trajectory.npz')
        t=prefix['trace'][:,0]
        assert abs(prefix['trace'][0,18]-speed)<1e-5
        landings=[v for v in [jump['both_far_after_flight_s'],jump['solid_bank_landing_s']] if v is not None]
        landing=min(landings) if landings else None
        tick=round(landing/.005) if landing is not None else None
        for down in range(3):
            row=orig
            pose=prefix['qpos']; suffix=''
            status='failed_prefix'
            if landing is not None:
                status='continuous' if down==0 else 'branched'
            if down and landing is not None:
                anchor=case/f'{skill}_anchor_{tick:05d}.pkl'
                child=group/f'{case.name}_root{lane}_{DIRECTIONS[down]}'
                ids=[3*up]*2
                ids[lane]+=down
                childref=routes(bank,ids,group/(child.name+'_routes.npz'))
                if anchor.exists():
                    child_cfg=dict(cfg,ground_handoff=dict(h,reference_path=childref),
                                   initial_snapshot_path=str(anchor),oracle_branch_reference=childref)
                    row=run(child_cfg,child,skill,10,output)[lane]
                    suffix=str(child/row['case_dir']/'trajectory.npz')
                    with np.load(suffix) as z:
                        k=np.flatnonzero(np.rint(t/.005)==tick).item()
                        np.testing.assert_array_equal(prefix['qpos'][k],z['qpos'][0])
                        np.testing.assert_array_equal(prefix['qvel'][k],z['qvel'][0])
                        pose=np.concatenate((pose[:k],z['qpos']))
                else:
                    status='missing_loaded_snapshot'
            _,clearance=obstacle_aabb_bounds(model,pose)
            success=(status in ('continuous','branched') and row['goal'] and
                     row['handoff_s'] is not None and landing<=row['handoff_s']+1e-5)
            results.append(dict(map=world_id,skill=skill,root=orig['variant']['root_variant'],
                initial_snapshot=str(root),activation_offset_m=cfg['planned_activation_offset_m'],approach=DIRECTIONS[up],speed_m_s=speed,departure=DIRECTIONS[down],
                strict_flight=jump['flight'],rear_launch_solid=any(e['solid_bank_rear_launch'] for e in jump['events']),
                landing_s=landing,handoff_s=row['handoff_s'],goal=row['goal'],full_success=success,
                status=status,termination=row['termination'],end_s=row['duration_s'],
                obstacle_distance_lower_bound_m=clearance,prefix=str(case/orig['case_dir']/'trajectory.npz'),suffix=suffix))
        prefix.close()
    return results


def evaluate(source, output):
    results=[]
    for width,contact in [('045','generic'),('055','native')]:
        for skill,stub in [('rb_run1','rb'),('source','source')]:
            cfg=json.loads((source/f'{stub}_{width}_box_{contact}.json').read_text())
            group=output/f'{width}_{skill}'
            group.mkdir()
            world=json.loads(Path(cfg['world_map']).read_text())
            world['map_id']=f'wide_oracle_{width}'
            world['terrain']['forbidden_rectangles']=[[13.3,13.5,-.06,.06]]
            world['terrain']['obstacle_height_m']=.5
            save(group/'world.json',world)
            h=cfg['ground_handoff']
            bank=dict(np.load(h['reference_path']))
            h['handoff_variants']=h['handoff_variants'][:2]
            h['homing_min_forward_m_s']=.5
            h['reference_path']=routes(bank,[0,0],group/'base_routes.npz')
            cfg.update(role='P2 known-map offline physical oracle',canonical_constraint_rows=True,
                       ordered_solver_reductions=True,world_map=str(group/'world.json'),
                       evaluation_lanes=[[0,'h75',0],[1,'h75',1]],verify_active_jump=False)
            base=group/'base'
            run(cfg,base,skill,.005,output)
            model=mujoco.MjModel.from_binary_path(str(base/'terrain.mjb'))
            root=base/f'{skill}_initial_snapshot.pkl'
            for up in range(3):
                for speed in (1.8,1.9):
                    case=group/f'{DIRECTIONS[up]}_{speed:.1f}'
                    ref=routes(bank,[3*up]*2,group/(case.name+'_routes.npz'))
                    parent_cfg=dict(cfg,ground_handoff=dict(h,reference_path=ref),
                                    initial_snapshot_path=str(root),oracle_branch_reference=ref,
                                    oracle_initial_speed_m_s=speed,snapshot_loaded_far=True,verify_active_jump=True)
                    originals=run(parent_cfg,case,skill,10,output)
                    results.extend(departures(originals,case,skill,cfg,bank,model,root,world['map_id'],
                                              up,speed,output,range(len(originals))))
                    write_matrix(output,results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,help='Directory of flat-bank contact base configs')
    parser.add_argument('output',type=Path)
    parser.add_argument('--gpu-lock',required=True,type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir()
    (output/'gpu.lock').symlink_to(args.gpu_lock.resolve())
    save(output/'plan.json',dict(scope='P2 offline known DEV maps, same full root per approach and unchanged physical goal',
         maps=['0.45 m generic pair','0.55 m native pair'],obstacle=[13.3,13.5,-.06,.06],source=str(args.source),
         root_variants=[0,3],skills=['source','rb_run1'],speeds=[1.8,1.9],alpha=0.,
         budget='Four initial batches, 24 full approach batches of two lanes, up to 96 conditional two-lane departures. 144 matrix cells. All natural failures retained. Only branch when an audited strict flight or solid-bank rear launch is followed by loaded far-bank landing. End at 10s or earlier real termination. No learning.'))
    try:
        rows=evaluate(args.source.resolve(),output)
        subprocess.run([sys.executable,'learning/cli/report_playground_wide_gap_oracle.py',str(output),
                        str(output/'analysis')],check=True)
        status=f'完成 {len(rows)} 个矩阵单元，真实飞越且整程完成 {sum(r["full_success"] for r in rows)}'
    except Exception as e:
        status=f'异常 {e}'
        raise
    finally:
        save(output/'terminal.json',dict(status=status))
        subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章 P2 宽沟分叉','--message',status,'--receipt',str(output/'bark.json')])


if __name__=='__main__':
    main()
