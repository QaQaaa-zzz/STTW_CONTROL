"""Screen takeover positions in GPU batches and replay shortlisted plans from original roots."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import mujoco
import numpy as np
from run_playground_wide_gap_oracle import DIRECTIONS, departures, routes, run, save, write_matrix


def screen(source, output):
    matrix=[]
    all_candidates=[]
    for width,skill in itertools.product(('045','055'),('rb_run1','source')):
        name=f'{width}_{skill}'
        base=source/name
        group=output/name
        group.mkdir()
        cfg=json.loads((base/'base.json').read_text())
        h=cfg['ground_handoff']
        inputs=Path(json.loads((source/'plan.json').read_text())['source'])
        stub='rb' if skill=='rb_run1' else 'source'
        pair='generic' if width=='045' else 'native'
        seed_cfg=json.loads((inputs/f'{stub}_{width}_box_{pair}.json').read_text())
        bank=dict(np.load(seed_cfg['ground_handoff']['reference_path']))
        variants=h['handoff_variants']
        offsets=(-2.,-1.95,-1.85,-1.75,-1.65,-1.55,-1.45)
        plans=list(itertools.product(range(2),range(3),(1.8,1.9),offsets))
        neutral=routes(bank,[0]*len(plans),group/'neutral.npz')
        lanes=[[i,'h75',i] for i in range(len(plans))]
        expanded=[dict(variants[lane],activation_offset_m=offset) for lane,up,speed,offset in plans]
        batch=dict(cfg,evaluation_lanes=lanes,ground_handoff=dict(h,handoff_variants=expanded,reference_path=neutral))
        root=group/'neutral'
        run(batch,root,skill,.005,output)
        routed=routes(bank,[3*up for lane,up,speed,offset in plans],group/'approaches.npz')
        trial=group/'batch'
        scan_cfg=dict(batch,verify_active_jump=True,oracle_initial_speed_m_s=[p[2] for p in plans],
            oracle_branch_reference=routed,initial_snapshot_path=str(root/f'{skill}_initial_snapshot.pkl'),
            ground_handoff=dict(batch['ground_handoff'],reference_path=routed))
        rows=run(scan_cfg,trial,skill,10,output)
        audit=json.loads((trial/'jump_validation.json').read_text())['cases']
        shortlist=set()
        for lane in range(2):
            flights=[]
            for i,(root_id,up,speed,offset) in enumerate(plans):
                if root_id!=lane:continue
                r,j=rows[i],audit[i]
                assert r['case_dir']==j['case']
                real=j['flight'] or any(e['solid_bank_rear_launch'] for e in j['events'])
                loaded=any(v is not None for v in (j['both_far_after_flight_s'],j['solid_bank_landing_s']))
                takeback=loaded and r['handoff_s'] is not None
                all_candidates.append(dict(map=name,root=r['variant']['root_variant'],lane=i,
                    approach=DIRECTIONS[up],speed_m_s=speed,activation_offset_m=offset,
                    strict_flight=j['flight'],solid_rear=any(e['solid_bank_rear_launch'] for e in j['events']),
                    landing=loaded,handoff=takeback,goal=r['goal'],termination=r['termination'],
                    end_s=r['duration_s'],archive=str(trial/r['case_dir']/'trajectory.npz')))
                if real:
                    flights.append(((takeback and r['goal'],takeback,loaded,r['duration_s']),up,speed,offset))
            # Require one loaded landing before spending matched replay budget on this root.
            if any(f[0][2] for f in flights):
                for _,up,speed,offset in sorted(flights,reverse=True)[:2]:
                    shortlist.add((up,speed,offset,lane))
        save(output/'screen_cases.json',all_candidates)
        for up,speed,offset in sorted({cand[:3] for cand in shortlist}):
            lane_ids=[lane for u,v,t,lane in sorted(shortlist) if (u,v,t)==(up,speed,offset)]
            label=f'{DIRECTIONS[up]}_{speed:.1f}_x{offset:.2f}'
            parent=group/label
            ref=routes(bank,[3*up]*2,group/(label+'_routes.npz'))
            state=base/'base'/f'{skill}_initial_snapshot.pkl'
            checked=dict(cfg,planned_activation_offset_m=offset,oracle_initial_speed_m_s=speed,
                         initial_snapshot_path=str(state),oracle_branch_reference=ref,snapshot_loaded_far=True,
                         verify_active_jump=True,ground_handoff=dict(h,reference_path=ref))
            original=run(checked,parent,skill,10,output)
            model=mujoco.MjModel.from_binary_path(str(parent/'terrain.mjb'))
            # Departure configs carry only the new route; speed was published at the root.
            departure_cfg=dict(cfg,planned_activation_offset_m=offset)
            matrix.extend(departures(original,parent,skill,departure_cfg,bank,model,state,
                'wide_oracle_'+width,up,speed,output,lane_ids))
            write_matrix(output,matrix)
    return all_candidates,matrix


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    out=args.output.resolve()
    out.mkdir()
    (out/'gpu.lock').symlink_to((args.source/'gpu.lock').resolve())
    save(out/'plan.json',dict(scope='P2 known maps; offline physical trigger screen, then matched original roots.',
        maps=['045','055'],skills=['Full RB1','Source'],initial_roots=[0,3],alpha=0,
        activation_offsets_m=[-2,-1.95,-1.85,-1.75,-1.65,-1.55,-1.45],
        speeds_m_s=[1.8,1.9],approaches=DIRECTIONS,
        budget='Four 84-lane screens; at most two flight candidates per root/skill/map ranked by full result, handoff, dual load, then duration. At least one loaded landing per root to qualify. Up to sixteen two-lane matched parents and 32 conditional departures. Silent change of physics/goal excluded. Stop on runtime error or after all groups.'))
    try:
        cases,matrix=screen(args.source.resolve(),out)
        status=f'完成 {len(cases)} 个触发候选与 {len(matrix)} 个同快照矩阵单元'
    except Exception as e:
        status=f'异常 {e}'
        raise
    finally:
        save(out/'terminal.json',dict(status=status))
        subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章 P2 触发位置筛查','--message',status,'--receipt',str(out/'bark.json')])


if __name__=='__main__':
    main()
