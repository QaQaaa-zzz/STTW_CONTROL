"""Replicate the frozen approach commands on reserved bank rows and two physical layouts."""
import argparse
from pathlib import Path

import numpy as np

from run_platform_task_oracle import branches, capture, departure_route, read
from run_playground_wide_gap_oracle import run, save


def wide_replication(plan, out):
    """Evaluate every root x approach x departure in one device batch per skill and map."""
    cfg = read(Path(plan['baseline']) / 'source_capture.json')
    for key in ('initial_snapshot_path', 'oracle_branch_reference', 'snapshot_first_locked'):
        cfg.pop(key)
    rows = plan['rows']
    bank_files = [Path(path) for path in plan['initial_banks']]
    banks = [dict(np.load(path)) for path in bank_files]
    combined = out/'initial_bank.npz'
    np.savez(combined, **{key: np.concatenate([b[key] for b in banks]) for key in ('qpos','qvel')})
    options = plan['approaches']
    profiles = [dict(np.load(Path(option['run'])/'approaches.npz')) for option in options]
    arc = profiles[0]['s']
    keys = ('xy','heading','curvature')
    base = {k: np.stack([prof[k][opt['route']] for prof,opt in zip(profiles,options)]) for k in keys}
    approaches = out/'approaches.npz'
    np.savez(approaches, s=arc, **base)
    neutral = {k: np.zeros_like(base[k]) for k in keys}
    neutral['xy'][:,:,0] = arc-5
    np.savez(out/'common_start.npz', s=arc, **neutral)
    hand = cfg['ground_handoff']
    directions = ['left','straight','right']
    hand['reference_path'] = str(out/'common_start.npz')
    hand['handoff_variants'] = [dict(hand['handoff_variants'][0],bank_row=root,root_variant=root,
                                    departure=direction) for root in rows for direction in directions]
    lanes = [[r,'h75',v] for v in range(len(hand['handoff_variants'])) for r in range(len(options))]
    cfg.update(approach_initial_bank=str(combined), evaluation_lanes=lanes, collect_model_state=False,
               world_contact_capacity=64, ccd_capacity=8*len(lanes), role=plan['role'])
    for name, mapfile in plan['maps'].items():
        folder = out/name
        folder.mkdir(exist_ok=True)
        # Reuse every geometric splice across all roots and both skills.
        goal = read(Path(mapfile))['goal']
        count = len(options)*(1+len(directions)*len(arc))
        route_bank = {k: np.empty((count,*base[k].shape[1:]),dtype=np.float32) for k in keys}
        for k in keys:
            route_bank[k][:len(options)] = base[k]
        index = len(options)
        for r in range(len(options)):
            for target in (.36,0.,-.36):
                for join in range(len(arc)):
                    ref = {k:base[k][r:r+1].copy() for k in keys}
                    ref['s'] = arc
                    departure_route(ref,0,join,target,1.6,plan['contracts']['x_sections_m'][-1],
                                    goal['x_min_m'])
                    for k in keys:
                        route_bank[k][index] = ref[k][0]
                    index += 1
        file = folder/'departure_bank.npz'
        np.savez(file,s=arc,**route_bank)
        del route_bank
        for start in range(0,len(rows),plan['roots_per_batch']):
            part = folder/f'roots_{rows[start]:04d}'
            part.mkdir(exist_ok=True)
            batch = lanes[9*start:9*(start+plan['roots_per_batch'])]
            if len(batch)<1024:
                raise ValueError('A partial batch needs at least 1024 paired worlds')
            cfg.update(world_map=mapfile,evaluation_lanes=batch,ccd_capacity=8*len(batch),
                       paired_departures=dict(route_bank=str(file),approach_file=str(approaches),
                                              direction_ids=[v%3 for _,_,v in batch]))
            for skill in ('source','rb_run1'):
                case = part/skill
                if not (case/'status.json').exists():
                    run(cfg,case,skill,10,out)
        save(folder/'complete.json',dict(lanes_per_skill=len(lanes),skills=['source','rb_run1']))
    save(out/'complete.json',dict(worlds_per_batch=plan['roots_per_batch']*9,
                                  total_rollouts=len(lanes)*2*len(plan['maps'])))


def score_wide(plan_path):
    """Score physical jumps, post-lock checkpoints and same-root reversal from saved rollouts."""
    import mujoco
    from report_platform_controls import flight
    from run_playground_task_reversal import crossings_by_section, write_rows

    plan = read(plan_path)
    out = plan_path.resolve().parent
    tasks = plan['contracts']
    scores, reversals = [], []
    for name, mapfile in plan['maps'].items():
        terrain = read(Path(mapfile))['terrain']
        edge = terrain['front_x_m']
        for part in sorted((out/name).glob('roots_*')):
            for skill in ('source','rb_run1'):
                case = part/skill
                model = mujoco.MjModel.from_binary_path(str(case/'terrain.mjb'))
                groups = {}
                for row in read(case/'status.json')['results']:
                    variant = row['variant']
                    key = variant['bank_row'], row['route']
                    path = case/row['case_dir']/'trajectory.npz'
                    with np.load(path) as data:
                        trace, q, v = (data[k] for k in ('trace','qpos','qvel'))
                        hand = row['handoff_s']
                        end = len(q) if hand is None else round(hand/.005)+2
                        state = np.column_stack((q[:end],v[:end]))
                        root = groups.setdefault(key,dict(start=state,paired=True,verified=False,cells={}))
                        root['paired'] &= np.array_equal(root['start'],state,equal_nan=True)
                        land = take = None
                        if hand is not None:
                            loads = data['contact_substep']
                            supported = data['wheel_contact_flags'][:,2] & (loads[:,-1,:2]>1.e-5).all(1)
                            take,_,land = flight(model,q[:end],loads[:end],supported[:end],trace[:end,0],
                                                 edge,edge,terrain['far_bank_height_m'],row['rb_entry_s'])
                        verified = land is not None and land<=hand+1.e-5
                        root['verified'] |= verified
                        for section,_,_,y in crossings_by_section(trace,hand if hand is not None else np.inf,
                                                                  tasks['x_sections_m']):
                            mask = root['cells'].setdefault(section,[False,False])
                            hits = []
                            for i,side in enumerate(('left','right')):
                                lo,hi = tasks[side+'_y_interval_m']
                                hit = bool(verified and row['goal'] and not row['forbidden_contact_steps']
                                           and y is not None and lo<=y<=hi)
                                mask[i] |= hit
                                hits.append(hit)
                            scores.append(dict(layout=name,skill=skill,root=key[0],approach=key[1],
                                               departure=variant['departure'],section=section,y_m=y,takeoff_s=take,
                                               loaded_s=land,handoff_s=hand,verified_jump=verified,
                                               goal=row['goal'],termination=row['termination'],
                                               left_success=hits[0],right_success=hits[1],trajectory=str(path)))
                    # Keep at most one root's trajectories in memory.
                    if variant['departure']=='right' and key[1]==len(plan['approaches'])-1:
                        for section in ('near','mid','far'):
                            masks = [tuple(prefix['cells'][section]) for prefix in groups.values()]
                            paired = all(prefix['paired'] for prefix in groups.values())
                            eligible = sum(prefix['verified'] for prefix in groups.values())
                            reversal = (True,False) in masks and (False,True) in masks
                            reversals.append(dict(layout=name,skill=skill,root=key[0],section=section,
                                                  paired_prefixes=paired,eligible_approaches=eligible,
                                                  reversal=reversal if paired and eligible>=2 else None))
                        groups.clear()
                write_rows(out/'physical_scores.csv',scores)
                write_rows(out/'paired_reversal.csv',reversals)


def replication(plan_path):
    plan=read(plan_path)
    out=plan_path.resolve().parent
    if 'initial_banks' in plan:
        return wide_replication(plan,out)
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
    p.add_argument('--score-wide',action='store_true')
    args=p.parse_args()
    if args.score_wide:
        score_wide(args.plan)
    else:
        replication(args.plan)
