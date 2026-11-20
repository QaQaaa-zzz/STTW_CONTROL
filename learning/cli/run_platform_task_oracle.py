"""Run matched left, straight and right departures on a measured raised-platform handback."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
from report_platform_controls import flight
from run_playground_task_reversal import crossings_by_section, write_rows
from run_playground_wide_gap_oracle import run, save


def read(path):
    return json.loads(path.read_text())


def capture(screen, out, contract):
    cfg = read(screen/'screen.json')
    h = cfg['ground_handoff']
    lanes = cfg['evaluation_lanes']
    cfg['evaluation_lanes'] = [[i,mode,v] for i,(_,mode,v) in enumerate(lanes)]
    with np.load(h['reference_path']) as raw:
        ref = {k: raw[k] if k == 's' else raw[k][[r for r,m,v in lanes]] for k in raw}
    np.savez_compressed(out/'approaches.npz', **ref)
    straight = {k:v.copy() for k,v in ref.items()}
    straight['xy'][:,:,0] = ref['s']-5.
    straight['xy'][:,:,1] = 0.
    straight['heading'][:] = straight['curvature'][:] = 0.
    np.savez_compressed(out/'common_start.npz', **straight)
    save(out/'plan.json', dict(physical_map=cfg['world_map'], rows=[v['bank_row'] for v in h['handoff_variants']],
         skills=['source','rb_run1'],role='known layout DEV paired physical oracle', contracts=read(contract)['contracts'],
         departures={'left':.36,'straight':0.,'right':-.36},transition_length_m=1.6,horizon_s=10))
    for skill in ('source','rb_run1'):
        neutral = out/(skill+'_neutral')
        run(dict(cfg,ground_handoff=dict(h,reference_path=str(out/'common_start.npz'))),neutral,skill,.005,out)
        run(dict(cfg, ground_handoff=dict(h,reference_path=str(out/'approaches.npz')),
                 initial_snapshot_path=str(neutral/(skill+'_initial_snapshot.pkl')),
                 oracle_branch_reference=str(out/'approaches.npz'),snapshot_first_locked=True),
            out/(skill+'_capture'),skill,10,out)
    save(out/'capture_complete.json',dict(skills=['source','rb_run1']))


def departure_route(ref, lane, join, target, length, far_gate, goal_start):
    s = ref['s']
    xy = ref['xy'][lane]
    x, nominal = xy[join:].T.copy()
    def blend(x):
        u = x.clip(0,1)
        return u**3*(10-15*u+6*u**2)
    w = blend((x-x[0])/length)*(1-blend((x-far_gate)/(goal_start-far_gate)))
    y = nominal + (target-nominal)*w
    arc = np.r_[s[join],s[join]+np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
    xy[join:,0] = np.interp(s[join:],arc,x) + np.maximum(s[join:]-arc[-1],0)
    xy[join:,1] = np.interp(s[join:],arc,y)
    heading = np.arctan2(np.gradient(xy[:,1]),np.gradient(xy[:,0]))
    ref['heading'][lane,join+1:] = heading[join+1:]
    ref['curvature'][lane,join+1:] = np.gradient(heading,s)[join+1:]


def branches(out):
    plan = read(out/'plan.json')
    approaches = plan.get('approach_labels',['left','right'])
    prefixes, scores, physical = [], [], []
    for skill in plan['skills']:
        cap = out/(skill+'_capture')
        cfg = read(out/(skill+'_capture.json'))
        hand = read(cap/'resolved_config.json')['ground_handoff']
        cases = read(cap/'status.json')['results']
        model = mujoco.MjModel.from_binary_path(str(cap/'terrain.mjb'))
        schema = read(cap/'controller_schema.json')
        progress_col = sum(int(np.prod(schema[k])) for k in list(schema)[:list(schema).index('projection_progress')])
        terrain = read(cap/'resolved_config.json')['ditch_terrain']
        front = terrain['front_x_m']
        reference = dict(np.load(out/'approaches.npz'))
        schedule = defaultdict(list)
        joins = {}
        starts = {}
        anchors = {}
        state_keys = ('qpos','qvel','rb_state','controller_state')
        for i, case in enumerate(cases):
            path = cap/case['case_dir']/'trajectory.npz'
            with np.load(path) as data:
                trace, q, load = (data[k] for k in ('trace','qpos','contact_substep'))
                states = (q,data['qvel'],data['rb_state'],data['controller_state'])
                start = np.concatenate([field[0] for field in states])
                root = case['variant']['bank_row']
                if root in starts:
                    np.testing.assert_array_equal(start,starts[root])
                starts[root] = start
                supported = data['wheel_contact_flags'][:,2] & (load[:,-1,:2]>1e-5).all(1)
                entry, h = case['rb_entry_s'], case['handoff_s']
                take, end, land = flight(model,q,load,supported,trace[:,0],front,front,terrain['far_bank_height_m'],
                                         np.inf if entry is None else entry)
                qualified = land is not None and h is not None and land <= h+1e-5
                row = dict(skill=skill,root=root,lane=i,approach=approaches[i%len(approaches)],takeoff_s=take,flight_end_s=end,
                           loaded_s=land,handoff_s=h,verified=int(qualified),termination=case['termination'],prefix=str(path))
                if qualified:
                    tick = round(h/.005)+1
                    k = np.searchsorted(np.rint(trace[:,0]/.005).astype(int),tick)
                    assert round(trace[k,0]/.005)==tick and trace[k,-1]==2
                    schedule[tick].append(i)
                    anchors[i] = np.concatenate([field[k] for field in states])
                    joins[i] = max(np.searchsorted(reference['xy'][i,:,0],q[k,0]+.05),
                                   np.searchsorted(reference['s'],states[-1][k,progress_col])+1)
                prefixes.append(row)
        write_rows(out/'prefixes.csv',prefixes)
        if not schedule:
            continue
        earliest = min(schedule)
        for direction,target in plan['departures'].items():
            ref = {key:value.copy() for key,value in reference.items()}
            for i, join in joins.items():
                departure_route(ref,i,join,target,plan['transition_length_m'],plan['contracts']['x_sections_m'][-1],
                                hand['goal']['x_min_m'])
            folder = out/(skill+'_'+direction)
            ref_path = out/(folder.name+'_routes.npz')
            np.savez_compressed(ref_path,**ref)
            suffix = dict(cfg,initial_snapshot_path=str(cap/f'{skill}_anchor_{earliest:05d}.pkl'),
                          locked_departures=dict(reference_path=str(ref_path),lanes_by_tick=schedule),
                          snapshot_first_locked=False)
            suffix.pop('oracle_branch_reference')
            result = read(folder/'status.json')['results'] if (folder/'status.json').exists() and read(folder/'status.json').get('stage')=='complete' else run(suffix,folder,skill,plan['horizon_s'],out)
            for tick, lanes in schedule.items():
                for i in lanes:
                    case = cases[i]
                    path = folder/result[i]['case_dir']/'trajectory.npz'
                    with np.load(path) as z:
                        trace = z['trace']
                        assert round(trace[0,0]/.005)==earliest
                        anchor = np.concatenate([z[key][tick-earliest] for key in state_keys])
                        np.testing.assert_array_equal(anchor,anchors[i])
                        np.testing.assert_array_equal(z['published_reference_xy'],ref['xy'][i].astype('float32'))
                        wall = bool(z['world_flags'][:,2].any())  # Forbidden contact terminates a prefix before it can qualify.
                        row = dict(skill=skill,root=case['variant']['bank_row'],approach=approaches[i%len(approaches)],
                                   lane=i,departure=direction,handoff_s=case['handoff_s'],branch_tick=tick,
                                   goal=result[i]['goal'],hold_s=result[i]['goal_hold_s'],forbidden_contact=wall,
                                   termination=result[i]['termination'],end_s=float(trace[-1,0]),trajectory=str(path))
                        physical.append(row)
                        for label,x,at,y in crossings_by_section(trace,case['handoff_s'],plan['contracts']['x_sections_m']):
                            for side in ('left','right'):
                                lo,hi = plan['contracts'][side+'_y_interval_m']
                                gate = y is not None and lo<=y<=hi
                                scores.append(dict(**row,bin=label,section_x_m=x,task=side,cross_s=at,cross_y_m=y,
                                    measured_T_s=None if at is None else at-case['handoff_s'],checkpoint=gate,
                                    success=bool(gate and not wall and result[i]['goal'])))
            write_rows(out/'physical_suffixes.csv',physical)
            write_rows(out/'oracle_task_reversal.csv',scores)
            save(out/'progress.json',dict(last=folder.name,scored_suffixes=len(physical),contracts=len(scores)))
    summary = []
    for skill in plan['skills']:
        for root in plan['rows']:
            for label in ('near','mid','far'):
                cell = [r for r in scores if (r['skill'],r['root'],r['bin'])==(skill,root,label)]
                masks = {}
                for approach in approaches:
                    subset = [r for r in cell if r['approach']==approach]
                    masks[approach] = ''.join(str(int(any(r['success'] for r in subset if r['task']==side))) for side in ('left','right')) if subset else 'NA'
                eligible = sum(value!='NA' for value in masks.values())
                reversal = {'10','01'} <= set(masks.values())
                summary.append(dict(skill=skill,root=root,bin=label,**{k+'_mask':v for k,v in masks.items()},
                                    reversal=int(reversal) if eligible>=2 else None))
    write_rows(out/'paired_reversal.csv',summary)
    save(out/'complete.json',dict(prefixes=len(prefixes),verified=sum(r['verified'] for r in prefixes),
                                  physical_suffixes=len(physical),scored_contracts=len(scores)))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--capture-screen',type=Path)
    p.add_argument('--contract',type=Path)
    args = p.parse_args()
    if args.capture_screen:
        capture(args.capture_screen,args.output,args.contract)
    else:
        branches(args.output)
