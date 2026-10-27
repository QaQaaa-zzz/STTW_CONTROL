"""Independent saved-XY audit: time extension, planning, and strong-zero headroom.

Reads the completed72 panel once. Never simulates, queries an Actor, or changes
the old execution plan. Numerical identities come from the original path/raw.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sttw_control.goal_arrival_analysis import digest


def endpoint(raw, goal, seconds):
    n = round(seconds/.005)
    active = raw['active_tick'][:n].astype(bool)
    index = np.flatnonzero(active)
    if not np.array_equal(index, np.arange(len(index))) or not np.array_equal(raw['physical_tick'][:n][active], index):
        raise ValueError('Actual observed prefix required')
    xy = raw['actual_xy'][:n]
    contact = active & raw['contact_failure'][:n].astype(bool)
    fault = active & (raw['nonfinite'][:n] | raw['lower_fault'][:n] | raw['policy_fault'][:n] | ~np.isfinite(xy).all(axis=1))
    fatal = np.flatnonzero(contact | fault)
    stop = int(fatal[0]) if len(fatal) else len(index)
    distances = np.sqrt(((xy[:stop]-goal)**2).sum(axis=1))
    hits = np.flatnonzero(distances <= 1.)
    physical = 'unknown' if fault.any() else 'contact' if contact.any() else 'survived' if len(index) == n else 'unknown'
    if physical == 'unknown':
        raise ValueError('Incomplete or faulty branch must remain unknown')
    return dict(arrived=bool(len(hits)), first_arrival_s=(int(hits[0])+1)*.005 if len(hits) else None,
                physical=physical, first_contact_s=(int(np.flatnonzero(contact)[0])+1)*.005 if contact.any() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.worktree.resolve()
    source = root/'runs/goal_replanning_h75_v1/combined_analysis/results.json'
    records = json.loads(source.read_text())
    if len(records) != 72:
        raise ValueError('Complete unique72 panel required')
    original48 = root/'runs/goal_replanning_github_v1/analysis/results.json'
    old = {r['case_id']: r for r in json.loads(original48.read_text())}
    plans = {}; decisions = {}; inputs = {}
    for name in ['goal_replanning_github_v1', 'goal_replanning_h75_v1']:
        base = root/'runs'/name
        for rel in ['plan.json', 'physical/replanning_decisions.json', 'common_environment_config.json']:
            file=base/rel; inputs[str(file)] = digest(file)
        plans[name] = json.loads((base/'plan.json').read_text())
        decisions[name] = {r['case_id']: r for r in json.loads((base/'physical/replanning_decisions.json').read_text())}
    rows=[]; groups={}; ids=set()
    for r in records:
        if r['case_id'] in ids:
            raise ValueError('Duplicate actual trajectory')
        ids.add(r['case_id'])
        if r['model'] != 'H75':
            if old[r['case_id']]['raw_sha256'] != r['raw_sha256']:
                raise ValueError('The48 reused trajectories changed identity')
        name = 'goal_replanning_h75_v1' if r['model']=='H75' else 'goal_replanning_github_v1'
        plan=plans[name]; ctx=plan['contexts'][r['context']]
        if digest(ctx['path'])!=ctx['path_sha256'] or digest(r['raw_file'])!=r['raw_sha256']:
            raise ValueError('Original path or actual raw changed')
        with np.load(ctx['path'], allow_pickle=False) as z:
            goal=np.array([np.interp(float(z['goal']), z['s'], z['xy'][:,k]) for k in range(2)])
        np.testing.assert_allclose(goal,r['original_goal_xy'],atol=1e-6,rtol=0)
        with np.load(r['raw_file'],allow_pickle=False) as z: raw={k:z[k] for k in z.files}
        actual=raw['active_tick'].astype(bool)
        if not np.allclose(raw['requested_speed'][actual],r['requested_speed_m_s'],atol=1e-6,rtol=0):
            raise ValueError('Original user request changed')
        if r['model']=='zero' and np.any(raw['source_keep_channels'][actual]):
            raise ValueError('Zero neural-source channels must stay masked')
        h={str(t):endpoint(raw,goal,t) for t in [10,20]}
        for t,q in h.items():
            original=r['horizons'][t]
            if q['arrived']!=original['goal_arrived'] or q['physical']!=original['physical_outcome'] or q['first_arrival_s']!=original['first_arrival_s']:
                raise ValueError('Independent endpoint differs from prior result')
        decision=decisions[name].get(r['case_id'])
        if bool(decision)!=r['replanned']:
            raise ValueError('Actual replan decision and index disagree')
        replan_s=decision['task_tick']*.005 if decision else None
        row=dict(case_id=r['case_id'],model=r['model'],planning=r['planning'],
            context=r['context'],alpha=r['alpha'],force_N=r['force_N'],event_id=r['event_id'],
            arrived10=h['10']['arrived'],arrived20=h['20']['arrived'],
            survived10=h['10']['physical']=='survived',survived20=h['20']['physical']=='survived',
            first_arrival_s=h['20']['first_arrival_s'],first_contact_s=h['20']['first_contact_s'],
            replan_s=replan_s,late_arrival=bool(not h['10']['arrived'] and h['20']['arrived']),
            actual_ticks=int(actual.sum()), raw_file=r['raw_file'],raw_sha256=r['raw_sha256'])
        rows.append(row)
        key=(r['context'],r['alpha'],r['event_id'])
        group=groups.setdefault(key,{})
        group[(r['model'],r['planning'])]=row
    if len(groups)!=12 or any(len(g)!=6 for g in groups.values()):
        raise ValueError('12 complete matched events x six methods required')
    summary=[]; paired=[]
    for model in ['current64','H75','zero']:
        for planning in ['original','replan']:
            rr=[r for r in rows if r['model']==model and r['planning']==planning]
            summary.append(dict(model=model,planning=planning,n=len(rr),
                arrived10=sum(r['arrived10'] for r in rr),arrived20=sum(r['arrived20'] for r in rr),
                survived20=sum(r['survived20'] for r in rr),late_arrivals=sum(r['late_arrival'] for r in rr),
                actual_replans=sum(r['replan_s'] is not None for r in rr)))
        for key,g in groups.items():
            original=g[(model,'original')]; replan=g[(model,'replan')];zero=g[('zero','original')]
            paired.append(dict(model=model,context=key[0],alpha=key[1],event_id=key[2],
                planning_arrival_delta=int(replan['arrived20'])-int(original['arrived20']),
                planning_survival_delta=int(replan['survived20'])-int(original['survived20']),
                replan_vs_zero_arrival_delta=int(replan['arrived20'])-int(zero['arrived20']),
                replan_vs_zero_survival_delta=int(replan['survived20'])-int(zero['survived20'])))
    output=dict(summary=summary,pairs=paired,records=rows,
        matched_events=len(groups),unique_raws=len({r['raw_sha256'] for r in rows}),
        cached48_sha_identity_checked=len(old),actual_historical_ticks=sum(r['actual_ticks'] for r in rows),
        input_sha256={**inputs,str(source):digest(source),str(original48):digest(original48),str(Path(__file__)):digest(__file__)},
        limits='Independent endpoint arithmetic on existing72.10s and20s are nested prefixes of the SAME20s trajectories, not independent deadline interventions. Zero masks neural source but retains shared geometric feedback/support/follower/frozen lower. No empirical force threshold, no causal effect of a single state component, no independent training or test.',
        new_physics_steps=0,optimizer_updates=0)
    args.output.mkdir(exist_ok=False)
    (args.output/'review.json').write_text(json.dumps(output,indent=2)+'\n')
    with (args.output/'records.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(summary=summary,matched_events=12,actual_historical_ticks=output['actual_historical_ticks']),indent=2))


if __name__=='__main__':
    main()
