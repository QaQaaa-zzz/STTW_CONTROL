"""Reinterpret all488 historical endpoints with the current original-goal rule.

Original collection, strict labels, models and costs stay immutable. Held-tail
XY remains factual position, but stale path metrics are never filled in.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics


ROLES = ['parent', 'task_only', 'physical_credit', 'parent_speed_floor', 'zero_middle']
FIELDS = ['active_tick', 'physical_tick', 'actual_xy', 'contact_failure',
          'nonfinite', 'lower_fault', 'policy_fault', 'fallback_active']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def review(root):
    root = Path(root).resolve()
    status = json.loads((root / 'status.json').read_text())
    if status['phase'] != 'completed' or not status['endpoint_authenticated']:
        raise ValueError('completed authenticated historical collection required')
    plan = json.loads((root / 'plan.json').read_text())
    protocol = plan['protocol_plan']
    if protocol['declared_actual_episodes'] != 488 or protocol['horizon_s'] != 10.:
        raise ValueError('original488 protocol required')
    source = root / 'task_analysis/task_results.json'
    rows = json.loads(source.read_text())
    index = json.loads((root / 'case_index.json').read_text())
    byid = {r['case_id']: r for r in index}
    if len(rows) != 488 or len(byid) != 488 or {r['case_id'] for r in rows} != set(byid):
        raise ValueError('complete488 one-to-one identities required')
    out = root / 'goal_arrival_relaxed_v1'
    out.mkdir(exist_ok=False)
    (out / 'status.json').write_text(json.dumps(dict(phase='running', physics_steps=0, optimizer_updates=0)))
    records, traces, paths, goals = [], {}, {}, {}
    for row in rows:
        item = byid[row['case_id']]
        file = root / item['file']
        if digest(file) != item['sha256']:
            raise ValueError('historical raw identity changed')
        reference = root / item['reference']
        if item['reference'] not in paths:
            with np.load(reference, allow_pickle=False) as z:
                path = {k: z[k] for k in z.files}
            goal = float(path['goal'])
            if not path['s'][0] <= goal <= path['s'][-1]:
                raise ValueError('original spatial goal outside path')
            paths[item['reference']] = path
            goals[item['reference']] = np.array([np.interp(goal, path['s'], path['xy'][:, k]) for k in [0, 1]])
        with np.load(file, allow_pickle=False) as z:
            raw = {k: z[k] for k in FIELDS}
        traces[row['case_id']] = (raw, item['reference'])
        for radius in [.1, .2, .3, .5]:
            result = goal_arrival_metrics(raw, goal_xy=goals[item['reference']], radius_m=radius,
                dt=protocol['lower_dt_s'], horizon_s=protocol['horizon_s'])
            # A held-tail entry is valid for spatial arrival, not restored Actor
            # observation/history, path tracking, or task reward.
            hit = round(result['first_arrival_s'] / protocol['lower_dt_s']) - 1 if result['goal_arrived'] else None
            records.append(dict(case_id=row['case_id'], panel=row['panel'], context_index=row['context_index'],
                route=row['route'], requested_speed=row['speed'], role=row['role'], alpha=row['alpha'],
                replicate=row['replicate'], force_N=row['force_N'], physical_outcome=row['outcome'],
                old_strict_task=row['task_recovered'], raw_sha256=item['sha256'],
                arrived_during_held_reference=bool(raw['fallback_active'][hit]) if hit is not None else None,
                **result))
    summaries = []
    for panel in dict.fromkeys(r['panel'] for r in rows):
        for role in ROLES:
            for disturbed in [False, True]:
                for radius in [.1, .2, .3, .5]:
                    selected = [r for r in records if r['panel'] == panel and r['role'] == role
                                and bool(r['force_N']) == disturbed and r['goal_radius_m'] == radius]
                    if not selected:
                        continue
                    summaries.append(dict(panel=panel, role=role, disturbed=disturbed, radius_m=radius,
                        actual_trajectories=len(selected), arrived=sum(r['goal_arrived'] is True for r in selected),
                        arrival_unknown=sum(r['goal_arrived'] is None for r in selected),
                        survived=sum(r['physical_outcome'] == 'survived' for r in selected),
                        contact=sum(r['physical_outcome'] == 'contact' for r in selected),
                        arrived_and_survived=sum(r['goal_arrived'] is True and r['physical_outcome'] == 'survived' for r in selected),
                        contact_after_arrival=sum(r['contact_after_arrival'] for r in selected),
                        held_tail_arrivals=sum(r['arrived_during_held_reference'] is True for r in selected),
                        old_strict_success=sum(r['old_strict_task'] is True for r in selected)))
    pairs = []
    factual = [r for r in records if r['goal_radius_m'] == .5 and r['role'] != 'zero_middle']
    cells = {}
    for row in factual:
        key = (row['panel'], row['context_index'], row['alpha'], row['replicate'])
        cells.setdefault(key, {})[row['role']] = row
    for target, base in [('physical_credit', 'task_only'), ('physical_credit', 'parent_speed_floor'),
                         ('physical_credit', 'parent'), ('task_only', 'parent')]:
        for key, cell in cells.items():
            a, b = cell[target], cell[base]
            pairs.append(dict(panel=key[0], context_index=key[1], alpha=key[2], replicate=key[3],
                target=target, baseline=base, force_N=a['force_N'],
                arrival_change=None if a['goal_arrived'] is None or b['goal_arrived'] is None else
                    int(a['goal_arrived'])-int(b['goal_arrived']),
                survival_change=int(a['physical_outcome']=='survived')-int(b['physical_outcome']=='survived')))
    (out / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    write_csv(out / 'summary.csv', summaries)
    write_csv(out / 'paired_changes.csv', pairs)
    figures = render(out, rows, traces, paths, goals)
    receipt = dict(actual_historical_trajectories=488, radius_queries=len(records), physics_steps=0,
        optimizer_updates=0, source_results_sha256=digest(source),
        source_index_sha256=digest(root/'case_index.json'), source_plan_sha256=digest(root/'plan.json'),
        script_sha256=digest(__file__), reference_sha256={k:digest(root/k) for k in paths},
        figures=figures, visual_review_complete=False,
        scope='Retrospective DEV reinterpretation, original ten-second XY goal at0.5m with0.1/0.2/0.3 sensitivity. No terminal speed/heading/hold/domain criterion; all actual488 raw SHA checked. Held physical tails may enter the original world goal and are labeled explicitly; this does not restore upper observations or demonstrate path recovery. Original strict labels immutable. Deterministic and four action-noise draws separate; unique zero-middle trajectories not copied over alpha/noise. One training pair/shared ancestor, not independent repeats or final TEST. No new method or gate selected.')
    (out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(phase='completed',actual_historical_trajectories=488,
        radius_queries=len(records),physics_steps=0,optimizer_updates=0,visual_review_complete=False),indent=2)+'\n')
    print(json.dumps([r for r in summaries if r['radius_m']==.5],ensure_ascii=False))


def render(out, rows, traces, paths, goals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    colors = dict(zip(ROLES, ['#0072B2','#E69F00','#009E73','#D55E00','#777777']))
    styles = dict(zip(ROLES, ['-', '--', '-.', ':', '-']))
    for panel in dict.fromkeys(r['panel'] for r in rows):
        for context in sorted({r['context_index'] for r in rows if r['panel']==panel}):
            selected = [r for r in rows if r['panel']==panel and r['context_index']==context]
            draws = sorted({r['replicate'] for r in selected})
            fig, axes = plt.subplots(len(draws),5,figsize=(17,3.3*len(draws)),squeeze=False,layout='constrained')
            for i, draw in enumerate(draws):
                for j, alpha in enumerate([0.,.25,.5,.75,1.]):
                    ax=axes[i,j]
                    for row in selected:
                        if row['replicate']!=draw or row['alpha'] not in [alpha,None]:
                            continue
                        raw, ref=traces[row['case_id']];active=raw['active_tick'].astype(bool)
                        xy=raw['actual_xy'][active];role=row['role']
                        ax.plot(*xy.T,color=colors[role],ls=styles[role],lw=1,label=role)
                        ax.scatter(*xy[-1],color=colors[role],marker='x' if row['outcome']=='contact' else 'o',s=18)
                    ref=traces[selected[0]['case_id']][1];path=paths[ref];goal=goals[ref]
                    keep=path['s']<=float(path['goal'])
                    ax.plot(*path['xy'][keep].T,color='black',lw=.7,alpha=.4,label='Original path')
                    ax.add_patch(plt.Circle(goal,.5,fill=False,color='black',lw=.8));ax.plot(*goal,'k*',ms=5)
                    ax.set(title=f'alpha={alpha:g}, draw={draw}',xlabel='World X (m)',ylabel='World Y (m)')
                    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=6,fontsize=8)
            row=selected[0]
            fig.suptitle(f'Historical488 / {panel} / {row["route"]}, v={row["speed"]}, F={row["force_N"]}N\n'
                'Original goal disk0.5m; true physical endpoints incl. held tail; no terminal speed/heading/hold rule')
            name=f'xy_{panel}_{context}.png';fig.savefig(out/name,dpi=110);plt.close(fig);files.append(name)
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    review(p.parse_args().run)
