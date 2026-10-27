"""Read-only matched comparison of cached48 and missing24 frozen-model arms."""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json, digest


def compare(root):
    root = root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    baseline = Path(plan['source'])
    cached = json.loads((root/'cached_baselines.json').read_text())
    rows = []
    for directory, count in ((baseline, 48), (root, 24)):
        declaration = json.loads((directory/'plan.json').read_text())
        assert digest(directory/'plan.json') == json.loads((directory/'seal.json').read_text())['plan_sha256']
        status = json.loads((directory/'physical/status.json').read_text())
        receipt = json.loads((directory/'analysis/receipt.json').read_text())
        assert status['phase'] == receipt['phase'] == 'completed'
        assert status['completed_cases'] == receipt['raw_traces_sha_checked'] == count
        current = json.loads((directory/'analysis/results.json').read_text())
        assert len(current) == count
        for r in current:
            assert digest(r['raw_file']) == r['raw_sha256']
            if directory == baseline:
                c = next(c for c in cached if c['case_id'] == r['case_id'])
                assert c['raw_sha256'] == r['raw_sha256']
            model = 'H75' if directory == root else 'zero' if r['role'].endswith('zero') else 'current64'
            rows.append(dict(r, model=model, planning='replan' if r['role'].startswith('replan') else 'original'))
    # Match physical conditions, not role names shared by different frozen models.
    event = lambda r: (r['context'], r['alpha'], r['event_id'])
    groups = {}
    for model in ('current64', 'H75', 'zero'):
        for planning in ('original', 'replan'):
            group = {event(r): r for r in rows if r['model'] == model and r['planning'] == planning}
            assert len(group) == 12
            groups[model+'/'+planning] = group
    reference = groups['current64/original']
    fields = ('force_N', 'pulse_onset_task_tick', 'pulse_duration_s', 'world_direction', 'requested_speed_m_s', 'original_goal_xy')
    for group in groups.values():
        assert set(group) == set(reference)
        for k, r in group.items():
            assert all(r[f] == reference[k][f] for f in fields)
    summary = []
    for name, group in groups.items():
        for horizon in ('10', '20'):
            hs = [r['horizons'][horizon] for r in group.values()]
            summary.append(dict(method=name, horizon_s=int(horizon), n=12,
                arrived=sum(h['goal_arrived'] is True for h in hs),
                survived=sum(h['physical_outcome'] == 'survived' for h in hs),
                unknown=sum(h['goal_arrived'] is None or h['physical_outcome'] == 'unknown' for h in hs)))
    comparisons = []
    pairs = [(m+'/replan', m+'/original') for m in ('current64', 'H75', 'zero')]
    pairs += [(m+'/'+p, 'zero/'+p) for m in ('current64', 'H75') for p in ('original', 'replan')]
    pairs += [('H75/'+p, 'current64/'+p) for p in ('original', 'replan')]
    for candidate, control in pairs:
        details = []
        for k, c in groups[candidate].items():
            b = groups[control][k]; ch = c['horizons']['20']; bh = b['horizons']['20']
            known = ch['goal_arrived'] is not None and bh['goal_arrived'] is not None and 'unknown' not in (ch['physical_outcome'], bh['physical_outcome'])
            both = known and ch['goal_arrived'] and bh['goal_arrived'] and ch['physical_outcome'] == bh['physical_outcome'] == 'survived'
            d = dict(key=k, known=known, candidate=c['case_id'], control=b['case_id'],
                arrival_change=int(ch['goal_arrived'])-int(bh['goal_arrived']) if known else None,
                survival_change=int(ch['physical_outcome']=='survived')-int(bh['physical_outcome']=='survived') if known else None,
                both_arrived_and_survived=bool(both))
            if both:
                for f in ('task_window_path_length_m', 'task_window_original_route_sample_distance_time_integral_m_s', 'task_window_speed_abs_error_time_integral_m', 'task_window_observed_s'):
                    d[f+'_delta'] = c[f]-b[f]
            details.append(d)
        comparisons.append(dict(candidate=candidate, control=control,
            arrival_gain=sum(d['arrival_change']==1 for d in details), arrival_loss=sum(d['arrival_change']==-1 for d in details),
            survival_gain=sum(d['survival_change']==1 for d in details), survival_loss=sum(d['survival_change']==-1 for d in details), details=details))
    out = root/'combined_analysis'; out.mkdir(exist_ok=False)
    atomic_json(out/'results.json', rows); atomic_json(out/'summary.json', summary)
    atomic_json(out/'paired.json', comparisons)
    lines = ['# Frozen models with common GitHub Dubins planning', '',
        'Cached48 plus new24; two routes, related shared ancestor, no independent training repeats. Original goal radius1m, unchanged user alpha.', '',
        '|Model/planning|Window|Arrived/12|Survived/12|Unknown|', '|---|---:|---:|---:|---:|']
    lines += [f'|{s["method"]}|{s["horizon_s"]}s|{s["arrived"]}|{s["survived"]}|{s["unknown"]}|' for s in summary]
    lines += ['', 'Paired task deltas are candidate minus control ONLY when both arrive and survive. Short failed trajectories are not credited with cheaper completion.',
        'The both-success subset differs across comparisons and the two first-arrival times can differ. These deltas are conditional, not a global causal ranking.',
        'Task ends at first entry/contact/fault/horizon. Original-route distance uses ~2cm discrete samples, not V3 utility. Changed-path reward is excluded.',
        'H75 is transferred to the common reference permissions, not its native task. Zero means zero neural correction with common feedback and frozen local300.',
        'All physical cases remain in results.json. Shared-ancestor force/alpha branches are not independent repeats; no significance or general safety claim.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    render(out, rows, plan)
    atomic_json(out/'receipt.json', dict(phase='completed', physics_steps=0, optimizer_updates=0,
        cached_raw_checked=48, new_raw_checked=24, matching_physical_events=12,
        analysis_source_sha256=digest(__file__), visual_review='pending',
        inputs={str(d/f):digest(d/f) for d in (baseline,root) for f in ('plan.json','analysis/results.json','analysis/receipt.json')}))
    print(json.dumps(summary))


def render(out, rows, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors = dict(current64='#D55E00', H75='#0072B2', zero='#555555')
    for force in (0., -32., 32.):
        fig, axes = plt.subplots(2,2,figsize=(12,10),layout='constrained')
        for col, (context, spec) in enumerate(plan['contexts'].items()):
            with np.load(spec['path']) as z: path = {k:z[k] for k in z.files}
            goal = np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,k]) for k in (0,1)])
            for row, alpha in enumerate((0.,1.)):
                ax = axes[row,col]
                ax.plot(*path['xy'][path['s']<=path['goal']].T,c='black',ls=':',lw=1,label='Original route')
                ax.add_patch(Circle(goal,1.,fill=False,ec='black',lw=.8))
                for r in rows:
                    if (r['context'],r['alpha'],r['force_N']) != (context,alpha,force): continue
                    with np.load(r['raw_file']) as z: xy=z['actual_xy'][z['active_tick'].astype(bool)]
                    n = int(round(r['task_window_observed_s']/.005));xy=xy[:n]
                    style = '--' if r['planning']=='replan' else '-'
                    ax.plot(*xy.T,c=colors[r['model']],ls=style,lw=1.2,label=r['model']+'/'+r['planning'])
                    h=r['horizons']['20']; marker='^' if h['goal_arrived'] else 'x' if h['physical_outcome']=='contact' else 's'
                    if len(xy):ax.scatter(*xy[-1],c=colors[r['model']],marker=marker,s=26)
                ax.set_title(context.split('__')[1]+f' / alpha={alpha:g}',fontsize=10)
                ax.set(xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'All six methods; F={force:+g}N / 0.1s; original goal unchanged\nFirst arrival / contact / 20s endpoint; triangle=arrival, x=contact, square=other')
        fig.savefig(out/f'task_xy_force_{force:+g}.png',dpi=150,bbox_inches='tight');plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    compare(p.parse_args().root)
