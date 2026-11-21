"""Score a new shared PCRP actor on the frozen layout/group matrix."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analyze_matched_controls import LABELS, OUT, check_pair, draw_xy, plt, np, read_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation', type=Path)
    p.add_argument('governor', type=Path)
    args = p.parse_args()
    manifest = json.loads((args.evaluation/'cases.json').read_text())
    archive = json.loads((args.governor/'source_cases.json').read_text())
    assert len(manifest) == len(archive) == 40
    identity = json.loads((Path(manifest[0]['output'])/'contract.json').read_text())['identity']
    native = identity['native_checkpoint']
    training = Path(native['training_run'])
    seed = json.loads((training/'config.json').read_text())['task']['seed']
    tag = 'shared' + str(seed)
    out = OUT/'training'/tag
    figures = out/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    rows, links = [], []
    for case in manifest:
        run = Path(case['output'])
        name = run.name
        condition, _, suffix = name.partition('_v2.6')
        if 'layout' in suffix:
            condition += '_' + suffix.strip('_').split('_seed')[0]
        group = case['seed']
        c, fresh = read_case(run, LABELS[:5])
        historical, h75 = read_case(Path(archive[name]), LABELS[:6])
        gc, cap = read_case(args.governor/name, ['Cap'])
        assert c['identity']['files']['actor.npz'] == identity['files']['actor.npz']
        assert c['settings']['alphas'] == [0,.25,.5,.75,1] and not any(c['settings']['zero_middle'])
        check_pair(c, historical, different_actor=True)
        check_pair(gc, historical)
        sets = [(tag, fresh), ('H75', h75[:5]), ('Zero', h75[5:]), ('Cap', cap)]
        shared_end = min(lane[2][-1] for _, lanes in sets for lane in lanes)
        with np.load(Path(c['bundle'])/'route.npz') as path, np.load(run/'trajectory.npz') as data:
            limits = json.loads((Path(c['bundle'])/'config.json').read_text())['stage_B_baseline']
            heading = (np.interp(data['progress'], path['s'], path['heading'])-data['pose'][:,:,2]+np.pi)%(2*np.pi)-np.pi
            for lane,(row,x,t,_,_) in enumerate(fresh):
                if row['full']:
                    tail=t > c['settings']['episode_s'] - limits['hold_s'] + 1e-5
                    assert tail.sum() == round(limits['hold_s']/c['lower_dt']) and len(t)==len(heading)
                    row.update(tail_path_pass=int(np.all(np.abs(x[tail,3]) <= limits['final_y_error_m'])),
                               tail_heading_pass=int(np.all(np.abs(heading[tail,lane]) <= limits['final_path_heading_error_rad'])),
                               tail_speed_pass=int(np.all(np.abs(x[tail,2]-case['user_speed_m_s']) <= limits['final_speed_error_m_s'])))
                    assert int(row['tail_path_pass'] and row['tail_heading_pass'] and row['tail_speed_pass']) == row['hold']
        for model, lanes in sets:
            for row, trace, t, progress, reward in lanes:
                common = t <= shared_end + 1e-7
                rows.append(dict(condition=condition, physical_group=group, model=model, **row,
                                 common_end_s=shared_end, common_path_rmse_m=np.mean(trace[common,3]**2)**.5,
                                 common_speed_rmse_m_s=np.mean((trace[common,2]-case['user_speed_m_s'])**2)**.5))
        fig, axes = plt.subplots(1,2,figsize=(8.2,4.9),sharex=True,sharey=True)
        with np.load(Path(c['bundle'])/'route.npz') as route:
            for ax, lanes, title in zip(axes,[fresh+h75[5:]+cap,h75+cap],[f'Seed {seed}, update {native["completed_updates"]}','Historical H75']):
                draw_xy(ax,route,lanes)
                ax.set_title(title)
        fig.suptitle(f'{condition} | physical group {group} | 2.6 m/s | × early termination')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=8,frameon=False)
        fig.tight_layout(rect=[0,.075,1,.98])
        panel=figures/(name+'.png')
        fig.savefig(panel,dpi=125)
        plt.close(fig)
        links.append(f'| {condition} | {group} | [XY]({panel.relative_to(out)}) | [Reward and errors]({args.evaluation}/diagnostics/{name}/overview.png) |')
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    lines=[f'# Shared PCRP seed {seed}, update {native["completed_updates"]}', '',
           'Each cell gives full 10 s / final hold / tasks A, B and C. The five columns use one fixed actor. Physical groups are paired across all policies. The CSV includes all current and archived runs and computes speed and path RMSE over the common observation window of all twelve lanes per group.', '',
           '| Terrain | Model | n per preference | 0 | .25 | .5 | .75 | 1 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for condition in sorted({r['condition'] for r in rows}):
        for model in (tag,'H75'):
            group=[r for r in rows if r['condition']==condition and r['model']==model]
            values=['/'.join(str(sum(r[k] for r in group if r['method']==alpha)) for k in ('full','hold','A','B','C')) for alpha in LABELS[:5]]
            lines.append(f'| {condition} | {model} | {len(group)//5} | '+' | '.join(values)+' |')
    end=Counter(r['terminal'] for r in rows if r['model']==tag)
    full = [r for r in rows if r['model']==tag and r['full']]
    misses = [sum(1-r[k] for r in full) for k in ('tail_path_pass','tail_heading_pass','tail_speed_pass')]
    lines+=['', f'Among {len(full)} complete runs, the last-second limits were exceeded in {misses[0]} for lateral error, {misses[1]} for heading and {misses[2]} for speed.',
            '', 'Endpoint classifications for the new seed: '+', '.join(f'{k} ({v})' for k,v in sorted(end.items()))+'.',
            '', '[Per-lane physical metrics and raw trajectory paths](episodes.csv). Detailed local diagnostics include the paired Zero trajectory, five preference-specific reward reconstructions, the original reference and actual endpoints.', '',
            '| Terrain | Physical group | Both actors and controls | Detailed diagnostics |','|---|---:|---|---|',*links]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:21]))


if __name__=='__main__':
    main()
