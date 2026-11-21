"""Rebuild the PCRP same-stack bend and analytical-control comparisons."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sttw_control.isaac_path_reporting import static_case_metrics

LABELS = ['0', '.25', '.5', '.75', '1', 'Zero', 'Cap']
COLORS = ['#0072B2', '#CC79A7', '#009E73', '#D55E00', '#78549A', '#555555', '#AA6600']
TASKS = dict(A=(.30, .20), B=(.60, .10), C=(.40, .15))
PAPER = Path(__file__).resolve().parent
OUT = PAPER.parents[1] / 'evidence/pcrp_matched_controls'


def read_case(path, labels):
    contract = json.loads((path / 'contract.json').read_text())
    with np.load(path / 'trajectory.npz') as archive:
        data = {k: archive[k] for k in ['reward', 'lower_trace', 'progress', 'pose', 'failure']}
        flags = {k: archive[k] for k in ('domain_exit', 'roll_limit_failure', 'contact_failure', 'policy_fault', 'numerical_failure')}
    metrics = static_case_metrics(data, contract)
    trace = data['lower_trace'].reshape(-1, len(metrics), 11)
    lanes = []
    for i, label in enumerate(labels):
        active = np.flatnonzero(trace[:, i, 6])
        t = active * contract['lower_dt'] + trace[active, i, 7] * contract['physics_dt']
        x = trace[active, i]
        m = metrics[i]
        failed = m['failed']
        reason = '+'.join(k for k, v in flags.items() if v[active, i].any())
        peak = float(np.abs(x[:, 3]).max())
        row = dict(method=label, full=int(not failed), hold=int(m['final_hold']), goal=int(m['goal_crossed']),
                   time_s=t[-1], terminal=(reason or 'unresolved') if failed else 'full_window',
                   peak_path_m=peak, path_rmse_m=m['path_rmse_m'], speed_rmse_m_s=m['speed_rmse_m_s'],
                   raw=str(path))
        for task, (width, speed) in TASKS.items():
            row[task] = int(row['hold'] and peak <= width and row['speed_rmse_m_s'] <= speed)
        lanes.append((row, x, t, data['progress'][active, i], data['reward'][np.unique(active // 4), i]))
    return contract, lanes


def check_pair(c, o, *, different_actor=False):
    if different_actor:
        for file in ('route.npz', 'config.json'):
            assert c['identity']['files'][file] == o['identity']['files'][file]
    else:
        assert c['identity'] == o['identity']
    assert all(t == c['terrain'][0] for t in c['terrain'] + o['terrain'][:6])
    for k in ('physics_dt', 'policy_dt', 'lower_dt'):
        assert c[k] == o[k]
    for k in ('lower_sha256', 'randomization', 'initial_state', 'seed', 'plant_options', 'user_speed_m_s', 'terrains'):
        assert c['settings'][k] == o['settings'][k], k
    assert Path(c['settings']['asset']).read_bytes() == Path(o['settings']['asset']).read_bytes()
    for group in ('randomization_readback', 'delay_steps'):
        for k, values in c[group].items():
            for lanes in (np.asarray(values), np.asarray(o[group][k])[:6]):
                np.testing.assert_array_equal(lanes, np.broadcast_to(values[0], lanes.shape))


def draw_xy(ax, route, lanes):
    limit = max(v[3].max() for v in lanes)
    ax.plot(*route['xy'][route['s'] <= limit + .4].T, 'k:', lw=1.6, label='Reference')
    for row, x, t, progress, reward in lanes:
        name = row['method']
        color = COLORS[LABELS.index(name)]
        ax.plot(*x[:, :2].T, color=color, ls='--' if name in ('Zero', 'Cap') else '-', lw=1.1, label=name)
        if not row['full']:
            ax.plot(*x[-1, :2], marker='x', color=color)
    ax.set(xlabel='World x (m)', ylabel='World y (m)', aspect='equal')
    ax.grid(alpha=.2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bends', type=Path)
    p.add_argument('governor', type=Path)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    gallery = OUT / 'figures'
    gallery.mkdir(exist_ok=True)
    cases = json.loads((args.governor / 'source_cases.json').read_text())
    rows, index = [], []
    paper, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    jobs = [(f'{name}_{seed}', name, seed, args.bends / name / f'flat_{seed}', None)
            for name in ('s_bend_pulse', 'right_bend') for seed in range(101, 106)]
    for key, source in sorted(cases.items()):
        terrain = key.split('_v2.6')[0]
        if 'layout' in key:
            terrain += '_' + key.split('_')[2]
        jobs.append((key, terrain, int(key.split('seed')[-1]), Path(source), args.governor / key))
    for key, terrain, seed, source, cap in jobs:
        c, lanes = read_case(source, LABELS[:6])
        if cap:
            cc, cap_lane = read_case(cap, ['Cap'])
            check_pair(cc, c)
            lanes.extend(cap_lane)
        limit = min(l[2][-1] for l in lanes)
        with np.load(Path(c['bundle']) / 'route.npz') as ref:
            if cap:
                fig, (xy, vax, rax) = plt.subplots(1, 3, figsize=(12, 3.6))
                draw_xy(xy, ref, lanes)
                for row, x, t, progress, reward in lanes:
                    name = row['method']
                    color = COLORS[LABELS.index(name)]
                    vax.plot(t, x[:, 2] - 2.6, color=color, lw=1)
                    rax.plot(np.arange(1, len(reward)+1), reward, color=color, lw=1)
                vax.set(xlabel='Time (s)', ylabel='Speed error (m/s)')
                rax.set(xlabel='Policy step', ylabel='Logged reward at assigned preference')
                fig.legend(*xy.get_legend_handles_labels(), loc='lower center', ncol=8)
                fig.suptitle(f'{terrain}, physical group {seed} | H75 / Local300 | x = early termination')
                fig.tight_layout(rect=[0,.09,1,.95])
                panel = gallery / (key + '.png')
                fig.savefig(panel, dpi=120)
                plt.close(fig)
            else:
                panel = source / 'xy_all_alpha.png'
                archived = gallery / (key + '.png')
                shutil.copyfile(panel, archived)
                panel = archived
                with np.load(source / 'applied_pulse.npz') as force:
                    np.testing.assert_allclose(force['impulse_Ns'], np.broadcast_to([0,5.4 if terrain=='s_bend_pulse' else 0,0], (6,3)), atol=1e-5)
                if seed == 101:
                    ax = axes[int(terrain=='right_bend')]
                    draw_xy(ax, ref, lanes)
                    ax.set_title('Pulsed S-bend' if terrain=='s_bend_pulse' else 'Unforced right bend')
        for row, x, t, progress, reward in lanes:
            shared = t <= limit + 1e-7
            if cap is None:
                assert ((x[:,0]>-3)&(x[:,0]<51)&(x[:,1]>-42)&(x[:,1]<16)).all()
            rows.append(dict(condition=terrain, physical_group=seed, **row, common_end_s=limit,
                             common_path_rmse_m=float(np.mean(x[shared,3]**2)**.5),
                             common_speed_rmse_m_s=float(np.mean((x[shared,2]-c['settings']['user_speed_m_s'])**2)**.5)))
        index.append(f'| {terrain} | {seed} | [XY]({panel.relative_to(OUT)}) |')
    paper.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=7, frameon=False)
    paper.tight_layout(rect=[0,.1,1,1])
    paper.savefig(PAPER/'figures/matched_bends.pdf', bbox_inches='tight')
    plt.close(paper)
    with (OUT/'episodes.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    lines = ['# Matched controls on previously examined routes', '',
             'Counts include every physical group. Each entry gives full 10 s / final hold / Task A / B / C counts. Conditions share the original route, speed request and lower controller. The curvature cap uses an independent simulation batch and alpha .5 for reward scoring. Zero disables middle corrections. Each row of episodes.csv retains the actual endpoint and tracking errors on both its observed lifetime and the prefix shared across all compared methods in its group.', '',
             '| Condition | n per method | '+' | '.join(LABELS)+' |', '|---|---:|'+'---:|'*7]
    for terrain in dict.fromkeys(r['condition'] for r in rows):
        group = [r for r in rows if r['condition']==terrain]
        cells = []
        for method in LABELS:
            m = [r for r in group if r['method']==method]
            cells.append('/'.join(str(sum(r[k] for r in m)) for k in ('full','hold','A','B','C')) if m else '—')
        lines.append(f'| {terrain} | {len(set(r["physical_group"] for r in group))} | '+' | '.join(cells)+' |')
    (OUT/'README.md').write_text('\n'.join(lines+['','## Per-group actual and reference paths','','| Condition | Physical group | Figure |','|---|---:|---|']+index)+'\n')


if __name__ == '__main__':
    main()
