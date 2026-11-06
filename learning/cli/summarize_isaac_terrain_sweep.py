"""Summarize completed paired terrain cases from saved trajectories only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, required=True)
a = p.parse_args()
index = json.loads((a.evaluation/'index.json').read_text())
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8A6500', '#333333']
labels = ['alpha 0', 'alpha .25', 'alpha .5', 'alpha .75', 'alpha 1', 'Zero']
fig, axes = plt.subplots(len(index), 3, figsize=(15, 3.6*len(index)), squeeze=False, layout='constrained')
rows = []
for case, ax in zip(index, axes):
    run = Path(case['run'])
    contract = json.loads((run/'contract.json').read_text())
    config = json.loads((Path(contract['bundle'])/'config.json').read_text())
    speed_floor = config['residual_interface']['speed_reference_bounds_m_s'][0]
    metrics = json.loads((run/'condition_metrics.json').read_text())
    data = np.load(run/'trajectory.npz')
    route = np.load(Path(contract['bundle'])/'route.npz')
    terrain = json.loads((Path(next(iter(contract['settings']['terrains'].values())))/'identity.json').read_text())
    start = terrain.get('start_s_m', terrain.get('slope_start_s_m'))
    end = terrain.get('end_s_m', terrain.get('slope_end_s_m'))
    ax[0].plot(*route['xy'][route['s']<=route['goal']+3].T, ':', color='gray', label='Original reference')
    for lane, (label, color) in enumerate(zip(labels, colors)):
        trace = data['lower_trace'][:, :, lane].reshape(-1, 11)
        ids = np.flatnonzero(trace[:, 6]); trace = trace[ids]
        progress = data['progress'][ids, lane]
        time = ids*.005+trace[:, 7]*.0002
        style = '--' if lane==5 else '-'
        ax[0].plot(*trace[:, :2].T, color=color, ls=style, label=label)
        ax[0].scatter(*trace[-1, :2], color=color, marker='x')
        ax[1].plot(time, trace[:, 2], color=color, ls=style)
        ax[2].plot(time, data['measurement'][ids, lane, 0], color=color, ls=style)
        row = dict(terrain=case['terrain'], label=label, **metrics[lane],
                   speed_reference_floor_m_s=speed_floor,
                   reference_floor_fraction=float(np.mean(np.isclose(trace[:, 8], speed_floor, atol=1e-5))))
        if start is not None and end is not None:
            on = (progress>=start)&(progress<=end)
            entered = np.flatnonzero(progress>=start)
            crossed = np.flatnonzero(progress>=end)
            row.update(terrain_start_s_m=start, terrain_end_s_m=end,
                       entry_speed_m_s=float(trace[entered[0], 2]) if len(entered) else None,
                       terrain_exit_reached=bool(len(crossed)),
                       min_terrain_speed_m_s=float(trace[on, 2].min()) if on.any() else None,
                       terrain_reference_floor_fraction=float(np.mean(np.isclose(trace[on, 8], speed_floor, atol=1e-5))) if on.any() else None,
                       mean_terrain_speed_offset_m_s=float(data['offsets'][ids[on], lane, 0].mean()) if on.any() else None)
            # Per-lane markers preserve different entry/exit times.
            for k in (1, 2):
                for crossed_ids, marker in [(entered, '>'), (crossed, '<')]:
                    if len(crossed_ids):
                        j = crossed_ids[0]
                        value = trace[j, 2] if k==1 else data['measurement'][ids[j], lane, 0]
                        ax[k].scatter(time[j], value, color=color, marker=marker, s=24)
        rows.append(row)
    ax[0].axis('equal'); ax[0].set(xlabel='World x (m)', ylabel='World y (m)')
    ax[1].axhline(case['user_speed_m_s'], color='gray', ls=':', label='Original requested speed')
    ax[1].axhline(speed_floor, color='gray', ls='-.', alpha=.6, label='Reference lower bound')
    ax[1].set(xlabel='Time (s)', ylabel='True forward speed (m/s)')
    ax[2].set(xlabel='Time (s)', ylabel='Signed roll (rad)')
    for axis in ax:
        axis.set_title(f"{case['terrain']} | request {case['user_speed_m_s']} m/s")
        axis.grid(alpha=.2)
axes[0, 0].legend(fontsize=7); axes[0, 1].legend(fontsize=7)
fig.suptitle('Frozen original H75 + selected Local300 | paired five-alpha and Zero | DEV, one initial state\n'
             'All begin at saved 2.3 m/s state; requested speeds differ. Crosses: observed XY endpoints; >/<: terrain entry/exit by chassis projection.')
fig.savefig(a.evaluation/'feasibility_overview.png', dpi=140); plt.close(fig)
(a.evaluation/'feasibility_metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
lines = ['# Paired terrain feasibility — completed cases only', '', '![Actual XY, speed and roll](feasibility_overview.png)', '',
         'Frozen original H75 and selected Local300; DEV, one matched initial state per condition. No training. All outcomes retained.',
         'The initial saved speed is 2.3 m/s; requested lower speeds are deceleration-approach conditions, not equilibrated starts.',
         'The frozen speed-reference lower bound is recorded per row; a request below that bound cannot test an unrestricted low-speed preference difference.',
         'Failure is the existing combined termination flag, not automatically a physical fall. Goal crossing and final hold are separate.',
         'Terrain exit is chassis route projection, not a measured rear-wheel clearance. Metrics use individual observed windows.', '',
         '|Terrain|Request m/s|Alpha|Early failure|Goal crossed|Final hold|Entry speed|Min terrain speed|Path RMSE m|Speed RMSE m/s|',
         '|---|---:|---|---|---|---|---:|---:|---:|---:|']
for r in rows:
    fmt = lambda x: '—' if x is None else f'{x:.3f}'
    lines.append(f"|{r['terrain']}|{r['requested_speed_m_s']}|{r['label']}|{r['failed']}|{r['goal_crossed']}|{r['final_hold']}|{fmt(r.get('entry_speed_m_s'))}|{fmt(r.get('min_terrain_speed_m_s'))}|{r['path_rmse_m']:.3f}|{r['speed_rmse_m_s']:.3f}|")
lines.extend(['', '[Per-condition full XY and paired reward diagnostics](RESULTS.md)', '',
              'No automatic feasibility or dynamic-switch benefit claim is made from this table.'])
(a.evaluation/'FEASIBILITY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(completed_conditions=len(index), trajectories=len(rows), early_failures=sum(r['failed'] for r in rows))))
