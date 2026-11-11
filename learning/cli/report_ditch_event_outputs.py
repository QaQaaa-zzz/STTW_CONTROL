"""Export frozen event-world-model outputs; CPU inference only, no new rollouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from sttw_control.ditch_event_model import EventPlanner

NAMES = ['Approach entry', 'Loss of support', 'Far-bank contact', 'STTW handoff', 'Goal continuation', 'Clearance (aux.)']
BLUE, ORANGE = '#0072B2', '#D55E00'


def save(fig, output, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=300, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--online', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    planner = EventPlanner(args.model)
    meta = planner.meta
    transitions = dict(np.load(args.model / 'transitions.npz'))
    collection = Path(meta['collection'])
    config = json.loads((collection / 'config.json').read_text())
    protocol = json.loads((collection / 'protocol.json').read_text())
    route_ids = np.asarray(config['ground_handoff']['reference_route_ids'])
    keep = transitions['role'] == 'DEV'
    lane = transitions['lane'][keep]
    modes = transitions['mode'][keep]
    state = transitions['state'][keep]
    observed = transitions['next_state'][keep]
    observed_time = transitions['time'][keep]
    hit = transitions['hit'][keep]
    parameters = np.asarray(meta['candidate_parameters'], dtype=np.float32)[route_ids[lane]]
    with torch.no_grad():
        inputs = torch.from_numpy(state)
        mode_tensor = torch.from_numpy(modes)
        mean, logvar, logtime, logit = planner.model(inputs, torch.from_numpy(planner.plan_features(state, parameters)), torch.from_numpy(planner.map.crop(state[:, :13])), mode_tensor)
        predicted_time = torch.expm1(logtime.clamp(0, np.log(11.)))
        predicted_state = (planner.model.physical_baseline(inputs, predicted_time, mode_tensor) + mean * planner.model.residual_std + planner.model.residual_mean).numpy()
        predicted_time = predicted_time.numpy()
        probability = torch.sigmoid(logit).numpy()
    xy_error = np.linalg.norm(predicted_state[:, :2] - observed[:, :2], axis=1)
    def motion(state_values):
        # Saved roots use world-frame xyzw orientation and world-frame velocity.
        quaternion = state_values[:, 3:7].astype(np.float64)
        quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
        x, y, z, w = quaternion.T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return np.column_stack((state_values[:, 7], roll, yaw))
    observed_motion = motion(observed)
    predicted_motion = motion(predicted_state)
    motion_error = predicted_motion - observed_motion
    motion_error[:, 1:] = np.arctan2(np.sin(motion_error[:, 1:]), np.cos(motion_error[:, 1:]))
    roots = [protocol['cases'][int(i)]['root'] for i in lane]
    with (args.output / 'dev_predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['segment_lane', 'root', 'mode', 'event', 'observed_hit', 'predicted_probability', 'observed_delay_s', 'predicted_delay_s', 'observed_x_m', 'observed_y_m', 'predicted_x_m', 'predicted_y_m', 'xy_error_m', 'observed_world_vx_m_s', 'observed_world_roll_rad', 'observed_world_yaw_rad', 'predicted_world_vx_m_s', 'predicted_world_roll_rad', 'predicted_world_yaw_rad', 'world_vx_error_m_s', 'wrapped_roll_error_rad', 'wrapped_yaw_error_rad'])
        for i in range(len(lane)):
            # Failed-event terminal states are not event locations or event times.
            writer.writerow([lane[i], roots[i], modes[i], NAMES[modes[i]], hit[i], probability[i], observed_time[i] if hit[i] else '', predicted_time[i], *(observed[i, :2] if hit[i] else ['', '']), *predicted_state[i, :2], xy_error[i] if hit[i] else '', *(observed_motion[i] if hit[i] else ['', '', '']), *predicted_motion[i], *(motion_error[i] if hit[i] else ['', '', ''])])
    metrics = {}
    for mode in range(6):
        mask = modes == mode
        positive = mask & (hit > 0)
        metrics[str(mode)] = dict(name=NAMES[mode], n=int(mask.sum()), positives=int(positive.sum()), negatives=int((mask & (hit == 0)).sum()), brier=float(np.mean((probability[mask] - hit[mask]) ** 2)), time_mae_s=float(np.mean(abs(predicted_time[positive] - observed_time[positive]))), xy_error_mean_m=float(xy_error[positive].mean()), xy_error_median_m=float(np.median(xy_error[positive])))
    motion_metrics = {}
    for mode in (2, 4):
        positive = (modes == mode) & (hit > 0)
        mae = np.abs(motion_error[positive]).mean(0)
        rmse = np.sqrt((motion_error[positive]**2).mean(0))
        motion_metrics[str(mode)] = dict(name=NAMES[mode], n=int(positive.sum()), world_vx_mae_m_s=float(mae[0]), world_vx_rmse_m_s=float(rmse[0]), world_roll_circular_mae_deg=float(np.degrees(mae[1])), world_roll_circular_rmse_deg=float(np.degrees(rmse[1])), world_yaw_circular_mae_deg=float(np.degrees(mae[2])), world_yaw_circular_rmse_deg=float(np.degrees(rmse[2])))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), layout='constrained')
    for mode, ax in enumerate(axes.flat):
        positive = (modes == mode) & (hit > 0)
        ax.scatter(observed_time[positive], predicted_time[positive], s=6, color=BLUE, alpha=.35, linewidths=0)
        upper = max(observed_time[positive].max(), predicted_time[positive].max()) * 1.04
        ax.plot([0, upper], [0, upper], '--', color='0.4', linewidth=.8)
        ax.set(xlim=(0, upper), ylim=(0, upper), xlabel='Observed delay (s)', ylabel='Predicted delay (s)', title=f'{NAMES[mode]} (n={positive.sum()})')
        ax.text(.04, .91, f'MAE {metrics[str(mode)]["time_mae_s"]:.3f} s', transform=ax.transAxes)
    fig.suptitle('Conditional event-time predictions on two held-out development roots', fontsize=10)
    save(fig, args.output, 'event_time_predictions')
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), layout='constrained')
    for mode, ax in enumerate(axes.flat):
        for label, marker, color, hit_value in [('Occurred', 'o', BLUE, 1), ('Did not occur', 'x', ORANGE, 0)]:
            mask = (modes == mode) & (hit == hit_value)
            values = np.sort(probability[mask])
            if len(values):
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post', color=color, label=f'{label}: n={len(values)}', linewidth=1.2, linestyle='-' if hit_value else '--')
                ax.plot(values[::max(1, len(values)//8)], (np.arange(1, len(values)+1)/len(values))[::max(1, len(values)//8)], marker=marker, linestyle='none', color=color, markersize=3)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Predicted event probability', ylabel='Empirical cumulative fraction', title=f'{NAMES[mode]} | Brier {metrics[str(mode)]["brier"]:.3f}')
        ax.legend(loc='upper left', frameon=False)
    fig.suptitle('Event occurrence predictions: positives and failures retained', fontsize=10)
    save(fig, args.output, 'event_probability_predictions')
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), layout='constrained')
    for mode, ax in enumerate(axes.flat):
        mask = (modes == mode) & (hit > 0)
        # Ground-truth origin subtraction displays where the event occurs relative to its causal input.
        observed_delta = observed[mask, :2] - state[mask, :2]
        predicted_delta = predicted_state[mask, :2] - state[mask, :2]
        ax.scatter(observed_delta[:, 0], predicted_delta[:, 0], color=BLUE, marker='o', s=6, alpha=.35, linewidths=0, label='Forward displacement')
        ax.scatter(observed_delta[:, 1], predicted_delta[:, 1], color=ORANGE, marker='x', s=8, alpha=.4, linewidths=.5, label='Lateral displacement')
        low = min(observed_delta.min(), predicted_delta.min())
        high = max(observed_delta.max(), predicted_delta.max())
        margin = max(.05, (high-low)*.03)
        ax.plot([low-margin, high+margin], [low-margin, high+margin], '--', color='0.4', linewidth=.8)
        ax.set(xlim=(low-margin, high+margin), ylim=(low-margin, high+margin), xlabel='Observed displacement (m)', ylabel='Predicted displacement (m)', title=f'{NAMES[mode]}\nMean XY error {metrics[str(mode)]["xy_error_mean_m"]:.2f} m')
        ax.title.set_fontsize(8)
        if mode == 0: ax.legend(frameon=False)
    fig.suptitle('Conditional event-state outputs: world-frame XY displacement', fontsize=10)
    save(fig, args.output, 'event_state_predictions')
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), layout='constrained')
    for row, mode in enumerate((2, 4)):
        mask = (modes == mode) & (hit > 0)
        for dimension, variable in enumerate(('World vx (m/s)', 'World Euler roll (deg)', 'World yaw (deg)')):
            ax = axes[row, dimension]
            actual = observed_motion[mask, dimension]
            # Place the angular prediction on the equivalent branch nearest the observation.
            forecast = actual + motion_error[mask, dimension]
            if dimension:
                actual, forecast = np.degrees(actual), np.degrees(forecast)
            ax.scatter(actual, forecast, s=7, color=BLUE, alpha=.45, linewidths=0)
            low, high = min(actual.min(), forecast.min()), max(actual.max(), forecast.max())
            margin = max(.01, (high-low)*.04)
            ax.plot([low-margin, high+margin], [low-margin, high+margin], '--', color='0.4', linewidth=.8)
            ax.set(xlim=(low-margin, high+margin), ylim=(low-margin, high+margin), xlabel='Observed '+variable, ylabel='Predicted '+variable, title=f'{NAMES[mode]} (n={mask.sum()})')
            ax.text(.04, .91, f'MAE {np.abs(forecast-actual).mean():.3f}', transform=ax.transAxes)
    fig.suptitle('Conditional event-state predictions: velocity and orientation', fontsize=10)
    save(fig, args.output, 'event_motion_predictions')
    online_protocol = json.loads((args.online / 'protocol.json').read_text())
    records = json.loads((args.online / 'evaluation/moving_planning.json').read_text())['records']
    summary = json.loads((args.online / 'evaluation/summary.json').read_text())['cases']
    with np.load(args.online / 'evaluation/trajectories.npz') as archive:
        online = {key: archive[key] for key in ('root_xyzw', 'valid', 'event_time', 'event_root_xyzw')}
    selected = sorted([r for r in records if r['method'] == 'event_model' and online_protocol['cases'][r['lane']]['root'] == 3], key=lambda r: online_protocol['cases'][r['lane']]['x'])
    exported = []
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 4.2), layout='constrained')
    for panel, (ax, record) in enumerate(zip(axes, selected)):
        i = record['lane']; candidate = record['selected']; condition = online_protocol['cases'][i]; outcome = summary[i]
        actual = online['root_xyzw'][online['valid'][:, i], i, :2]
        ax.plot(actual[:, 0], actual[:, 1], color='0.3', linewidth=1, label='Executed trajectory')
        events = []
        for event in record['prediction']['events']:
            mode = event['mode']
            if not event['active_candidates'][candidate]: continue
            item = dict(mode=mode, event=NAMES[mode], predicted_probability=event['mean_event_probability'][candidate])
            if 'mean_root' in event: item['predicted_root'] = event['mean_root'][candidate]
            if mode == 0:
                t = outcome['entry_s']
                observed_root = online['root_xyzw'][max(0, round(t/.02)-1), i].tolist() if t is not None else None
            else:
                index = {1: 0, 2: 2, 3: 3, 4: 4, 5: 1}[mode]
                t = float(online['event_time'][i, index]);t = t if np.isfinite(t) else None
                observed_root = online['event_root_xyzw'][i, index].tolist() if t is not None else None
            item.update(observed_time_s=t, observed_root=observed_root)
            events.append(item)
            if mode < 5:
                xy = item['predicted_root'][:2]
                ax.plot(*xy, marker='x', color=ORANGE, markersize=5, linestyle='none', label='Predicted event mean' if mode == (0 if candidate < 45 else 4) else None)
                ax.annotate(str(mode), xy, xytext=((-8 if mode == 2 else 3), (-13 if mode == 2 else 4)), textcoords='offset points', color=ORANGE, fontsize=7)
                if observed_root is not None:
                    ax.plot(*observed_root[:2], marker='o', markerfacecolor='none', markeredgecolor=BLUE, markersize=5, linestyle='none', label='Observed event' if mode == (0 if candidate < 45 else 4) else None)
        ax.axvspan(10.3, 10.6, color='0.85', label='Ditch x interval' if panel == 0 else None)
        ax.set(xlabel='World x (m)', ylabel='World y (m)', title=f'Query x={condition["x"]:.1f} m; {"completed" if outcome["success"] else "failed"}')
        ax.set(xlim=(1., 18.), ylim=(-.8, 3.0))
        ax.set_aspect('equal', adjustable='box')
        exported.append(dict(root=3, lane=i, query_x=condition['x'], candidate=candidate, candidate_name=online_protocol['route_names'][candidate], query_root=record['query_root'], predicted_completion=record['prediction']['predicted_completion'][candidate], predicted_total_time_s=record['prediction']['predicted_time_s'][candidate], actual_success=outcome['success'], events=events))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Online autoregressive event forecasts and actual execution (same root 3)', fontsize=10)
    save(fig, args.output, 'online_event_chain')
    logs = meta['training_logs']
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')
    for key, label, color, marker in [('train_loss', 'Training minibatch', BLUE, 'o'), ('dev_loss', 'Full development set', ORANGE, 's')]:
        ax.plot([row['update'] for row in logs], [row[key] for row in logs], color=color, marker=marker, label=label, linewidth=1.2, markersize=4)
    ax.axvline(meta['selected_update'], color='0.4', linestyle=':', linewidth=1)
    ax.set(xlabel='Optimization update', ylabel='Composite model loss', title='Frozen 600-update fitting run')
    ax.legend(frameon=False)
    save(fig, args.output, 'training_loss')
    fig, ax = plt.subplots(figsize=(7.1, 3.5), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis('off')
    boxes = [(1.25, 4.4, 'Measured state (30)\nroot 13 + history PCA 16 + road 1'), (1.25, 1.9, 'Known local terrain (128)\n8 × 8 height + support mask'), (3.8, 1.9, 'Terrain encoder\n128 → 32 → 32; SiLU'), (1.25, 3.2, 'Plan (11) + event ID (6)\nroute/speed/α/skill + distances'), (6.1, 3.3, 'Shared encoder\n79 → 128 → 64; SiLU'), (8.7, 3.3, 'Three output heads\n64 → 62 each'), (8.7, 1.8, 'Residual mean + log-variance (60)\nlog-time (1) + occurrence logit (1)'), (5.7, .55, 'Physics baseline + learned residual\nnext state, event delay, event probability')]
    for x, y, label in boxes:
        ax.text(x, y, label, ha='center', va='center', fontsize=6.5, bbox=dict(boxstyle='round,pad=.4', facecolor='#F0F4F8', edgecolor='0.5'))
    for start, end in [((2.35,1.9),(2.9,1.9)), ((2.4,4.4),(5.1,3.6)), ((2.4,3.2),(5.1,3.3)), ((4.6,1.9),(5.6,2.9)), ((7.0,3.3),(7.7,3.3)), ((8.7,2.9),(8.7,2.25)), ((8.7,1.4),(6.8,.8))]:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='0.3', linewidth=1))
    ax.text(.05, .5, 'Ground entry / airborne-contact / continuation heads\nFixed event order; no learned event-type classifier', fontsize=7)
    fig.suptitle('Implemented contact-event-conditioned world model', fontsize=10)
    save(fig, args.output, 'model_architecture')
    hashes = {name: hashlib.sha256((args.model/name).read_bytes()).hexdigest() for name in ('model.pt', 'codec.npz', 'model.json')}
    result = dict(model=str(args.model), collection=str(collection), online=str(args.online), checkpoint=meta['selected_update'], hashes=hashes, dev_roots=sorted(set(roots)), dev_transitions=len(lane), metrics=metrics, motion_metrics=motion_metrics, online_cases=exported, inference='CPU; frozen checkpoint; no training or simulation', distinction='DEV: true previous-event state supplied (teacher-forced conditional transitions). Online: stored 32-particle autoregressive forecasts at the original query; no observed future supplied. Predicted event points are means, not a dynamically valid interpolated trajectory.')
    (args.output/'outputs.json').write_text(json.dumps(result, indent=2))
    text = ['# Event-world-model outputs', '', 'Frozen-model prediction evidence for a manuscript draft; publisher-specific layout approval is pending.', '', '## Conditional DEV predictions', '', f'Checkpoint update {meta["selected_update"]}; {len(lane)} transitions from two whole held-out development roots {sorted(set(roots))}. These roots were used to select the checkpoint; they are not independent TEST. Repeated prefixes and route candidates are dependent observations. No confidence intervals or independent-sample claims are made.', '', '| Event | All / positive / negative | Time MAE (s) | Mean XY error (m) | Brier |', '|---|---:|---:|---:|---:|']
    for m in metrics.values(): text.append(f'|{m["name"]}|{m["n"]} / {m["positives"]} / {m["negatives"]}|{m["time_mae_s"]:.4f}|{m["xy_error_mean_m"]:.4f}|{m["brier"]:.4f}|')
    text += ['', 'Time and state errors include only events that occurred. Failure-terminal states are not treated as event observations; their CSV targets are blank. Occurrence probabilities include all positive and negative labels. STTW handoff has no negative DEV transition, so its discrimination is untested. Its near-perfect constant-rate baseline can outperform the network.', '', '![Event times](event_time_predictions.png)', '', '![Occurrence probabilities](event_probability_predictions.png)', '', '![Event XY state predictions](event_state_predictions.png)', '', '## Online autoregressive outputs', '', '![Online event chain](online_event_chain.png)', '', 'The three panels deliberately include both complete tasks and the failed middle query for root 3. This is a case illustration, not a representative performance estimate. Ground bypass uses only the goal-continuation event. For the jump, entry, loss of support, far-bank contact, STTW handoff and goal are recursively forecast with 32 particles. Auxiliary clearance is predicted separately and is not multiplied into completion. Mean event positions are isolated markers; they are not a predicted continuous trajectory. Missing actual events are omitted rather than assigned a terminal location. The shaded strip marks the ditch x interval, which has a legal side bridge; it is not a full-width impassable barrier.', '', 'The online JSON stores event probabilities and mean root states but not per-stage elapsed times. Consequently only the stored total predicted completion time is exported; no per-stage online timing is reconstructed or invented. The conditional DEV timing plot uses the saved transition delay, not an absolute event timestamp.', '', '## Reproduction and provenance', '', '```bash', 'PYTHONPATH=learning/src /home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/report_ditch_event_outputs.py \\', f'  --model {args.model} \\', f'  --online {args.online} \\', f'  --output {args.output}', '```', '', 'The CLI reads frozen transitions and reconstructs route parameters by the segment reference-route identity, local crop from the actual mesh, and 11-dimensional plan features. DEV event means use the implemented physics baseline plus the network residual, with time expm1(clamp(log-time, 0, log(11))). Sources and identity hashes are in `outputs.json`; all DEV prediction pairs are in `dev_predictions.csv`. No resampling, smoothing, fitting, or new physical rollouts are performed. Figures use vector PDF and 300 dpi PNG, redundant markers/line styles, and untrimmed data ranges. The empirical probability curves are descriptive ECDFs, not confidence bands. Numerical agreement with the original frozen-model metrics is checked below.', '']
    original = json.loads((args.model/'results.json').read_text())['metrics']['DEV']
    for mode, values in metrics.items():
        for ours, theirs in [('brier', 'event_brier'), ('time_mae_s', 'time_mae_s'), ('xy_error_mean_m', 'xy_error_m')]:
            if not np.isclose(values[ours], original[mode][theirs], rtol=1e-4, atol=1e-6):
                raise ValueError(f'Frozen metric disagreement: {mode} {ours}')
    text += ['', '## Architecture and fitting', '', '![Architecture](model_architecture.png)', '', '![Model loss](training_loss.png)', '', 'The six logged loss pairs are the actual recorded training minibatch and full DEV losses every 100 updates. The Gaussian NLL can be negative; this is not a reward curve. Dotted line marks checkpoint selection. The continuation head predicts the goal-reaching event after handoff or along bypass; it does not predict a separately labeled recovery duration. The auxiliary clearance output is not included in the planner completion score. The score is a heuristic product of event probabilities followed by an additional non-violation factor, although goal success already includes the road constraint; it must not be described as a calibrated joint task probability.', '']
    text += ['', '## Conditional event velocity and attitude', '', '![Event velocity and attitude](event_motion_predictions.png)', '', '| Event | Positive transitions | World vx MAE (m/s) | Roll circular MAE (deg) | Yaw circular MAE (deg) |', '|---|---:|---:|---:|---:|']
    for row in motion_metrics.values():
        text.append(f'|{row["name"]}|{row["n"]}|{row["world_vx_mae_m_s"]:.4f}|{row["world_roll_circular_mae_deg"]:.3f}|{row["world_yaw_circular_mae_deg"]:.3f}|')
    text += ['', 'Quaternion predictions and observations are normalized before xyzw-to-Euler conversion. Roll here is the saved root world Euler angle, not a body lean error; its nominal offset follows the imported asset orientation. vx is the world x velocity, not body-forward speed. Angular errors use atan2(sin(predicted-observed), cos(predicted-observed)); scatter predictions are shown on the equivalent angular branch nearest the observed angle. The CSV retains principal-value angles in radians and circular errors, with failure observations blank. These are positive-event conditional predictions on two DEV roots, not autoregressive attitude accuracy or independent TEST performance. No uncertainty or calibration claim is made for the residual covariance.', '']
    text.append('All six modes reproduce the stored DEV Brier, time MAE and XY error within numerical tolerance (rtol=1e-4, atol=1e-6).')
    (args.output/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(dev_transitions=len(lane), dev_roots=sorted(set(roots)), online_cases=len(exported), output=str(args.output))))


if __name__ == '__main__':
    main()
