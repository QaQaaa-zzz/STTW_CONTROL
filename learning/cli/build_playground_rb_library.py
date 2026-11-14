#!/usr/bin/env python3
"""Extract observed closed-loop RB motions; never replay them as motor actions."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from sttw_control.playground_prediction import BodyEnvelope


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
source = {'results': []}
for batch,path in enumerate(a.evaluation):
    for row in json.loads((path/'status.json').read_text())['results']:
        source['results'].append(dict(row,source_evaluation=str(path),artifact_prefix='' if len(a.evaluation)==1 else f'batch{batch:02d}_'))
schema = json.loads((a.evaluation[0] / 'controller_schema.json').read_text())
declaration = [json.loads((path/'declaration.json').read_text()) for path in a.evaluation]
model = mujoco.MjModel.from_binary_path(str(a.evaluation[0] / 'terrain.mjb'))
envelope = BodyEnvelope(model, device='cpu', dtype=torch.float64, dimensions=3)
pose_order = list(range(7)) + [int(model.jnt_qposadr[model.joint(name).id]) for name in
                             ['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']]
records = []
motions = {}
fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
for row in source['results']:
    source_case = row['case_dir'];case=row['artifact_prefix']+source_case
    evaluation=Path(row['source_evaluation'])
    with np.load(evaluation / source_case / 'trajectory.npz') as data:
        trace = data['trace']; columns = list(data['columns'])
        phase = trace[:, columns.index('phase')]
        entry = np.flatnonzero(phase == 1)
        complete = np.flatnonzero(phase == 2)
        if not len(entry):
            records.append(dict(case=case, complete=False, reason='no_rb_entry'))
            continue
        start = int(entry[0]); stop = int(complete[0]) if len(complete) else len(trace)-1
        segment = slice(start, stop+1)
        q = data['qpos'][segment].astype(np.float64)
        with torch.no_grad():
            low, high = (x.numpy() for x in envelope(torch.from_numpy(q[:,pose_order])))
        destination = a.output / (case + '.npz')
        # Observed endpoints plus the full motion retain state/history and contact evidence.
        # They are not complete simulator snapshots: RB FIFO and solver state are absent.
        arrays = {key: data[key][segment] for key in
                  ['trace', 'qpos', 'qvel', 'controller_state', 'wheel_contact_flags',
                   'world_flags', 'contact_substep']}
        for key in ['applied_targets','applied_target_valid','rb_state']:
            if key in data:arrays[key]=data[key][segment]
        arrays.update(columns=data['columns'], time_from_entry_s=trace[segment,0]-trace[start,0],
                      geometry_lower_xyz=low, geometry_upper_xyz=high,
                      controller_history_before_entry=data['controller_state'][max(0,start-64):start+1],
                      qpos_history_before_entry=data['qpos'][max(0,start-64):start+1],
                      qvel_history_before_entry=data['qvel'][max(0,start-64):start+1])
        np.savez_compressed(destination, **arrays)
        route = int(row['route']); variant = row['variant']
        observed_complete = bool(len(complete) and row['airborne'] and row['far_contact'])
        record = dict(case=case, artifact=destination.name, source=str(evaluation/source_case/'trajectory.npz'),
                      route=route, upstream=route//3, downstream=route%3,
                      root=variant['root_variant'], activation_id=variant.get('activation_id','original'), split=variant['data_split'],
                      source_group=f"upstream{route//3}_root{variant['root_variant']}_activation{variant.get('activation_id','original')}",
                      actual_targets_recorded='applied_targets' in data,rb_fifo_recorded='rb_state' in data,
                      full_runtime_anchors=str(evaluation/'snapshot_anchors.json') if (evaluation/'snapshot_anchors.json').exists() else None,
                      complete=observed_complete, termination=row['termination'], full_chain_goal=row['goal'],
                      entry_time_s=float(trace[start,0]), end_time_s=float(trace[stop,0]),
                      duration_s=float(trace[stop,0]-trace[start,0]),
                      entry_qpos=data['qpos'][start].tolist(), entry_qvel=data['qvel'][start].tolist(),
                      end_qpos=data['qpos'][stop].tolist(), end_qvel=data['qvel'][stop].tolist(),
                      displacement_m=(q[-1,:3]-q[0,:3]).tolist(),
                      peak_abs_roll_rad=float(np.abs(trace[segment,columns.index('roll')]).max()),
                      peak_abs_pitch_rad=float(np.abs(trace[segment,columns.index('pitch')]).max()),
                      observed_world_violation=bool(data['world_flags'][segment].any()),
                      substep_body_contact=bool(data['contact_substep'][segment,:,3].any()),
                      geometry_count=low.shape[1], trajectory_samples=len(q),
                      replayable_snapshot=False)
        records.append(record)
        motions[case] = arrays
        ax = axes.flat[route]
        ax.plot(data['qpos'][:,0],data['qpos'][:,1],color='0.8',lw=.6)
        ax.plot(q[:,0],q[:,1],lw=1,label=f"root{record['root']} a{record['activation_id']} {record['split']}" + (' FAILED' if not observed_complete else ''))
        ax.scatter(q[0,0],q[0,1],s=12,color='black')
        ax.scatter(q[-1,0],q[-1,1],s=20,marker='x',color='green' if observed_complete else 'red')
        ax.plot(data['reference_xy'][:,0],data['reference_xy'][:,1],color='black',ls=':',lw=.5)
        ax.set_title(f'route{route}: upstream{route//3}, downstream{route%3}')
        ax.set(xlabel='World x (m)',ylabel='World y (m)',xlim=(7.5,13),ylim=(-1.5,1.5),aspect='equal')
        ax.grid(alpha=.2)
for ax in axes.flat:
    ax.axvspan(10.3,10.6,color='black',alpha=.1)
    ax.legend(fontsize=5,ncol=2)
fig.suptitle('Observed RB motion: complete chain in gray, RB entry to handoff in color; original references dotted')
fig.tight_layout();fig.savefig(a.output/'xy.png',dpi=160);plt.close(fig)
fig, axes = plt.subplots(3,3,figsize=(16,9),sharex=True,sharey=True,layout='constrained')
for r in records:
    if 'route' not in r:
        continue
    with np.load(r['source']) as data:
        ax=axes.flat[r['route']];q=data['qpos'];ref=data['reference_xy']
        ax.plot(ref[:,0],ref[:,1],color='black',ls=':',lw=.5)
        ax.plot(q[:,0],q[:,1],lw=.8,label=f"root{r['root']} a{r['activation_id']} {r['termination']}")
        ax.scatter(q[-1,0],q[-1,1],s=12,marker='x')
for route,ax in enumerate(axes.flat):
    ax.set(title=f'route{route}',xlabel='World x (m)',ylabel='World y (m)',xlim=(1,19),ylim=(-2,2),aspect='equal')
    ax.axvspan(10.3,10.6,color='black',alpha=.1);ax.grid(alpha=.2);ax.legend(fontsize=5,ncol=2)
fig.suptitle('All actual full-chain trajectories, original references, and real terminal states')
fig.savefig(a.output/'full_chain_xy.png',dpi=160);plt.close(fig)
groups = sorted({r['source_group'] for r in records if 'source_group' in r})
summary = dict(total=len(records),completed_rb=sum(r['complete'] for r in records),
               source_groups=len(groups),split_records=dict(Counter(r.get('split','unknown') for r in records)),
               complete_by_split=dict(Counter(r['split'] for r in records if r['complete'])),
               source=[str(path) for path in a.evaluation],source_declaration=declaration,
               controller_schema=schema, trace_timing='Rows are pre-control after events; four trace targets are prior-interval values with event resets. They are not exact current RB actions. Contacts refer to preceding5ms.', role='Observed TRAIN/DEV catalog; not certified legal domain or independent TEST',
               limitations=[f'{len(groups)} upstream/root/activation groups, three downstream siblings per group; not independent episodes',
                            'Motion templates are for prediction only; actual execution remains closed-loop frozen RB225',
                            'Per-geometry 3D supports sampled at5ms are not continuous-time clearance or exact swept-volume unions',
                            'Per-step FIFO/actual targets availability is recorded per case; full runtime anchors are separate and precede entry',
                            'No calibrated nearest-template acceptance radius; do not publish unseen starts from this catalog'])
(a.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
# Entry-only nearest-template diagnosis. DEV outcomes never choose the template.
# No rotation/time warp: the forward obstacle and skill direction remain fixed.
training = [r for r in records if r.get('split') == 'TRAIN' and r['complete']]
def entry_features(record):
    motion = motions[record['case']]
    return np.r_[motion['trace'][0,1:7], motion['qpos'][0,np.array(pose_order)[[7,10,11]]], motion['qvel'][0]]
features = np.stack([entry_features(r) for r in training])
# Empirical TRAIN scale with SI floors prevents numerical-constant channels dominating.
floor = np.r_[[.05,.05,.03,.05,.05,.05], [.05]*3, [.1]*3, [.2]*3, [.2]*5]
scale = np.maximum(features.std(0), floor)
diagnostics = []
for r in records:
    if r.get('split') != 'DEV_ROOT':
        continue
    distance = np.linalg.norm((features-entry_features(r))/scale,axis=1)
    selected = training[int(distance.argmin())]
    predicted = motions[selected['case']]; actual = motions[r['case']]
    times = actual['time_from_entry_s']
    mask = times <= predicted['time_from_entry_s'][-1]
    t = times[mask]
    def interpolate(values):
        flat = values.reshape(len(values),-1)
        return np.stack([np.interp(t,predicted['time_from_entry_s'],flat[:,i]) for i in range(flat.shape[1])],1).reshape(len(t),*values.shape[1:])
    # Translate position only to the measured entry; descriptor distance remains a feature.
    shift = actual['qpos'][0,:3]-predicted['qpos'][0,:3]
    positions = interpolate(predicted['qpos'][:,:3])+shift
    low = interpolate(predicted['geometry_lower_xyz'])+shift
    high = interpolate(predicted['geometry_upper_xyz'])+shift
    support_error = np.maximum(abs(low-actual['geometry_lower_xyz'][mask]),abs(high-actual['geometry_upper_xyz'][mask]))
    diagnostics.append(dict(case=r['case'],template=selected['case'],entry_distance=float(distance.min()),
                            actual_complete=r['complete'],overlap_s=float(t[-1]),
                            actual_duration_s=r['duration_s'],predicted_duration_s=selected['duration_s'],
                            complete_actual_horizon_covered=bool(mask.all()),
                            max_xy_error_m=float(np.linalg.norm(positions[:,:2]-actual['qpos'][mask,:2],axis=1).max()),
                            max_geometry_support_error_m=float(support_error.max()),
                            endpoint_error_m=float(np.linalg.norm(predicted['qpos'][-1,:3]+shift-actual['qpos'][-1,:3]))))
(a.output/'nearest_template_dev.json').write_text(json.dumps(dict(feature_scale=scale.tolist(),records=diagnostics,
    method='Entry-only nearest successful TRAIN; translation only; no future observations; uncalibrated diagnostic'),indent=2)+'\n')
lines=['# RB observed motion library','','![All full-chain actual/reference trajectories](full_chain_xy.png)',
       '', '![RB motion detail for all routes and roots](xy.png)','',
       f"{summary['completed_rb']}/{summary['total']} observed complete RB motions; {len(groups)} upstream/root groups.",'',
       *['- '+x for x in summary['limitations']],'',
       '|Case|Split|RB complete|Full-chain outcome|Duration s|Peak roll rad|Peak pitch rad|',
       '|---|---|---|---|---:|---:|---:|']
for r in records:
    lines.append(f"|{r['case']}|{r.get('split')}|{r['complete']}|{r.get('termination')}|{r.get('duration_s',0):.3f}|{r.get('peak_abs_roll_rad',0):.3f}|{r.get('peak_abs_pitch_rad',0):.3f}|")
lines += ['', '## Entry-only nearest TRAIN template on DEV', '',
          'Errors stop at the shorter actual/template duration; endpoint errors compare each complete endpoint without time warping. No legal-domain radius or uncertainty guarantee is inferred.', '',
          '|Case|Template|Entry distance|Full actual duration covered|Max XY m|Max 3D support m|Endpoint m|',
          '|---|---|---:|---|---:|---:|---:|']
for r in diagnostics:
    lines.append(f"|{r['case']}|{r['template']}|{r['entry_distance']:.3f}|{r['complete_actual_horizon_covered']}|{r['max_xy_error_m']:.4f}|{r['max_geometry_support_error_m']:.4f}|{r['endpoint_error_m']:.4f}|")
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['source_declaration','controller_schema']},indent=2))
