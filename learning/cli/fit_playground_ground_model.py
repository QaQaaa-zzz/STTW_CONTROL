"""Fit TRAIN-only nominal ARX and report causal multihorizon root-pose errors."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import Reference,states,design,fit,rollout,robot_radius_bound,geometry_error_bound

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.collection/'manifest.json').read_text());bank=dict(np.load(a.collection/'routes.npz'))
entries=manifest['entries'];radius,joint_radii=robot_radius_bound(mujoco.MjModel.from_binary_path(str(Path(entries[0]['output'])/'terrain.mjb')));datasets=[];train_x=[];train_y=[];coverage=[]
for entry in entries:
    root=Path(entry['output']);status=json.loads((root/'status.json').read_text());schema=json.loads((root/'model_state_schema.json').read_text())
    cfg=json.loads((root/'resolved_config.json').read_text())
    # The resolved H75 policy configuration is archived in its frozen bundle.
    bundle=Path(cfg['ground_handoff']['bundle']);hcfg=json.loads((bundle/'config.json').read_text())
    hcfg['follower'].update(cfg['ground_handoff'].get('follower_overrides',{}))
    group=[];initial={};initial_gap=0.;controller_gap=0.
    fig,axes=plt.subplots(4,2,figsize=(12,7),layout='constrained')
    for row in status['results']:
        d=np.load(root/row['case_dir']/'trajectory.npz');q=d['qpos'][::4];v=d['qvel'][::4]
        pose,xi=states(q,v,schema);ref=Reference(bank,row['route'],hcfg['follower']);eta=ref.features(pose,xi)
        data=dict(entry=entry,row=row,pose=pose,xi=xi,reference=ref,case=root/row['case_dir']);group.append(data);datasets.append(data)
        if entry['kind']=='flat' and entry['split']=='TRAIN':train_x.append(design(xi[1:-1],xi[:-2],eta[1:-1]));train_y.append(xi[2:])
        rid=row['variant']['root_id']
        if rid not in initial:initial[rid]=(d['qpos'][0],d['qvel'][0],d['controller_state'][0])
        oq,ov,oc=initial[rid];initial_gap=max(initial_gap,float(np.max(np.abs(d['qpos'][0]-oq))),float(np.max(np.abs(d['qvel'][0]-ov))));controller_gap=max(controller_gap,float(np.max(np.abs(d['controller_state'][0]-oc))))
        ax=axes.flat[row['route']];ax.plot(d['qpos'][:,0],d['qpos'][:,1],linewidth=.8,alpha=.75)
        ax.plot(d['qpos'][-1,0],d['qpos'][-1,1],'x' if row['termination']!='timeout' else '.',color='black',ms=4)
    for route,ax in enumerate(axes.flat):
        ref=bank['xy'][route];mask=(ref[:,0]>=1.4)&(ref[:,0]<=9.2);ax.plot(*ref[mask].T,'k--',lw=1,label='Fixed reference')
        ax.set(title=f'Candidate {route}: eight roots',xlabel='World x (m)',ylabel='World y (m)',aspect='equal');ax.grid(alpha=.2)
    fig.suptitle(entry['name']+' | H75 alpha=0, request 1.8 m/s | actual trajectories end at real termination')
    fig.savefig(a.output/(entry['name']+'_xy.png'),dpi=160);plt.close(fig)
    coverage.append(dict(name=entry['name'],split=entry['split'],count=len(group),terminations=dict(Counter(r['termination'] for r in status['results'])),transitions_20ms=sum(len(d['xi'])-1 for d in group),elapsed_s=status['elapsed_s'],max_same_root_initial_qv_gap=initial_gap,max_same_root_initial_controller_gap=controller_gap))
x=np.concatenate(train_x);y=np.concatenate(train_y);model=fit(x,y);np.savez(a.output/'nominal_arx.npz',**model)
# Every 0.2 s anchor with two observed states; never read future physical/control states.
horizons=[5,15,50,100,150];records=[];examples=[]
for data in datasets:
    pose,xi,ref=data['pose'],data['xi'],data['reference']
    for horizon in horizons:
        anchors=np.arange(1,len(pose)-horizon,10)
        if not len(anchors):continue
        predicted,vel=rollout(model,ref,pose[anchors],xi[anchors],xi[anchors-1],horizon)
        actual=pose[anchors[:,None]+np.arange(horizon+1)[None,:]]
        error=np.linalg.norm(predicted[:,:,:2]-actual[:,:,:2],axis=-1)
        rotation=Rotation.from_quat(predicted[:,:,3:7].reshape(-1,4)[:,[1,2,3,0]]).inv()*Rotation.from_quat(actual[:,:,3:7].reshape(-1,4)[:,[1,2,3,0]])
        angle=rotation.magnitude().reshape(len(anchors),-1)
        body_bound=geometry_error_bound(predicted,actual,radius,joint_radii)
        for j,anchor in enumerate(anchors):records.append(dict(group=data['entry']['name'],split=data['entry']['split'],kind=data['entry']['kind'],case=data['row']['case_dir'],root=data['row']['variant']['root_id'],route=data['row']['route'],anchor_s=float(anchor*.02),horizon_s=horizon*.02,max_body_error_bound_m=float(body_bound[j].max()),max_xy_error_m=float(error[j].max()),endpoint_xy_error_m=float(error[j,-1]),max_orientation_error_rad=float(angle[j].max()),max_height_error_m=float(np.max(np.abs(predicted[j,:,2]-actual[j,:,2])))))
        if horizon==50 and data['row']['case_dir'].endswith('_v0'):
            examples.append(dict(group=data['entry']['name'],route=data['row']['route'],predicted=predicted[-1],actual=actual[-1]))
(a.output/'windows.json').write_text(json.dumps(records,indent=2));(a.output/'coverage.json').write_text(json.dumps(coverage,indent=2))
summary=[]
for entry in entries:
    for horizon in horizons:
        rows=[r for r in records if r['group']==entry['name'] and r['horizon_s']==horizon*.02]
        if not rows:continue
        e=np.array([r['max_xy_error_m'] for r in rows]);summary.append(dict(group=entry['name'],split=entry['split'],horizon_s=horizon*.02,windows=len(e),roots=len(set(r['root'] for r in rows)),max_xy_m=float(e.max()),median_max_xy_m=float(np.median(e)),p95_max_body_bound_m=float(np.quantile([r['max_body_error_bound_m'] for r in rows],.95)),max_body_bound_m=max(r['max_body_error_bound_m'] for r in rows),p95_max_xy_m=float(np.quantile(e,.95)),max_orientation_rad=max(r['max_orientation_error_rad'] for r in rows),max_height_m=max(r['max_height_error_m'] for r in rows)))
(a.output/'geometry_bound.json').write_text(json.dumps(dict(root_radius_m=radius,joint_radius_m=joint_radii.tolist(),joint_order=['steer','frontwheel','rearwheel','hip','knee'],method='Conservative kinematic radius and summed rotation displacement; exact spin invariance detected for centered axisymmetric leaf wheel ellipsoids'),indent=2))
(a.output/'calibration.json').write_text(json.dumps(summary,indent=2))
np.savez_compressed(a.output/'prediction_examples.npz',**{f'{e["group"]}_r{e["route"]}_{k}':e[k] for e in examples for k in ('predicted','actual')})
fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
for entry,ax in zip(entries,axes.flat):
    rows=[r for r in summary if r['group']==entry['name']];ax.plot([r['horizon_s'] for r in rows],[r['median_max_xy_m'] for r in rows],'o-',label='Median window maximum');ax.plot([r['horizon_s'] for r in rows],[r['p95_max_xy_m'] for r in rows],'s--',label='95th percentile of window maxima');ax.set(title=entry['name'],xlabel='Prediction horizon (s)',ylabel='Max XY error within window (m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(a.output/'multihorizon_errors.png',dpi=160);plt.close(fig)
lines=['# P1 nominal closed-loop model: first bounded collection','', '## Actual/reference XY coverage','']
for entry in entries:lines.append(f'![{entry["name"]}]({entry["name"]}_xy.png)')
lines+=['','Fixed H75 and native locked plant, 1.8 m/s, alpha=0. Flat TRAIN only fits ARX(2), ridge=0.001 with training-only scaling. No policy training. Rough maps are out-of-domain model diagnostics. Flat DEV_ROOT is a disjoint-root split on the same planar map, not independent map validation.','', '| Batch | 4 s slices | Early failures | Actual 20 ms transitions |','|---|---:|---:|---:|']
for r in coverage:lines.append(f'| {r["name"]} | {r["terminations"].get("timeout",0)}/{r["count"]} | {r["count"]-r["terminations"].get("timeout",0)} | {r["transitions_20ms"]} |')
lines+=['','Four-second slices are data collection, not goal success. Failure trajectories remain through the real terminal state. Initial snapshot and all 5 ms controller states are archived. Contact normals are collected at 1 ms; row zero has no prior interval.','', '## Causal free prediction','', '![Multihorizon errors](multihorizon_errors.png)','', '| Group | Horizon s | Windows / roots | Median max XY m | P95 max XY m | Worst max XY m | P95 max body bound m |','|---|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f'| {r["group"]} | {r["horizon_s"]:.1f} | {r["windows"]}/{r["roots"]} | {r["median_max_xy_m"]:.4f} | {r["p95_max_xy_m"]:.4f} | {r["max_xy_m"]:.4f} | {r["p95_max_body_bound_m"]:.4f} |')
lines+=['','Each anchor uses two observed states; all later state, reference projection, curvature and preview features are recomputed from prediction. Quaternion integration uses predicted body angular velocity. No future true controller state enters the predictor. Overlapping windows are not independent trials; percentiles are descriptive and not probability guarantees.','', f'Full-body diagnostic uses a conservative point-displacement upper bound with kinematic radius {radius:.4f} m: translation plus root rotation and all shape-changing hinge rotations. Exact spin invariance of centered axisymmetric leaf wheel ellipsoids is recognized; per-joint radii are in geometry_bound.json. The maximum over the prediction window is stored per window and summarized in calibration.json. This is not exact surface distance, obstacle clearance, or a probabilistic safety guarantee. P1 is not accepted until errors on the actual planning horizon fit the scene clearance and a paired E1 pilot is executed. Long-horizon drift and early failures must guide additional data or a smaller supported planning horizon.','', f'TRAIN regression rows: {len(x)}. Full per-window records: windows.json; frozen collection manifest: {a.collection}/manifest.json. No independent TEST or E1/E2/E3 benefit has been established.']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(training_rows=len(x),episodes=len(datasets),output=str(a.output)),indent=2))
