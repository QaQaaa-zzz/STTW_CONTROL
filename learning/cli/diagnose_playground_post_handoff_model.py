"""Reuse full-chain recordings to diagnose nominal ARX after RB; no simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import Reference,states,rollout,robot_radius_bound,geometry_error_bound

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
model=dict(np.load(a.collection/'model/nominal_arx.npz'));schema=json.loads((a.collection/'flat_00_train/evaluation/model_state_schema.json').read_text());cfg=json.loads((a.evaluation/'resolved_config.json').read_text());bank=np.load(cfg['ground_handoff']['reference_path']);hcfg=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text());radius,jr=robot_radius_bound(mujoco.MjModel.from_binary_path(str(a.evaluation/'terrain.mjb')))
status=json.loads((a.evaluation/'status.json').read_text());windows=[];summaries=[];fig,axes=plt.subplots(3,3,figsize=(13,8),layout='constrained');saved={}
for row,ax in zip(status['results'],axes.flat):
    d=np.load(a.evaluation/row['case_dir']/'trajectory.npz');q=d['qpos'][::4];v=d['qvel'][::4];trace=d['trace'][::4];pose,xi=states(q,v,schema);phase=trace[:,-1]
    follower=dict(hcfg['follower'],lookahead_scale=row['variant']['lookahead_scale']);ref=Reference(bank,row['route'],follower)
    for horizon in [5,15,50,100,150]:
        candidates=np.arange(1,len(pose)-horizon);candidates=candidates[(phase[candidates]==2)&(phase[candidates-1]==2)&(phase[candidates+horizon]==2)][::10]
        if not len(candidates):continue
        pred,_=rollout(model,ref,pose[candidates],xi[candidates],xi[candidates-1],horizon);actual=pose[candidates[:,None]+np.arange(horizon+1)[None,:]]
        error=np.linalg.norm(pred[:,:,:2]-actual[:,:,:2],axis=-1).max(1);bound=geometry_error_bound(pred,actual,radius,jr).max(1)
        for i,t in enumerate(candidates):windows.append(dict(route=row['route'],anchor_s=float(trace[t,0]),horizon_s=horizon*.02,max_xy_m=float(error[i]),body_bound_m=float(bound[i])))
        summaries.append(dict(route=row['route'],horizon_s=horizon*.02,windows=len(candidates),p95_max_xy_m=float(np.quantile(error,.95)),max_xy_m=float(error.max()),p95_body_bound_m=float(np.quantile(bound,.95)),max_body_bound_m=float(bound.max())))
        if horizon==50:
            # Earliest post-handoff window shows the actual handoff transient.
            actual=actual[0];pred=pred[0];reference=bank['xy'][row['route']];mask=(reference[:,0]>=min(actual[:,0].min(),pred[:,0].min())-.1)&(reference[:,0]<=max(actual[:,0].max(),pred[:,0].max())+.1)
            ax.plot(*reference[mask].T,'k:',label='Fixed reference');ax.plot(*actual[:,:2].T,label='Actual');ax.plot(*pred[:,:2].T,'--',label='ARX free rollout');ax.set(title=f'Route {row["route"]}, start {trace[candidates[0],0]:.2f}s',xlabel='World x (m)',ylabel='World y (m)',aspect='equal');ax.grid(alpha=.2)
            saved[f'route{row["route"]}_actual']=actual;saved[f'route{row["route"]}_predicted']=pred
axes.flat[0].legend(fontsize=7);fig.suptitle('Post-RB phase 2: earliest complete 1 s prediction; model trained only on upstream flat phase 0')
fig.savefig(a.output/'prediction_xy.png',dpi=160);plt.close(fig);np.savez_compressed(a.output/'prediction_examples.npz',**saved)
(a.output/'windows.json').write_text(json.dumps(windows,indent=2));(a.output/'summary.json').write_text(json.dumps(summaries,indent=2))
lines=['# Post-handoff nominal model transfer diagnostic','','![Actual and predicted XY for all nine routes](prediction_xy.png)','','Reuses existing nine full-chain DEV trajectories; no simulation or policy/model refitting. Model remains flat upstream TRAIN ARX. Phase-2 lookahead multiplier 1.3 is supplied explicitly. Initial two states are observed; future state and reference inputs are predicted. This is a domain diagnostic, not training coverage or independent TEST. The original full-chain nine goals remain physical-run facts, not prediction success.','','| Route | Horizon s | Windows | P95 maximum XY m | Worst XY m | P95 body bound m |','|---|---:|---:|---:|---:|---:|']
for r in summaries:lines.append(f'| {r["route"]} | {r["horizon_s"]:.1f} | {r["windows"]} | {r["p95_max_xy_m"]:.4f} | {r["max_xy_m"]:.4f} | {r["p95_body_bound_m"]:.4f} |')
lines+=['','Windows overlap and each route has one initial root. No probabilistic safety or whole-domain calibration is inferred. The source full-chain run did not save controller histories at every step; its physical states suffice for this ARX diagnostic but not for later controller-history learning. Expanded phase-2 collection remains needed.',f'\nSource: {a.evaluation}; model: {a.collection}/model/nominal_arx.npz.']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([r for r in summaries if r['horizon_s']==1.],indent=2))
