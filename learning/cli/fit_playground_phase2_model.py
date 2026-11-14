"""Identify phase-2 nominal dynamics from real flat-buffer handoff transitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import Reference,states,design,fit,fit_local,rollout,robot_radius_bound,geometry_error_bound

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--upstream-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fit-x-min',type=float,default=11.0);p.add_argument('--fit-x-max',type=float,default=12.2);p.add_argument('--model-kind',choices=['arx','local'],default='arx');p.add_argument('--ridge',type=float,default=.001);p.add_argument('--identity-prior',action='store_true');p.add_argument('--reuse-upstream-results',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((a.evaluation/'resolved_config.json').read_text());status=json.loads((a.evaluation/'status.json').read_text());schema=json.loads((a.evaluation/'model_state_schema.json').read_text());bank=np.load(cfg['ground_handoff']['reference_path']);hcfg=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text());radius,jr=robot_radius_bound(mujoco.MjModel.from_binary_path(str(a.evaluation/'terrain.mjb')))
datasets=[];xx=[];yy=[];fit_counts=[]
for row in status['results']:
    data=np.load(a.evaluation/row['case_dir']/'trajectory.npz');q=data['qpos'][::4];v=data['qvel'][::4];trace=data['trace'][::4];pose,xi=states(q,v,schema);phase=trace[:,-1]
    ref=Reference(bank,row['route'],dict(hcfg['follower'],lookahead_scale=row['variant']['lookahead_scale']));eta=ref.features(pose,xi)
    valid=(phase==2)&(q[:,0]>=a.fit_x_min)&(q[:,0]<=a.fit_x_max);triple=valid[:-2]&valid[1:-1]&valid[2:]
    if row['variant']['data_split']=='TRAIN':xx.append(design(xi[1:-1],xi[:-2],eta[1:-1])[triple]);yy.append(xi[2:][triple])
    fit_counts.append(dict(case=row['case_dir'],split=row['variant']['data_split'],flat_transitions=int(triple.sum())))
    datasets.append(dict(row=row,pose=pose,xi=xi,trace=trace,ref=ref))
model=fit_local(np.concatenate(xx),np.concatenate(yy)) if a.model_kind=='local' else fit(np.concatenate(xx),np.concatenate(yy),ridge=a.ridge,identity_prior=a.identity_prior);np.savez(a.output/('nominal_phase2_'+a.model_kind+'.npz'),**model)
models={'phase2_flat_model':model};records=[]
if a.reuse_upstream_results:records=[r for r in json.loads(a.reuse_upstream_results.read_text()) if r['model']=='phase0_model']
else:models['phase0_model']=dict(np.load(a.upstream_model))
fig,axes=plt.subplots(3,3,figsize=(13,8),layout='constrained');predicted_examples={}
for data in datasets:
    row=data['row'];pose=data['pose'];xi=data['xi'];phase=data['trace'][:,-1];ref=data['ref']
    for name,model in models.items():
        for h in [5,15,50,100,150]:
            indices=np.arange(1,len(pose)-h);indices=indices[(phase[indices-1]==2)&(phase[indices]==2)&(phase[indices+h]==2)][::10]
            if not len(indices):continue
            pred,_=rollout(model,ref,pose[indices],xi[indices],xi[indices-1],h);actual=pose[indices[:,None]+np.arange(h+1)[None,:]];e=np.linalg.norm(pred[:,:,:2]-actual[:,:,:2],axis=-1).max(1);b=geometry_error_bound(pred,actual,radius,jr).max(1)
            for i,t in enumerate(indices):records.append(dict(model=name,case=row['case_dir'],route=row['route'],variant=row['variant']['root_variant'],split=row['variant']['data_split'],anchor_s=float(data['trace'][t,0]),horizon_s=h*.02,max_xy_m=float(e[i]),max_body_bound_m=float(b[i])))
            if name=='phase2_flat_model' and h==50 and row['variant']['data_split']=='DEV_ROOT':
                ax=axes.flat[row['route']];ax.plot(*actual[0,:,:2].T,color='#0072B2',lw=.9);ax.plot(*pred[0,:,:2].T,'--',color='#D55E00',lw=.9);predicted_examples[row['case_dir']+'_actual']=actual[0];predicted_examples[row['case_dir']+'_predicted']=pred[0]
for route,ax in enumerate(axes.flat):
    ref=bank['xy'][route];mask=(ref[:,0]>=10.8)&(ref[:,0]<=13.5);ax.plot(*ref[mask].T,'k:',lw=.8);ax.set(title=f'Route{route}: DEV roots, earliest1s',xlabel='World x(m)',ylabel='World y(m)',aspect='equal');ax.grid(alpha=.2)
fig.suptitle('Phase-2 nominal fit: actual blue; free prediction orange dashed; original reference dotted')
fig.savefig(a.output/'dev_prediction_xy.png',dpi=160);plt.close(fig);np.savez_compressed(a.output/'prediction_examples.npz',**predicted_examples)
summary=[]
for name in sorted({r['model'] for r in records}):
    for split in ['TRAIN','DEV_ROOT']:
        for route in range(9):
            for h in [.1,.3,1.,2.,3.]:
                rows=[r for r in records if r['model']==name and r['split']==split and r['route']==route and r['horizon_s']==h]
                if not rows:continue
                e=[r['max_xy_m'] for r in rows];summary.append(dict(model=name,split=split,route=route,horizon_s=h,windows=len(rows),roots=len(set(r['variant'] for r in rows)),p95_max_xy_m=float(np.quantile(e,.95)),max_xy_m=max(e),p95_body_bound_m=float(np.quantile([r['max_body_bound_m'] for r in rows],.95))))
(a.output/'windows.json').write_text(json.dumps(records,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'fit_counts.json').write_text(json.dumps(fit_counts,indent=2))
contract=dict(training_rows=sum(len(x) for x in xx),training_region_root_x_m=[a.fit_x_min,a.fit_x_max],terrain_flat_interval_m=cfg.get('phase2_model_flat_interval_m',[10.6,12.6]),grouping='TRAIN variants0-5, DEV_ROOT6-7; downstream siblings grouped by upstream+variant; same map, not independent map TEST',physics_and_policy='Frozen successful shared runner; actual handoff histories; no RB adaptation',ridge=a.ridge,identity_prior=a.identity_prior,model=a.model_kind+' fitted only from declared nominal phase2 triples; all later rollout inputs predicted',limitation='Nominal fixture and dependent windows only; no independent-map coverage, online GPU latency or planning-safety claim')
(a.output/'contract.json').write_text(json.dumps(contract,indent=2))
lines=['# Phase2 nominal flat-buffer identification','','![DEV actual and prediction XY](dev_prediction_xy.png)','',json.dumps(contract,ensure_ascii=False),'','| Model | Split | Route | Horizon s | Windows/roots | P95 max XY m | Worst XY m | P95 body bound m |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f'|{r["model"]}|{r["split"]}|{r["route"]}|{r["horizon_s"]}|{r["windows"]}/{r["roots"]}|{r["p95_max_xy_m"]:.4f}|{r["max_xy_m"]:.4f}|{r["p95_body_bound_m"]:.4f}|')
lines+=['','Overlapping windows are dependent descriptive diagnostics. All failures remain in physical reports; prediction windows stop at the real terminal state. Methods share states, integrator and geometric-bound calculation. No independent TEST, planner benefit or physical-success improvement follows from these model fits.']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(contract,indent=2));print(json.dumps([r for r in summary if r['model']=='phase2_flat_model' and r['split']=='DEV_ROOT' and r['horizon_s']==1.],indent=2))
