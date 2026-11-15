"""Causal phase2 prediction on frozen DEV trajectories, including rough terrain."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.playground_history_model import load_history_model

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-root',type=Path,required=True)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',default='best.pt')
p.add_argument('--worst-examples',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter()
model=load_history_model(a.model_root,a.checkpoint)
cfg=json.loads((a.evaluation/'resolved_config.json').read_text());status=json.loads((a.evaluation/'status.json').read_text());schema=json.loads((a.evaluation/'model_state_schema.json').read_text())
if model.terrain is not None:
    from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec
    model.terrain=TerrainFeatures(**raster_from_spec(cfg['ditch_terrain']),device='cuda')
bank=np.load(cfg['ground_handoff']['reference_path']);tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
s=tensor(bank['s']);xy=tensor(bank['xy']);heading=tensor(bank['heading']);curvature=tensor(bank['curvature'])
envelope=BodyEnvelope(mujoco.MjModel.from_binary_path(str(a.evaluation/'terrain.mjb')),dimensions=3)
records=[];examples={};coverage=[]
fig,axes=plt.subplots(3,3,figsize=(14,6),layout='constrained')
with torch.no_grad():
    for row in status['results']:
        if row['variant']['data_split']=='TRAIN':continue
        with np.load(a.evaluation/row['case_dir']/'trajectory.npz') as data:
            pose,velocity=states(data['qpos'][::4],data['qvel'][::4],schema);phase=data['trace'][::4,-1];controller=data['controller_state'][::4];requests=data['trace'][::4,18]
        stage_alpha=row.get('continuation_alpha',cfg['ground_handoff']['continuation']['alpha']) if model.alpha_conditioned else None
        poses=tensor(pose);velocities=tensor(velocity);controllers=tensor(controller);speed_requests=tensor(requests);route=row.get('final_route',row['route']);windows=0
        for h in [5,15,50,100,150]:
            best_example_error=-1.
            indices=np.arange(1,len(pose)-h);bad=np.r_[0,np.cumsum(phase!=2)];indices=indices[(bad[indices+h+1]-bad[indices-1])==0][::10]
            for start in range(0,len(indices),64):
                anchor=torch.tensor(indices[start:start+64],device='cuda');n=len(anchor)
                prediction,_=model.rollout(poses[anchor],velocities[anchor],velocities[anchor-1],controllers[anchor],s,xy[route:route+1].expand(n,-1,-1),heading[route:route+1].expand(n,-1),curvature[route:route+1].expand(n,-1),h,speed_request=speed_requests[anchor],alpha_request=torch.full((n,),stage_alpha,device=poses.device) if model.alpha_conditioned else None)
                actual=poses[anchor[:,None]+torch.arange(h+1,device='cuda')]
                low,high=envelope(prediction);actual_low,actual_high=envelope(actual)
                support=torch.maximum((low-actual_low).abs(),(high-actual_high).abs())
                errors=(prediction[:,:,:2]-actual[:,:,:2]).norm(dim=-1).amax(1).cpu().numpy()
                support_xy=support[:,:,:,:2].amax((1,2,3)).cpu().numpy();support_xyz=support.amax((1,2,3)).cpu().numpy()
                for j,idx in enumerate(indices[start:start+64]):
                    records.append(dict(alpha=stage_alpha,case=row['case_dir'],route=route,root=row['variant']['root_variant'],horizon_s=h*.02,anchor_s=float(idx*.02),max_xy_m=float(errors[j]),max_support_xy_m=float(support_xy[j]),max_support_xyz_m=float(support_xyz[j])))
                windows+=n
                j=int(errors.argmax()) if a.worst_examples else 0
                if (a.worst_examples and float(errors[j])>best_example_error) or (not a.worst_examples and start==0):
                    best_example_error=float(errors[j]);key=f"{row['case_dir']}_h{h}"
                    examples[key+'_actual']=actual[j].cpu().numpy();examples[key+'_prediction']=prediction[j].cpu().numpy()
            if h==50 and len(indices):
                ax=axes.flat[route];ax.plot(pose[phase==2,0],pose[phase==2,1],color='0.75',lw=.6)
                actual_example=examples[f"{row['case_dir']}_h{h}_actual"];predicted_example=examples[f"{row['case_dir']}_h{h}_prediction"]
                ax.plot(actual_example[:,0],actual_example[:,1],color='#0072B2',label='saved actual' if row['variant']['root_variant']==6 else None)
                ax.plot(predicted_example[:,0],predicted_example[:,1],'--',color='#D55E00',label='prediction' if row['variant']['root_variant']==6 else None)
        coverage.append(dict(case=row['case_dir'],physical_outcome=row['termination'],windows=windows))
for route,ax in enumerate(axes.flat):
    ref=bank['xy'][route];mask=(ref[:,0]>=10.7)&(ref[:,0]<=18.5);ax.plot(ref[mask,0],ref[mask,1],'k:',lw=.7)
    ax.set(title=f'Route{route}: '+('worst' if a.worst_examples else 'earliest')+'1s per DEV root',xlabel='World x (m)',ylabel='World y (m)',aspect='equal');ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
fig.savefig(a.output/'prediction_xy.png',dpi=160);plt.close(fig)
summary=[]
for route in range(9):
    for horizon in [.1,.3,1.,2.,3.]:
        rows=[r for r in records if r['route']==route and r['horizon_s']==horizon]
        if not rows:continue
        summary.append(dict(route=route,horizon_s=horizon,windows=len(rows),roots=len({r['root'] for r in rows}),
                            mean_max_xy_m=float(np.mean([r['max_xy_m'] for r in rows])),p95_max_xy_m=float(np.quantile([r['max_xy_m'] for r in rows],.95)),max_xy_m=max(r['max_xy_m'] for r in rows),
                            p95_max_support_xy_m=float(np.quantile([r['max_support_xy_m'] for r in rows],.95)),max_support_xyz_m=max(r['max_support_xyz_m'] for r in rows)))
identity=dict(alpha_conditioned=model.alpha_conditioned,model_root=str(a.model_root),terrain_source='Known evaluation ditch_terrain; weights and TRAIN normalizers unchanged',checkpoint=a.checkpoint,example_selection='worst available1s' if a.worst_examples else 'earliest available1s',checkpoint_sha256=hashlib.sha256((a.model_root/a.checkpoint).read_bytes()).hexdigest(),evaluation=str(a.evaluation),role='Existing DEV_ROOT; no policy simulation, no future true controller input, not independent TEST',elapsed_s=time.perf_counter()-started)
for name,value in [('windows',records),('summary',summary),('coverage',coverage),('identity',identity)]: (a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
np.savez_compressed(a.output/'examples.npz',**examples)
lines=['# Phase2 history prediction DEV evaluation','','![All route actual/predicted XY](prediction_xy.png)','',json.dumps(identity),'',
       '|Route|Horizon s|Windows/roots|Mean max XY m|P95 max XY m|Worst XY m|P95 planar support m|Worst3D support m|',
       '|---|---:|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"|{r['route']}|{r['horizon_s']}|{r['windows']}/{r['roots']}|{r['mean_max_xy_m']:.4f}|{r['p95_max_xy_m']:.4f}|{r['max_xy_m']:.4f}|{r['p95_max_support_xy_m']:.4f}|{r['max_support_xyz_m']:.4f}|")
lines+=['','Windows overlap and end before actual termination. Missing long windows are not padded or counted as successful predictions. Axis support error is not a safety guarantee. No new physical success rate follows from predictor accuracy.']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(identity,indent=2))
