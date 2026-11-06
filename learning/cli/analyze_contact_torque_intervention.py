"""Paired G5 effects with shared observation windows and actual submitted torques."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
c=json.loads((a.run/'contract.json').read_text());s=c['settings'];protocol=s['contact_torque_intervention'];n=s['num_envs']
index=json.loads((a.run/'contact_intervention/index.json').read_text())
chunks=[]
for item in index['chunks']:
    with np.load(a.run/'contact_intervention'/item['file']) as archive:chunks.append({k:archive[k] for k in archive.files})
physics={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]};del chunks
with np.load(a.run/'trajectory.npz') as archive:
    pose=archive['pose'];alive=archive['alive'].astype(bool);measurement=archive['measurement'];trace=archive['lower_trace'].reshape(len(pose),n,11)
status=json.loads((a.run/'status.json').read_text());time=(np.arange(len(pose))+1)*c['lower_dt'];pt=physics['time_s'];dt=index['physics_dt_s']
seed={i:g['seed'] for g in s['fixed_randomization_groups'] for i in g['env_ids']}
identity=lambda i:(seed[i],s['lane_speeds'][i],s['alphas'][i])
baselines={identity(i):i for i,m in enumerate(protocol['modes']) if m=='none'}
start,end=protocol['active_window_s'];windows={'before':(0,start),'intervention':(start,end),'after':(end,len(pose)*c['lower_dt']+1e-9)}
rows=[]
for lane,mode in enumerate(protocol['modes']):
    if mode=='none':continue
    base=baselines[identity(lane)];row=dict(seed=seed[lane],speed_m_s=s['lane_speeds'][lane],alpha=s['alphas'][lane],mode=mode,label=s['case_labels'][lane],lane=lane,baseline_lane=base,windows={},failures=int(status['failures'][lane]),baseline_failures=int(status['failures'][base]),observed_s=float(alive[:,lane].sum()*c['lower_dt']),baseline_observed_s=float(alive[:,base].sum()*c['lower_dt']))
    lane_windows=dict(windows)
    if mode=='pulse':
        pulse_start,pulse_end=protocol['pulse_window_s']
        lane_windows.update(pulse_first_20ms=(pulse_start,pulse_start+.02),pulse_recovery_100ms=(pulse_end,pulse_end+.1))
    for name,(lo,hi) in lane_windows.items():
        m=(time>=lo)&(time<hi)&alive[:,lane]&alive[:,base]
        pm=(pt>=lo)&(pt<hi)&physics['alive_before'][:,lane]&physics['alive_before'][:,base]
        if not m.any() or not pm.any():
            row['windows'][name]=dict(shared_observed_s=0.);continue
        pair={}
        for label,i in [('intervention',lane),('baseline',base)]:
            pair[label]=dict(path_rmse_m=float(np.sqrt(np.mean(trace[m,i,3]**2))),speed_rmse_m_s=float(np.sqrt(np.mean((trace[m,i,2]-s['lane_speeds'][i])**2))),roll_rms_rad=float(np.sqrt(np.mean(measurement[m,i,0]**2))),roll_peak_abs_rad=float(abs(measurement[m,i,0]).max()),steer_rate_rms_rad_s=float(np.sqrt(np.mean(physics['steer_velocity'][pm,i]**2))))
        difference={k:pair['intervention'][k]-pair['baseline'][k] for k in pair['baseline']}
        requested=physics['requested_next_torque'][pm,lane];bounded=physics['bounded_next_torque'][pm,lane];submitted=physics['submitted_previous_torque'][pm,lane]
        row['windows'][name]=dict(shared_observed_s=float(m.sum()*c['lower_dt']),**pair,intervention_minus_baseline=difference,paired_xy_rms_m=float(np.sqrt(np.mean(np.sum((pose[m,lane,:2]-pose[m,base,:2])**2,axis=1)))),paired_steer_rate_peak_difference_rad_s=float(abs(physics['steer_velocity'][pm,lane]-physics['steer_velocity'][pm,base]).max()),submitted_signed_impulse_Nms=float(submitted.sum()*dt),submitted_absolute_impulse_Nms=float(abs(submitted).sum()*dt),clipped_request_duration_s=float((abs(requested-bounded)>1e-6).sum()*dt))
    rows.append(row)
report=dict(role='G5 paired DEV; shared surviving windows plus separate full failure/observation lengths; external generalized torque submission is not a measured contact cancellation fraction',code_protocol=index,conditions=rows)
(a.run/'contact_intervention_effects.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(paired_conditions=len(rows),output=str(a.run/'contact_intervention_effects.json'))))

# Paired figures retain each condition and actual observation endpoint.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figures=a.run/'intervention_figures';figures.mkdir(exist_ok=True)
route=np.load(Path(c['bundle'])/'route.npz')['xy'];colors=['#333333','#0072B2','#D55E00','#009E73','#CC79A7'];styles=[':','-','--','-.','-'];manifest=[]
for gi,(key,base) in enumerate(baselines.items()):
    lanes=[i for i in range(n) if identity(i)==key];fig,axes=plt.subplots(3,2,figsize=(11,11),layout='constrained');xy,steer_ax,roll_ax,path_ax,speed_ax,torque_ax=axes.flat
    xy.plot(*route.T,':',color='.6',label='Original reference');extent=[]
    for j,lane in enumerate(lanes):
        label=s['case_labels'][lane].split('_alpha')[-1];valid=alive[:,lane];pvalid=physics['alive_before'][:,lane].astype(bool)
        color=colors[j];style=styles[j];points=np.where(valid[:,None],pose[:,lane,:2],np.nan)
        xy.plot(*points.T,color=color,ls=style,label=label,lw=1.1)
        ids=np.flatnonzero(valid)
        if len(ids):xy.plot(*pose[ids[-1],lane,:2],marker='x',color=color);extent.append(pose[valid,lane,:2])
        steer_ax.plot(pt,np.where(pvalid,physics['steer_velocity'][:,lane],np.nan),color=color,ls=style,lw=.65)
        roll_ax.plot(time,np.where(valid,np.rad2deg(measurement[:,lane,0]),np.nan),color=color,ls=style)
        path_ax.plot(time,np.where(valid,trace[:,lane,3],np.nan),color=color,ls=style)
        speed_ax.plot(time,np.where(valid,trace[:,lane,2]-key[1],np.nan),color=color,ls=style)
        torque_ax.plot(pt,np.where(pvalid,physics['submitted_previous_torque'][:,lane],np.nan),color=color,ls=style,lw=.65)
    points=np.concatenate(extent);xy.set_xlim(points[:,0].min()-.3,points[:,0].max()+.3);xy.set_ylim(points[:,1].min()-.3,points[:,1].max()+.3)
    xy.set(xlabel='World x (m)',ylabel='World y (m)');xy.set_aspect('equal',adjustable='box');xy.legend(fontsize=8)
    for ax,label in [(steer_ax,'Steering rate (rad/s)'),(roll_ax,'Roll (deg)'),(path_ax,'Signed path error (m)'),(speed_ax,'Speed error (m/s)'),(torque_ax,'Submitted external torque (Nm)')]:
        ax.set(xlabel='Time (s)',ylabel=label);ax.axvspan(start,end,color='.5',alpha=.1)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f'Frozen H75 + original Local300 | G5 DEV | seed{key[0]} | {key[1]} m/s | alpha={key[2]:g}\nCrosses: last observed endpoints; gray interval: declared intervention window; raw samples, no smoothing')
    filename=f'condition_{gi:03d}.png';fig.savefig(figures/filename,dpi=140);plt.close(fig)
    manifest.append(dict(seed=key[0],speed_m_s=key[1],alpha=key[2],figure=filename,lanes=lanes,description='Actual/reference XY and paired steering rate, roll, original-path/speed errors, actually submitted external diagnostic torque. No aggregation, smoothing or failure deletion.',sources=['trajectory.npz','contract.json','contact_intervention/index.json']))
(figures/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
