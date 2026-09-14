"""Reward reconstruction and paired raw-command diagnostics, including failures."""
import json
from pathlib import Path
import numpy as np
from .media import state_series

def reconstruct(path):
    path=Path(path);tr=dict(np.load(path/'trace.npz'));c=json.loads((path/'declaration.json').read_text())['config'];m=c['motion_commands']
    v=state_series(tr,c)['speed'];a=tr['priority_alpha'][:-1];t=tr['time'][1:]
    ev=v[1:]-tr['user_command'][:-1,0];ey=tr['yaw_rate_world'][1:]-tr['user_command'][:-1,1]
    parts=dict(attitude=m['attitude_weight']*np.maximum(np.abs(tr['measurement'][1:,0])-m['roll_working_limit'],0)**2,roll_rate=m['rate_weight']*tr['measurement'][1:,1]**2,speed=m['tracking_scale']*10**(2*a-1)*(ev/m['speed_scale'])**2,yaw=m['tracking_scale']*10**(1-2*a)*(ey/m['yaw_scale'])**2,action=.01*np.sum(tr['action'][1:]**2,axis=1))
    rewards={k:-c['controller']['dt']*x for k,x in parts.items()};rewards['alive']=np.full(len(t),c['controller']['dt']*c['alive_reward_rate']);failed=tr['terminated'][1:]
    for x in rewards.values():x[failed]=0
    rewards['failure']=np.where(failed,-c['failure_penalty'],0.)
    prediction=sum(rewards.values());err=float(np.max(abs(prediction-tr['reward'][1:])))
    if not np.allclose(prediction,tr['reward'][1:],rtol=3e-5,atol=3e-5):raise ValueError(f'command reward reconstruction mismatch {path}: {err}')
    return dict(trace=tr,speed=v,speed_error=ev,yaw_error=ey,parts=parts,rewards=rewards,reconstruction_error=err,config=c)

def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True);rows=[];links=[]
    for panel in sorted((root/'evaluation').glob('alpha_*/seed_*')):
        for case in sorted(panel.iterdir()):
            if not (case/'residual/trace.npz').exists():continue
            x=reconstruct(case/'residual');b=reconstruct(case/'baseline');tr=x['trace'];bt=b['trace'];c=x['config']
            dest=out/panel.parent.name/panel.name/case.name;dest.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(3,2,figsize=(14,12),layout='constrained');tt=tr['time'][1:]
            for data,label in [(b,'ECBC + ESO'),(x,'Residual')]:
                z=data['trace'];t=z['time'][1:]
                for ax,val in zip(axes.flat,[data['speed_error'],data['yaw_error'],np.rad2deg(z['measurement'][1:,0])]):
                    ax.plot(t,val,label=label)
                    if z['terminated'][-1]:ax.plot(t[-1],val[-1],'x',ms=9)
            for key in ('speed','yaw','attitude','roll_rate'):axes[1,1].plot(tt,x['parts'][key],label=key)
            for key,val in x['rewards'].items():axes[2,0].plot(tt,np.cumsum(val),label=key)
            keys=list(x['parts']);common=min(tr['time'][-1],bt['time'][-1]);means=[]
            for data in (b,x):
                mask=(data['trace']['time'][1:]<=common)&~data['trace']['terminated'][1:]
                means.append([float(val[mask].mean()) if mask.any() else 0. for val in data['parts'].values()])
            bottom=np.zeros(2)
            for i,key in enumerate(keys):
                val=np.asarray(means)[:,i];axes[2,1].bar([0,1],val,bottom=bottom,label=key);bottom+=val
            axes[2,1].set_xticks([0,1],['Baseline','Residual'])
            titles=['Speed error vs original command [m/s]','Yaw-rate error vs original command [rad/s]','Actual roll [deg]','Penalty components [reward/s]','Signed component returns','Mean penalties over common surviving window']
            schedule=json.loads((case/'residual/commands.json').read_text())['schedule']
            for ax,title in zip(axes.flat,titles):
                ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
            for ax in list(axes.flat)[:5]:
                ax.set_xlabel('Time [s]')
                for j,row in enumerate(schedule[1:]):
                    end=schedule[j+2][0] if j+2<len(schedule) else c['horizon_seconds']
                    ax.axvspan(row[0],end,color='gray',alpha=.05 if j%2 else .13)
                ax.set_xlim(0,c['horizon_seconds'])
            axes[1,1].set_yscale('symlog',linthresh=.1)
            limit=np.rad2deg(c['motion_commands']['roll_working_limit'])
            for val in (-limit,limit):axes[1,0].axhline(val,ls=':',color='gray')
            fig.suptitle(f'{case.name} | alpha={tr["priority_alpha"][0]:g} | {panel.name}\nShading: command intervals; x: failure. Dotted roll bound is a soft working target, not a guarantee.')
            for ext in ('png','pdf'):fig.savefig(dest/f'reward_and_errors.{ext}',dpi=140)
            plt.close(fig)
            fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
            for data,label in [(b,'Baseline'),(x,'Residual')]:
                z=data['trace'];t=z['time'];estimate=z['measurement'][:,5]*.1
                axes[0,0].plot(t,data['speed'],label=label+' true');axes[0,0].plot(t,estimate,ls=':',label=label+' wheel estimate')
                axes[0,1].plot(t,estimate-data['speed'],label=label)
                axes[1,0].plot(t,z['yaw_rate_world'],label=label)
                axes[1,1].plot(z['pose'][:,0],z['pose'][:,1],label=label)
                if z['terminated'][-1]:
                    for ax,value in [(axes[0,0],data['speed'][-1]),(axes[0,0],estimate[-1]),(axes[0,1],estimate[-1]-data['speed'][-1]),(axes[1,0],z['yaw_rate_world'][-1])]:ax.plot(t[-1],value,'x',ms=8)
                    axes[1,1].plot(z['pose'][-1,0],z['pose'][-1,1],'x',ms=8)
            axes[0,0].step(tr['time'],tr['user_command'][:,0],where='post',color='black',ls='--',label='Original speed request')
            axes[1,0].step(tr['time'],tr['user_command'][:,1],where='post',color='black',ls='--',label='Original yaw-rate request')
            for ax,title in zip(axes.flat,['True / estimated / requested speed [m/s]','Speed estimate error [m/s]','Actual / requested yaw rate [rad/s]','XY motion [m]; no reference path in this task']):ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
            axes[1,1].set_aspect('equal',adjustable='datalim')
            for ax in (axes[0,0],axes[0,1],axes[1,0]):
                for row in schedule[1:]:ax.axvline(row[0],color='gray',alpha=.4)
            for ext in ('png','pdf'):fig.savefig(dest/f'speed_yaw_trajectory.{ext}',dpi=140)
            plt.close(fig)
            np.savez_compressed(dest/'speed_pair.npz',baseline_time=bt['time'],baseline_true=b['speed'],baseline_estimate=bt['measurement'][:,5]*.1,baseline_target=bt['user_command'][:,0],residual_time=tr['time'],residual_true=x['speed'],residual_estimate=tr['measurement'][:,5]*.1,residual_target=tr['user_command'][:,0])
            np.savez_compressed(dest/'components.npz',time=tt,speed_error=x['speed_error'],yaw_error=x['yaw_error'],**{k:v for k,v in x['rewards'].items()})
            for data,label in [(b,'baseline'),(x,'residual')]:
                z=data['trace'];rows.append(dict(case=case.name,alpha=float(z['priority_alpha'][0]),seed=panel.name,policy=label,failed=bool(z['terminated'][-1]),observed_seconds=float(z['time'][-1]),speed_rmse=float(np.sqrt(np.mean(data['speed_error']**2))),yaw_rmse=float(np.sqrt(np.mean(data['yaw_error']**2))),roll_peak=float(abs(z['measurement'][:,0]).max()),reward_reconstruction_error=data['reconstruction_error']))
            links.append(f'- [{case.name} {panel.parent.name} {panel.name}]({dest.relative_to(out)}/reward_and_errors.png) · [speed/yaw/XY]({dest.relative_to(out)}/speed_yaw_trajectory.png)')
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'INDEX.md').write_text('# Command tracking diagnostics\n\nAll errors refer to original requests. Observed-window RMSE of failed runs is not a full-task ranking. Shading denotes command intervals, not an external load.\n\n'+'\n'.join(links)+'\n')
    return rows
