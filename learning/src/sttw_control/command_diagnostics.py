"""Reward reconstruction and paired raw-command diagnostics, including failures."""
import json
from pathlib import Path
import numpy as np
from .media import state_series
from .motion_commands import MotionCommands, reward_terms, signed_reward_components, gated_action

def reconstruct(path):
    path=Path(path);tr=dict(np.load(path/'trace.npz'));c=json.loads((path/'declaration.json').read_text())['config'];m=MotionCommands(**c['motion_commands'])
    v=state_series(tr,c)['speed'];a=tr['priority_alpha'][:-1];t=tr['time'][1:]
    ev=v[1:]-tr['user_command'][:-1,0];ey=tr['yaw_rate_world'][1:]-tr['user_command'][:-1,1]
    args=(tr['measurement'][1:,0],tr['measurement'][1:,1],ev,ey,gated_action(tr['action'][1:],a,m,xp=np),a,m)
    parts=reward_terms(*args,xp=np)
    rewards=signed_reward_components(*args,c['controller']['dt'],c['alive_reward_rate'],c['failure_penalty'],tr['terminated'][1:],xp=np)
    prediction=sum(rewards.values());err=float(np.max(abs(prediction-tr['reward'][1:])))
    if not np.allclose(prediction,tr['reward'][1:],rtol=3e-5,atol=3e-5):raise ValueError(f'command reward reconstruction mismatch {path}: {err}')
    return dict(trace=tr,speed=v,speed_error=ev,yaw_error=ey,parts=parts,rewards=rewards,reconstruction_error=err,config=c)


def paired_reward_summary(baseline,residual):
    """Signed return differences with explicit observed and shared-step scopes.

    A failure transition is retained, including its replacement penalty. The
    observed-episode delta can compare different durations and is not a task
    success score. Common-window deltas stop at the earlier recorded endpoint.
    """
    b,x=baseline['trace'],residual['trace']
    count=min(len(b['time']),len(x['time']))-1
    if count<1:raise ValueError('paired reward summary requires control transitions')
    if not np.allclose(b['time'][:count+1],x['time'][:count+1],rtol=0.,atol=1e-9):
        raise ValueError('paired control time mismatch')
    for field in ('user_command','priority_alpha'):
        if not np.allclose(b[field][:count],x[field][:count],rtol=0.,atol=1e-7):
            raise ValueError('paired command or alpha mismatch')

    def summarize(data,steps):
        tr=data['trace'];window=slice(1,steps+1)
        components={key:float(np.sum(value[:steps],dtype=np.float64)) for key,value in data['rewards'].items()}
        observed=float(np.sum(tr['reward'][window],dtype=np.float64))
        failure_times=tr['time'][window][tr['terminated'][window]]
        return dict(observed_return=observed,component_returns=components,component_sum=sum(components.values()),
                    return_reconstruction_error=abs(sum(components.values())-observed),steps=steps,
                    start_seconds=float(tr['time'][0]),end_seconds=float(tr['time'][steps]),
                    terminal_failure_included=bool(len(failure_times)),
                    failure_time_seconds=float(failure_times[0]) if len(failure_times) else None)

    result={label:summarize(data,len(data['trace']['time'])-1) for label,data in [('baseline',baseline),('residual',residual)]}
    result['scope']='each_observed_episode_including_terminal_failure; endpoints_may_differ'
    result['observed_windows_match']=len(b['time'])==len(x['time'])
    result['delta_residual_minus_baseline']=result['residual']['observed_return']-result['baseline']['observed_return']
    common={label:summarize(data,count) for label,data in [('baseline',baseline),('residual',residual)]}
    common.update(scope='same_control_steps_including_terminal_failure',steps=count,end_seconds=float(b['time'][count]))
    common['delta_residual_minus_baseline']=common['residual']['observed_return']-common['baseline']['observed_return']
    result['common_window']=common
    surviving=~(b['terminated'][1:count+1]|x['terminated'][1:count+1])
    survivor_times=b['time'][1:count+1][surviving]
    survivor=dict(scope='same_control_steps_excluding_either_terminal_failure',steps=int(surviving.sum()),
                  end_seconds=float(survivor_times[-1]) if len(survivor_times) else None)
    for label,data in [('baseline',baseline),('residual',residual)]:
        survivor[label+'_mean_penalties']={key:float(np.mean(value[:count][surviving])) if surviving.any() else None
                                           for key,value in data.get('parts',{}).items()}
    result['common_surviving_window']=survivor
    return result

def plot_step_rewards(destination, baseline, residual, schedule, title):
    import matplotlib.pyplot as plt
    destination=Path(destination)
    comparison=paired_reward_summary(baseline,residual)
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    arrays={}
    for data,label,ax in [(baseline,'baseline',axes[1,0]),(residual,'residual',axes[1,1])]:
        tr=data['trace'];steps=np.arange(1,len(tr['time']));reward=tr['reward'][1:]
        arrays[label+'_steps']=steps;arrays[label+'_time']=tr['time'][1:];arrays[label+'_total']=reward
        arrays[label+'_cumulative_return']=np.cumsum(reward,dtype=np.float64)
        arrays[label+'_component_sum']=sum(data['rewards'].values())
        axes[0,0].plot(steps,reward,label=label)
        axes[0,1].plot(steps,np.cumsum(reward),label=label)
        for key,value in data['rewards'].items():
            ax.plot(steps,value,label=key);arrays[label+'_'+key]=value
        if tr['terminated'][-1]:
            axes[0,0].plot(steps[-1],reward[-1],'x',ms=10)
            axes[0,1].plot(steps[-1],np.sum(reward),'x',ms=10)
            ax.axvline(steps[-1],color='red',ls=':',label='Failure endpoint')
        ax.set_title(label+' signed components per control step')
    common=comparison['common_window']['steps']
    arrays['common_steps']=np.arange(1,common+1)
    arrays['common_cumulative_delta']=arrays['residual_cumulative_return'][:common]-arrays['baseline_cumulative_return'][:common]
    axes[0,1].plot(arrays['common_steps'],arrays['common_cumulative_delta'],ls='--',label='Residual - baseline, common steps')
    axes[0,0].set_title('Actual total reward per control step (not cumulative)')
    axes[0,1].set_title('Cumulative return; failed episodes end at failure')
    dt=residual['config']['controller']['dt']
    for ax in axes.flat:
        ax.set_xlabel('Control step');ax.set_ylabel('Reward');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_yscale('symlog',linthresh=.001)
        for i,row in enumerate(schedule[1:]):
            end=schedule[i+2][0] if i+2<len(schedule) else residual['config']['horizon_seconds']
            ax.axvspan(row[0]/dt,end/dt,color='gray',alpha=.06 if i%2 else .13)
    fig.suptitle(title+'\nSigned rewards reconstructed and checked against trace; shading: command intervals. Symlog y-axis.')
    for ext in ('png','pdf'):fig.savefig(destination/f'reward_vs_steps.{ext}',dpi=140)
    plt.close(fig)
    np.savez_compressed(destination/'reward_steps.npz',**arrays)
    (destination/'reward_comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')

def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True);rows=[];pairs=[];links=[]
    for panel in sorted((root/'evaluation').glob('alpha_*/seed_*')):
        for case in sorted(panel.iterdir()):
            if not (case/'residual/trace.npz').exists():continue
            x=reconstruct(case/'residual');b=reconstruct(case/'baseline');tr=x['trace'];bt=b['trace'];c=x['config']
            comparison=paired_reward_summary(b,x)
            pairs.append(dict(case=case.name,alpha=float(tr['priority_alpha'][0]),seed=panel.name,**comparison))
            dest=out/panel.parent.name/panel.name/case.name;dest.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(3,2,figsize=(14,12),layout='constrained');tt=tr['time'][1:]
            for data,label in [(b,'ECBC + ESO'),(x,'Residual')]:
                z=data['trace'];t=z['time'][1:]
                for ax,val in zip(axes.flat,[data['speed_error'],data['yaw_error'],np.rad2deg(z['measurement'][1:,0])]):
                    ax.plot(t,val,label=label)
                    if z['terminated'][-1]:ax.plot(t[-1],val[-1],'x',ms=9)
            for key in x['parts']:
                if key!='action':axes[1,1].plot(tt,x['parts'][key],label=key)
            if 'yaw_tracking' in x['rewards']:
                axes[1,1].plot(tt,x['rewards']['yaw_tracking']/c['controller']['dt'],label='yaw_tracking (+reward)',ls='--')
            for key,val in x['rewards'].items():axes[2,0].plot(tt,np.cumsum(val),label=key)
            keys=list(x['parts']);means=[]
            for label in ('baseline','residual'):
                mean=comparison['common_surviving_window'][label+'_mean_penalties']
                means.append([mean[key] if mean[key] is not None else np.nan for key in keys])
            bottom=np.zeros(2)
            for i,key in enumerate(keys):
                val=np.asarray(means)[:,i];axes[2,1].bar([0,1],val,bottom=bottom,label=key);bottom+=val
            axes[2,1].set_xticks([0,1],['Baseline','Residual'])
            shared=comparison['common_surviving_window']
            titles=['Speed error vs original command [m/s]','Yaw-rate error vs original command [rad/s]','Actual roll [deg]','Penalty components [reward/s]','Signed component returns',f'Mean penalties: {shared["steps"]} shared steps, neither failed']
            if 'yaw_tracking' in x['rewards']:titles[3]='Penalty costs and positive tracking bonus [reward/s]'
            schedule=json.loads((case/'residual/commands.json').read_text())['schedule']
            plot_step_rewards(dest,b,x,schedule,f'{case.name} | alpha={tr["priority_alpha"][0]:g} | {panel.name}')
            for ax,title in zip(axes.flat,titles):
                ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
            for ax in list(axes.flat)[:5]:
                ax.set_xlabel('Time [s]')
                for j,row in enumerate(schedule[1:]):
                    end=schedule[j+2][0] if j+2<len(schedule) else c['horizon_seconds']
                    ax.axvspan(row[0],end,color='gray',alpha=.05 if j%2 else .13)
                ax.set_xlim(0,c['horizon_seconds'])
            axes[1,1].set_yscale('symlog',linthresh=.1)
            mc=c['motion_commands']
            if mc.get('tolerance_penalty_rate',0)>0:
                alpha=float(tr['priority_alpha'][0])
                for ax,threshold,enabled in [(axes[0,0],mc['speed_tolerance'],alpha<1),(axes[0,1],mc['yaw_tolerance'],alpha>0)]:
                    if enabled:
                        for sign in (-1,1):ax.axhline(sign*threshold,color='red',ls=':',label='Tolerance' if sign==1 else None)
                        ax.legend(fontsize=8)
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
                z=data['trace'];rows.append(dict(case=case.name,alpha=float(z['priority_alpha'][0]),seed=panel.name,policy=label,failed=bool(z['terminated'][-1]),observed_seconds=float(z['time'][-1]),speed_rmse=float(np.sqrt(np.mean(data['speed_error']**2))),yaw_rmse=float(np.sqrt(np.mean(data['yaw_error']**2))),roll_peak=float(abs(z['measurement'][:,0]).max()),reward_reconstruction_error=data['reconstruction_error'],return_scope=comparison['scope'],**comparison[label]))
            if c['motion_commands'].get('tolerance_penalty_rate',0)>0:
                for data,row in [(b,rows[-2]),(x,rows[-1])]:
                    for name in ('speed','yaw'):
                        excess=np.abs(data[name+'_error'])>c['motion_commands'][name+'_tolerance']
                        row[name+'_tolerance_exceed_seconds']=float(excess.sum()*c['controller']['dt'])
                        row[name+'_tolerance_exceed_fraction']=float(excess.mean())
            links.append(f'- [{case.name} {panel.parent.name} {panel.name}]({dest.relative_to(out)}/reward_and_errors.png) · [reward vs steps]({dest.relative_to(out)}/reward_vs_steps.png) · [paired returns and components]({dest.relative_to(out)}/reward_comparison.json) · [speed/yaw/XY]({dest.relative_to(out)}/speed_yaw_trajectory.png)')
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'paired_rewards.json').write_text(json.dumps(pairs,indent=2,allow_nan=False)+'\n')
    table=['| Case | Alpha | Seed | Baseline end / failure [s] | Residual end / failure [s] | Baseline observed return | Residual observed return | Residual - baseline | Common end [s] | Common return delta |',
           '| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for pair in pairs:
        b,x=pair['baseline'],pair['residual'];shared=pair['common_window']
        ends=[f'{item["end_seconds"]:g}'+(' / failed' if item['terminal_failure_included'] else '') for item in (b,x)]
        table.append(f'| {pair["case"]} | {pair["alpha"]:g} | {pair["seed"]} | {ends[0]} | {ends[1]} | {b["observed_return"]:.8g} | {x["observed_return"]:.8g} | {pair["delta_residual_minus_baseline"]:.8g} | {shared["end_seconds"]:g} | {shared["delta_residual_minus_baseline"]:.8g} |')
    (out/'INDEX.md').write_text('# Command tracking diagnostics\n\nAll errors refer to original requests. Observed-window RMSE and return of failed runs are not full-task rankings. Shading denotes command intervals, not an external load.\n\nReturns sum actual per-step rewards and include the terminal failure penalty. Observed endpoints may differ; common-window deltas use identical control steps through the earlier endpoint, including any failure at that endpoint. All deltas are signed residual minus baseline, never return ratios. Mean penalty bars instead exclude a step if either policy failed. Per-step reward figures remain the primary reward diagnostic.\n\n[All paired return scopes and component totals](paired_rewards.json)\n\n'+'\n'.join(table)+'\n\n'+'\n'.join(links)+'\n')
    with (out/'INDEX.md').open('a') as f:
        if (out/'trajectory_comparison/INDEX.md').exists():f.write('\n- [Trajectory comparison](trajectory_comparison/INDEX.md)\n')
        media=out/'media_selection.json'
        if media.exists():
            for item in json.loads(media.read_text())['selected']:
                for label in ('baseline','residual'):
                    path=Path(item['directory'])/label/'media/replay.mp4'
                    if path.exists():f.write(f"- [{item['case']} alpha={item['alpha']} {label} video]({path})\n")
    return rows
