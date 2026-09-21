"""Frozen-trajectory V2 audit and complete-episode development metrics."""
from pathlib import Path
import json
import numpy as np
from .priority_return_v2 import initial_state,advance,reward_terms

def replay(trace,config,alpha=None,*,return_raw=False):
    dt=config['controller']['dt'];end=config['horizon_seconds'];s=initial_state(xp=np);rows=[];previous=np.zeros(2)
    arc=0.;vprev=float(trace['reference_command'][0,0]);totals={};allparts=[];allraw=[]
    for i in range(1,len(trace['time'])):
        v,w=trace['reference_command'][i-1];vnext,wnext=trace['reference_command'][i];arc+=float(v)*dt
        ev=float(trace['true_forward_speed'][i]-v);ey,ep=map(float,trace['path_features'][i,:2]);roll,rate=map(float,trace['measurement'][i,:2]);failed=bool(trace.get('physical_failed',trace['terminated'])[i]);old=s
        s,events=advance(s,t=i*dt,dt=dt,episode_end=end,reference_speed=float(vnext),reference_yaw=float(wnext),published_yaw_request=float(trace["raw_reference_request"][i,1]) if "raw_reference_request" in trace else float(wnext),previous_reference_speed=float(v),reference_progress=arc,actual_progress=float(trace['path_progress'][i]),speed_error=ev,lateral_error=ey,heading_error=ep,roll=roll,roll_rate=rate,failed=failed,xp=np)
        a=float(trace['priority_alpha'][i-1]) if alpha is None else alpha
        r=reward_terms(alpha=a,speed_error=ev,lateral_error=ey,heading_error=ep,roll=roll,roll_rate=rate,action=trace.get('effective_action',trace['action'])[i],previous_action=previous,relaxation=float(s.q),dt=dt,failed=failed,task_penalty_already_paid=bool(old.penalty_paid),**{k:bool(events[k]) for k in ('missed_deadline_now','terminal_incomplete_now')},xp=np)
        previous=trace.get('effective_action',trace['action'])[i];allparts.append(r['reward_parts']);allraw.append(r['raw_costs']);rows.append({k:float(np.asarray(v)) for k,v in vars(s).items()})
    return (s,rows,allparts,allraw) if return_raw else (s,rows,allparts)

def metrics(trace,config):
    from .priority_return_v2 import huber
    t=trace;dt=config['controller']['dt'];n=len(t['time'])-1;ev=t['true_forward_speed'][1:]-t['reference_command'][:-1,0];ey=t['path_features'][1:,0];ep=t['path_features'][1:,1];roll=abs(t['measurement'][1:,0]);failed=bool(t.get('physical_failed',t['terminated'])[-1]);s,states,parts=replay(t,config)
    full=n*dt>=config['horizon_seconds']-1e-6
    eng=bool(full and not failed and s.task_complete and not s.deadline_missed)
    out={'transitions':n,'physical_failed':failed,'full_horizon':full,'v2_engineering_accepted':eng,'v2_strict_accepted':bool(eng and roll.max()<=.3 and np.maximum(ev,0).max()<=.05),'v2_deadline_missed':bool(s.deadline_missed),'v2_final_complete':bool(s.task_complete),'v2_exit_valid':bool(s.exit_valid),'v2_exit_progress':float(s.exit_progress),'v2_deadline':float(s.deadline),'v2_late_recovery':bool(s.late_but_finally_recovered),'v1_deadline_missed':bool(t['return_state'][-1,7]),'v1_final_hold':bool(t['return_state'][-1,4]>=.5-1e-6 and not t['return_state'][-1,2]),'roll_peak':float(roll.max()),'roll_excess_seconds':float(np.sum(roll>.3)*dt),'overspeed_peak':float(np.maximum(ev,0).max()),'overspeed_seconds':float(np.sum(ev>.05)*dt),'negative_rear_action_seconds':float(np.sum(t['action'][1:,1]<0)*dt),'v2_return':float(sum(sum(p.values()) for p in parts)),'recorded_return':float(t['reward'][1:].sum())}
    schedule=np.asarray(t['command_schedule']);changes=np.flatnonzero(np.any(schedule[1:,1:3]!=schedule[:-1,1:3],axis=1))+1
    task_start=float(schedule[changes[0],0]) if len(changes) else 0.
    out['task_window_start_s']=task_start
    for name,mask in [('full',np.ones(n,bool)),('task',np.arange(n)*dt>=task_start)]:
        for key,x in [('speed',ev),('path',ey),('heading',ep),('under',np.maximum(-ev,0)),('over',np.maximum(ev,0))]:
            z=x[mask];
            if not len(z):
                for stat in ('rmse','iae','peak','p95'):out[f'{name}_{key}_{stat}']=None
                continue
            out[f'{name}_{key}_rmse']=float(np.sqrt(np.mean(z*z)));out[f'{name}_{key}_iae']=float(abs(z).sum()*dt);out[f'{name}_{key}_peak']=float(abs(z).max());out[f'{name}_{key}_p95']=float(np.percentile(abs(z),95))
    out['Jp']=float(np.mean(huber(ey/.1,xp=np)+.3*huber(ep/.15,xp=np)));out['Jv']=float(np.mean(huber(ev/.05,xp=np)))
    pre=round(task_start/dt)
    out['precommand_speed']=float(t['true_forward_speed'][min(pre,n)]);out['postcommand_min_speed']=float(t['true_forward_speed'][min(pre+1,n):].min());out['active_slowdown_observed']=out['postcommand_min_speed']<out['precommand_speed']-.01
    out['command_clipped_fraction']=float(np.mean(np.any(abs(t['prelimit_command'][1:]-t['command'][1:])>1e-5,axis=1)))
    return out,states,parts

def select(candidates,alpha=0.):
    def choose(xs):
        if not xs:return None
        if alpha==.5:return min(xs,key=lambda x:.5*(x['Jp']+x['Jv']))
        primary,secondary=('Jp','Jv') if alpha==0 else ('Jv','Jp');best=min(x[primary] for x in xs);eps=max(.02*best,1e-4)
        return min((x for x in xs if x[primary]<=best+eps),key=lambda x:x[secondary])
    return {'best_diagnostic':choose(candidates),'best_accepted':choose([x for x in candidates if x['engineering_accepted']]),'best_strict':choose([x for x in candidates if x['strict_accepted']]),'selection':'equal scenario weights; primary 2% tolerance then secondary, no reward selection'}


def development_stop_reason(history):
    """Predeclared three-point trend gate; no automatic parameter retries."""
    if len(history)<3:return None
    a,b,c=history[-3:]
    if a['physical_failures']<b['physical_failures']<c['physical_failures']:
        return 'physical failures increased at three consecutive frozen evaluations'
    if a['Jp']<b['Jp']<c['Jp']:
        return 'primary path metric worsened at three consecutive frozen evaluations'
    return None


def absorbed_returns(rewards,horizon_steps,gamma=.9995):
    """Fixed horizon: terminal reward once, zero-reward absorbing tail.

    This describes the implemented task, not a new failure penalty or a
    guarantee that failing can never improve an unbounded cost objective.
    """
    rewards=np.asarray(rewards,dtype=float)
    if len(rewards)>horizon_steps or not np.isfinite(rewards).all():raise ValueError('invalid finite episode rewards')
    padded=np.pad(rewards,(0,horizon_steps-len(rewards)))
    return {'undiscounted':float(padded.sum()),'discounted':float(padded@np.power(gamma,np.arange(horizon_steps))),
            'actual_transitions':len(rewards),'absorbing_transitions':horizon_steps-len(rewards)}


def comparison_figure(paths,labels,output,title):
    """Research artifact from complete saved physical traces, with explicit rescore."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for index,(path,label) in enumerate(zip(paths,labels)):
        path=Path(path);tr=dict(np.load(path/'trace.npz'));cfg=json.loads((path/'declaration.json').read_text())['config']
        _,_,parts=replay(tr,cfg,0.);reward=np.array([sum(p.values()) for p in parts]);steps=np.arange(1,len(reward)+1)
        if index==0:axes[0,0].plot(tr['reference_pose'][:,0],tr['reference_pose'][:,1],'k--',label='Original published reference')
        line=axes[0,0].plot(tr['pose'][:,0],tr['pose'][:,1],label=label)[0];color=line.get_color()
        axes[0,1].plot(steps,reward,color=color,label=label);axes[0,2].plot(steps,np.cumsum(reward),color=color,label=label)
        axes[1,0].plot(steps,tr['path_features'][1:,0],color=color,label=label)
        axes[1,1].plot(steps,tr['true_forward_speed'][1:]-tr['reference_command'][:-1,0],color=color,label=label)
        axes[1,2].plot(steps,tr['measurement'][1:,0],color=color,label=label)
        if bool(tr['terminated'][-1]):
            axes[0,0].plot(tr['pose'][-1,0],tr['pose'][-1,1],'x',color=color,ms=9)
            for ax in axes.flat:
                if ax is not axes[0,0]:ax.axvline(steps[-1],color=color,ls=':',alpha=.6)
    axes[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Physical XY trajectories');axes[0,0].axis('equal')
    for ax,heading,ylabel in zip([axes[0,1],axes[0,2],*axes[1]],['V2 replay: per-step reward','V2 replay: cumulative reward','Signed geometric path error','Pre-command aligned speed error','Roll and working band'],['Reward / transition','Cumulative reward','Lateral error (m)','Forward speed error (m/s)','Roll (rad)']):
        ax.set(xlabel='Control steps (200 Hz)',ylabel=ylabel,title=heading)
        commands=np.asarray(tr['command_schedule']);dt=cfg['controller']['dt']
        if len(commands)>2:ax.axvspan(commands[1,0]/dt,commands[2,0]/dt,color='gray',alpha=.10)
    for ax,band in [(axes[1,0],.1),(axes[1,1],.05),(axes[1,2],.3)]:
        ax.axhline(band,color='gray',ls='--',lw=.8);ax.axhline(-band,color='gray',ls='--',lw=.8)
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle(title+'\nalpha=0; seed=49001; gray: raw request interval (reference turn continues afterward)',fontsize=12)
    for suffix in ('.png','.pdf'):fig.savefig(output.with_suffix(suffix),dpi=150)
    plt.close(fig)
