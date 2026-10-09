"""PreferenceV5 saved-trajectory evidence. Never launches physics or edits NPZ."""
from pathlib import Path
import importlib.util
import json
import numpy as np

ATTACHMENT = Path(__file__).resolve().parents[3] / 'docs/preference_v5/attachment'
ATTACHMENT51 = Path(__file__).resolve().parents[3] / 'docs/preference_v51/attachment'
DT = .005

def _write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def _load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def _window(d,lo,hi):return (d['time']>=lo-1e-6)&(d['time']<hi-1e-6)

def _longest(mask):
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
    return float(max(edges[1::2]-edges[::2],default=0)*DT)

def _stats(x):
    x=np.asarray(x)
    if not len(x):return {'count':0}
    if not np.isfinite(x).all():return {'count':len(x),'nonfinite':True}
    return dict(count=len(x),mean=float(x.mean()),rms=float(np.sqrt(np.mean(x*x))),minimum=float(x.min()),maximum=float(x.max()),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)))

def _proposal(d):
    z=np.tanh(d['latent_z']); return np.column_stack((z[:,0]*np.where(z[:,0]>=0,.25,1.),.2*z[:,1]))

def mode_costs(d,end=None):
    """Exclusive descriptive mode bins; recovery excludes initial straight g>0."""
    t=d['time']; valid=np.ones(len(t),bool) if end is None else _window(d,0,end)
    turned=np.maximum.accumulate(abs(d['limited_command'][:,1])>.01)
    recovery=np.asarray(d.get('recovery_phase',turned&(d['g']>0)),bool)
    conflict=(d['chi']>0)&(d['g']==0)&~recovery
    masks={'ordinary':valid&~recovery&~conflict,'conflict_nonrecovery':valid&conflict,'recovery_after_turn':valid&recovery}
    out={}
    for mode,mask in masks.items():
        totals={k.removeprefix('scored_cost_'):float(np.sum(v[mask])*DT) for k,v in d.items() if k.startswith('scored_cost_')}
        if not totals:totals={k.removeprefix('effective_cost_'):float(np.sum(v[mask])*DT) for k,v in d.items() if k.startswith('effective_cost_')}
        out[mode]={'seconds':float(mask.sum()*DT),'cost_integrals':totals,'top3':sorted(totals.items(),key=lambda kv:-kv[1])[:3], 'cap_fraction':{k.removeprefix('component_capped_'):float(np.mean(v[mask])) if mask.any() else None for k,v in d.items() if k.startswith('component_capped_')}}
    return out

def chain_audit(d):
    out={'semantics':'lower_reference_centered=true: u_nom and u_goal use governed reference. requested_residual includes scaled lower_action. Raw-command ECBC output is not logged; do not infer unique upper/lower motor decomposition.','missing':['raw_command_ECBC_output'],'windows':{}}
    for name,lo,hi in [('full',0,float(d['time'][-1]+DT)),('legacy_1_6',1,6),('steady_2_4',2,4)]:
        mask=_window(d,lo,hi); w={}
        fields={k:d[k] for k in ['u_nom','u_goal','requested_residual','applied_residual','lower_action','final_command','wheel_speed_proxy','slip_proxy','governed','actual_forward_speed','actual_delta'] if k in d}
        fields['upper_offset']=d['governed']-d['limited_command']
        if 'latent_z' in d:fields['proposal_offset']=_proposal(d)
        for k,v in fields.items():w[k]=_stats(v[mask]) if v.ndim==1 else [_stats(v[mask,j]) for j in range(v.shape[1])]
        for k in ['proposal_offset','upper_offset']:
            if k in fields:
                w[k+'_negative_dwell']={str(th):{'seconds':float(np.sum(mask&(fields[k][:,0]<=th))*DT),'longest_seconds':_longest(mask&(fields[k][:,0]<=th))} for th in [-.15,-.30]}
        out['windows'][name]=w
    for k in ['u_nom','u_goal','requested_residual','applied_residual','lower_action','final_command','wheel_speed_proxy','slip_proxy']:
        if k not in d:out['missing'].append(k)
    out['mode_costs_full']=mode_costs(d);out['mode_costs_10s']=mode_costs(d,10)
    return out

def _plots(traces,out,case,update,end,reward_context=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'alpha0':'#0072b2','alpha1':'#d55e00','B0':'.5'}
    fig,ax=plt.subplots(figsize=(8,7)); ref=next(iter(traces.values()));mask=_window(ref,0,end)
    ax.plot(*ref['reference_xy'][mask].T,'k--',label='original command integral reference')
    for label,d in traces.items():
        m=_window(d,0,end); xy=d['actual_xy'][m]; ax.plot(*xy.T,color=colors.get(label),label=label)
        if np.any(d['physical_failure'][m]):ax.scatter(*xy[-1],marker='x',color=colors.get(label))
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f'{case} | update {update} | {end:g}s | lower R196 alpha=1')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/f'{case}_{end:g}s_xy.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(6,1,figsize=(12,15),sharex=True)
    for j in range(2):axes[j].plot(ref['time'][mask],ref['limited_command'][mask,j],'k',label='raw original issued')
    for label,d in traces.items():
        m=_window(d,0,end);t=d['time'][m];c=colors.get(label)
        for j in range(2):
            axes[j].plot(t,d['governed'][m,j],color=c,ls='--',label=label+' governed')
            if 'latent_z' in d and label!='B0':axes[j].plot(t,(d['limited_command']+_proposal(d))[m,j],color=c,ls=':',alpha=.6,label=label+' request')
            axes[j].plot(t,d[['actual_forward_speed','actual_delta'][j]][m],color=c,label=label+' actual')
        for ax,key in zip(axes[2:],['phi','e_psi_unwrapped','wheel_speed_proxy','slip_proxy']):
            if key in d:ax.plot(t,d[key][m],color=c,label=label)
        if np.any(d['physical_failure'][m]):
            for ax in axes:ax.axvline(t[-1],color=c,ls=':')
    for ax,ylabel in zip(axes,['Speed (m/s)','Steer (rad)','Roll (rad)','Heading error (rad)','Wheel speed proxy (m/s)','Slip proxy (m/s)']):
        ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=3);ax.axvline(1,color='.5',ls=':');ax.axvspan(2,min(4.5,end),alpha=.04,color='green')
    for v in [-.3,.3]:axes[2].axhline(v,color='red',ls=':')
    axes[-1].set_xlabel('Time (s)');fig.suptitle(f'{case} | checkpoint update {update} | original reference fixed');fig.tight_layout();fig.savefig(out/f'{case}_{end:g}s_control.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for label,d in traces.items():
        m=_window(d,0,end)
        for key,ls in [('u_nom',':'),('u_goal','--'),('requested_residual','-.'),('applied_residual','--'),('final_command','-')]:
            if key in d:
                for j,ax in enumerate(axes):ax.plot(d['time'][m],d[key][m,j],ls=ls,label=label+' '+key,alpha=.8)
    for ax,l in zip(axes,['Front motor (rad/s)','Rear motor (rad/s)']):ax.set_ylabel(l);ax.grid(alpha=.25);ax.legend(fontsize=6,ncol=3)
    axes[-1].set_xlabel('Time (s)');fig.suptitle('Governed-centered control chain; raw ECBC unavailable');fig.tight_layout();fig.savefig(out/f'{case}_{end:g}s_motor_chain.png',dpi=130);plt.close(fig)

    _reward_plots(traces,out,case,update,end,reward_context)

def _reward_plots(traces,out,case,update,end,context=None):
    import matplotlib.pyplot as plt
    series={}
    for label,d in traces.items():
        components={k.removeprefix('scored_cost_'):-.1*DT*v for k,v in d.items() if k.startswith('scored_cost_')}
        if 'failure_cost' in d:components['failure']=-d['failure_cost']
        if 'scored_tick_reward' in d:
            scoring=int(np.asarray(d.get('alpha',[0])).reshape(-1)[0])
            series[label+f' (reward alpha{scoring})']=(d,d['scored_tick_reward'],components)
    if context is not None and 'B0' in traces:
        root,horizon=context
        for alpha in [0,1]:
            c=reward_audit(traces['B0'],alpha,root,horizon,include_reconstruction=True)
            if 'passed' not in c:
                series[f'B0 rescored alpha{alpha}']=(traces['B0'],sum(c.values()),c)
        for key in list(series):
            if key.startswith('B0 ('):del series[key]
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for label,(d,total,_) in series.items():
        m=_window(d,0,end);steps=np.arange(1,len(d['time'])+1)[m]
        for ax,values in zip(axes,[total,np.cumsum(total)]):ax.plot(steps,values[m],ls='--' if label.startswith('B0') else '-',label=label)
        if np.any(d['physical_failure'][m]):
            for ax in axes:ax.axvline(steps[-1],ls=':',color='.5')
    for ax,y in zip(axes,['Reward per control step','Cumulative reward']):ax.set_ylabel(y);ax.grid(alpha=.25);ax.legend(fontsize=7)
    axes[-1].set_xlabel('Actual control steps (5 ms)');fig.suptitle(f'{case} | update {update} | {end:g}s');fig.tight_layout();fig.savefig(out/f'{case}_{end:g}s_reward.png',dpi=130);plt.close(fig)
    names=sorted({k for _,_,c in series.values() for k in c})
    if not names:return
    fig,axes=plt.subplots(len(names),2,figsize=(15,2.2*len(names)),squeeze=False,sharex=True)
    for row,name in enumerate(names):
        for label,(d,_,c) in series.items():
            if name not in c:continue
            m=_window(d,0,end);steps=np.arange(1,len(d['time'])+1)[m]
            axes[row,0].plot(steps,c[name][m],ls='--' if label.startswith('B0') else '-',label=label)
            axes[row,1].plot(steps,np.cumsum(c[name])[m],ls='--' if label.startswith('B0') else '-',label=label)
        axes[row,0].set_ylabel(name)
        for ax in axes[row]:ax.grid(alpha=.25)
    axes[0,0].set_title('Signed reward components per control step');axes[0,1].set_title('Cumulative signed components');axes[0,0].legend(fontsize=6)
    for ax in axes[-1]:ax.set_xlabel('Actual control steps (5 ms)')
    fig.tight_layout();fig.savefig(out/f'{case}_{end:g}s_reward_components.png',dpi=110);plt.close(fig)

def audit_saved_v4(source_root,output_root):
    source_root=Path(source_root);out=Path(output_root);out.mkdir(parents=True,exist_ok=True)
    traces={f'alpha{a}':_load(source_root/'fast_turn'/f'alpha{a}.npz') for a in [0,1]}
    result={'new_physics_steps':0,'source':str(source_root),'methods':{k:chain_audit(v) for k,v in traces.items()}}
    _write(out/'chain_audit.json',result);_plots(traces,out,'fast_turn_V4',250,10)
    return result

def reward_audit(d,alpha,root,horizon,include_reconstruction=False):
    module=importlib.util.spec_from_file_location('v5_reference',ATTACHMENT/'reference_math.py')
    import sys
    ref=importlib.util.module_from_spec(module);sys.modules[module.name]=ref;module.loader.exec_module(ref)
    root=Path(root);p=root/f'alpha{alpha}'/('frozen_config_stage1.json' if horizon==5 else 'frozen_config_stage2.json')
    if not p.exists():p=root/f'alpha{alpha}/frozen_config.json'
    if not p.exists():return {'passed':False,'missing_config':str(p)}
    cfg=json.loads(p.read_text());caps=cfg['reward'].get('independent_component_caps',cfg['reward'].get('caps'))
    if caps is None:return {'passed':False,'missing_caps':True}
    ref51=None
    if cfg.get('priority_recovery_v51'):
        module51=importlib.util.spec_from_file_location('v51_reference',ATTACHMENT51/'reward_delta_reference.py')
        ref51=importlib.util.module_from_spec(module51);sys.modules[module51.name]=ref51;module51.loader.exec_module(ref51)
    reconstructed={k:np.zeros(len(d['time'])) for k in caps};reconstructed['failure']=np.zeros(len(d['time']))
    errors={};efferrors={};maxreward=0.;previous=np.zeros(2);prior_rate=np.zeros(2);valid=False
    for start in range(0,len(d['time']),4):
        stop=min(start+4,len(d['time']));last=stop-1
        offset=d['governed'][last]-d['limited_command'][last];rate=(offset-previous)/.02;acc=(rate-prior_rate)/.02 if valid else np.zeros(2)
        total=0.
        for i in range(start,stop):
            raw_yaw_rate=d['limited_command'][i,0]*np.cos(np.deg2rad(cfg['reference']['caster_deg_expected']))*np.tan(d['limited_command'][i,1])/cfg['reference']['wheelbase_m_expected']
            costs=ref.costs(alpha,d['limited_command'][i],d['governed'][i],d['actual_forward_speed'][i],d['actual_delta'][i],d['phi'][i],d['phi_dot'][i],d['e_psi_unwrapped'][i],d['yaw_rate'][i],raw_yaw_rate,d['chi'][i],d['g'][i])
            if cfg.get('priority_recovery_v51'):
                delta=ref51.reward_changes(alpha=alpha,
                    ev=float(d['actual_forward_speed'][i]-d['limited_command'][i,0]),
                    ed=float(d['actual_delta'][i]-d['limited_command'][i,1]),
                    e_heading=float(d['e_psi_unwrapped'][i]),yaw_rate=float(d['yaw_rate'][i]),
                    raw_yaw_rate=float(raw_yaw_rate),chi=float(d['chi'][i]),g=float(d['g'][i]))
                costs.pop('yaw_damping')
                costs['primary_excess']=delta.primary_excess_raw
                costs['yaw_recovery']=delta.yaw_recovery_raw
            rho=.2+.8*(1-d['chi'][last])*np.exp(-(d['e_psi_unwrapped'][last]/.1)**2)
            costs['upper_rate']=.03*rho*np.sum((rate/np.array([1.,.4]))**2)
            costs['upper_acceleration']=.01*rho*np.sum((acc/10)**2)
            for k,value in costs.items():
                rawkey='raw_cost_'+k;effkey='effective_cost_'+k
                if rawkey not in d:return {'passed':False,'missing_component':rawkey}
                errors[k]=max(errors.get(k,0.),abs(float(value)-float(d[rawkey][i])))
                effective=min(float(value),caps[k]);efferrors[k]=max(efferrors.get(k,0.),abs(effective-float(d[effkey][i])))
                reconstructed[k][i]=-.1*DT*effective
                total-=.1*DT*effective
        if np.any(d['physical_failure'][start:stop]):
            remaining=round(horizon/.02)-round(float(d['time'][start])/.02)
            if cfg.get('priority_recovery_v51'):
                gamma=cfg['ppo']['gamma'];bound=cfg['reward']['failure_absorbing_cost_rate']
                total=-cfg['reward']['failure_extra_penalty']-cfg['reward']['scale']*cfg['plant']['policy_dt_s']*bound*(1-gamma**remaining)/(1-gamma)
            else:
                total=ref.failure_reward(remaining)
            for values in reconstructed.values():values[start:stop]=0
            reconstructed['failure'][stop-1]=total
        maxreward=max(maxreward,abs(total-float(np.sum(d['scored_tick_reward'][start:stop]))))
        previous=offset;prior_rate=rate;valid=True
    if include_reconstruction:return reconstructed
    return dict(passed=max(errors.values(),default=0)<.003 and max(efferrors.values(),default=0)<.003 and maxreward<.003,max_raw_errors=errors,max_effective_errors=efferrors,max_policy_reward_error=maxreward,config=str(p),mode_costs=mode_costs(d),caps=caps)

def _stage1_metrics(d):
    mask=_window(d,2,4.5);pre=_window(d,.5,1);allmask=_window(d,0,5)
    peak_values=d.get('peak_roll')
    finite=peak_values is not None and all(np.isfinite(d[k][allmask]).all() for k in ['actual_forward_speed','actual_delta','phi','e_psi_unwrapped','peak_roll'])
    peak=float(np.max(np.abs(peak_values[allmask]))) if peak_values is not None and allmask.any() and np.isfinite(peak_values[allmask]).all() else None
    complete=allmask.sum()==1000 and not np.any(d['physical_failure'][allmask]) and finite
    pre_speed=float(np.mean(d['actual_forward_speed'][pre])) if pre.any() else None;drop=(pre_speed-d['actual_forward_speed']) if pre_speed is not None else np.full(len(mask),np.nan)
    rms=lambda x:float(np.sqrt(np.mean(x*x))) if len(x) else None
    return dict(complete5s=bool(complete),physical_failure=bool(np.any(d['physical_failure'][allmask])),peak_roll=peak,peak_roll_source='logged substep peak_roll' if peak_values is not None else 'MISSING',finite=bool(finite),speed_rmse=rms(d['actual_forward_speed'][mask]-d['limited_command'][mask,0]),steer_rmse=rms(d['actual_delta'][mask]-d['limited_command'][mask,1]),pre_turn_speed_mean=pre_speed,pre_turn_speed_std=float(np.std(d['actual_forward_speed'][pre])) if pre.any() else None,mean_drop=float(np.mean(drop[mask])) if mask.any() and pre_speed is not None else None,drop_ge_0p2_longest_seconds=_longest(mask&(drop>=.2)),chain=chain_audit(d))

def report_stage1(root,update):
    root=Path(root);out=root/f'evaluation{update}';result={'passed':False,'signs':{},'reward_audits':{},'scope':'Stage1 pilot only; not complete dynamic-task success'}
    for sign in ['positive','negative']:
        paths={k:out/sign/(k+'.npz') for k in ['alpha0','alpha1','B0']}
        if not all(p.exists() for p in paths.values()):result['signs'][sign]={'passed':False,'missing':[str(p) for p in paths.values() if not p.exists()]};continue
        traces={k:_load(p) for k,p in paths.items()};_plots(traces,out,sign,update,5,(root,5))
        m={k:_stage1_metrics(d) for k,d in traces.items()};a,b=m['alpha0'],m['alpha1']
        checks={'complete_and_safe':all(x['complete5s'] and x['peak_roll'] is not None and x['peak_roll']<=.302 for x in [a,b]),'alpha0_steer':a['steer_rmse'] is not None and a['steer_rmse']<=.05,'alpha1_speed':b['speed_rmse'] is not None and b['speed_rmse']<=.08,'speed_difference':a['speed_rmse'] is not None and b['speed_rmse'] is not None and a['speed_rmse']-b['speed_rmse']>=.10,'steer_difference':a['steer_rmse'] is not None and b['steer_rmse'] is not None and b['steer_rmse']-a['steer_rmse']>=.01,'alpha0_sustained_actual_drop':a['drop_ge_0p2_longest_seconds']>=.30}
        audits={f'alpha{x}':reward_audit(traces[f'alpha{x}'],x,root,5) for x in [0,1]};result['reward_audits'][sign]=audits
        result['signs'][sign]={'passed':all(checks.values()),'checks':checks,'metrics':m}
    result['physical_gate_passed']=all(s['passed'] for s in result['signs'].values())
    result['passed']=result['physical_gate_passed'] and len(result['reward_audits'])==2 and all(a['passed'] for s in result['reward_audits'].values() for a in s.values())
    _write(out/'stage1_gate.json',result)
    lines=[f'# PreferenceV5 stage1 update {update}','',f"Pilot gate passed: {result['passed']}. Full task success is not evaluated.",'']
    for sign in result['signs']:lines += [f'![{sign} XY]({sign}_5s_xy.png)',f'![{sign} control]({sign}_5s_control.png)',f'![{sign} reward]({sign}_5s_reward.png)',f'[Signed reward components]({sign}_5s_reward_components.png)','']
    lines+=['Metrics and independent reward reconstruction: [stage1_gate.json](stage1_gate.json).']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');return result

def report_stage2(root,update):
    from .smooth_command_reporting import physical
    root=Path(root);out=root/f'evaluation{update}';result={}
    for case in sorted(out.iterdir()):
        if not case.is_dir() or not (case/'alpha0.npz').exists():continue
        traces={k:_load(case/(k+'.npz')) for k in ['alpha0','alpha1','B0'] if (case/(k+'.npz')).exists()}
        _plots(traces,out,case.name,update,10,(root,16));result[case.name]={k:{'main10':physical(d,10),'chain':chain_audit(d),'v51_contract':_v51_contract_metrics(d,10)} for k,d in traces.items()}
        for a in [0,1]:result[case.name][f'alpha{a}']['reward_audit']=reward_audit(traces[f'alpha{a}'],a,root,16)
        if case.name=='fast_turn':
            _plots(traces,out,case.name,update,16,(root,16))
            for k,d in traces.items():
                result[case.name][k]['extension16']=physical(d,16)
                result[case.name][k]['v51_contract_extension16']=_v51_contract_metrics(d,16)
    _write(out/'stage2_metrics.json',result);return result


def _v51_contract_metrics(d,end):
    """Physical V5.1 acceptance inputs; no reward or survival substitution."""
    valid=_window(d,0,end);steady=valid&_window(d,2,4);pre=valid&_window(d,.5,1.)
    ev=d['actual_forward_speed']-d['limited_command'][:,0]
    ed=d['actual_delta']-d['limited_command'][:,1]
    pre_speed=float(np.mean(d['actual_forward_speed'][pre])) if pre.any() else None
    drop=np.zeros(len(valid)) if pre_speed is None else pre_speed-d['actual_forward_speed']
    rms=lambda x:float(np.sqrt(np.mean(x*x))) if len(x) else None
    return dict(
        horizon_s=end, steady_window_s=[2.,4.],
        steady_speed_rmse_m_s=rms(ev[steady]), steady_steer_rmse_rad=rms(ed[steady]),
        steady_actual_speed_mean_m_s=float(np.mean(d['actual_forward_speed'][steady])) if steady.any() else None,
        steady_actual_steer_mean_rad=float(np.mean(d['actual_delta'][steady])) if steady.any() else None,
        peak_abs_roll_rad=float(np.max(np.abs(d['peak_roll'][valid]))) if valid.any() else None,
        roll_over_0p302_s=float(np.sum(np.abs(d['peak_roll'][valid])>.302)*DT),
        preturn_actual_speed_mean_m_s=pre_speed,
        actual_drop_ge_0p1_longest_s=_longest(valid&(drop>=.1)),
        actual_drop_ge_0p2_longest_s=_longest(valid&(drop>=.2)),
        component_integrals_and_caps=mode_costs(d,end))
