"""Complete fixed80 readback: original goal, contact and task costs separately."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial import cKDTree
from sttw_control.preference_best import atomic_json,digest
from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics
from sttw_control.anticipatory_run import physical_outcome
from sttw_control.reference_preservation_confirmation import audit_pulse
from sttw_control.support_permission_audit import verify_correction_chain

MODES=('H75','current','zero','equilibrium')
COLORS=('#0072B2','#D55E00','#333333','#008060')
STYLES=('-','--',':','-.')


def analyze(root):
    root=root.resolve();run=root/'evaluation';out=root/'analysis'
    p=json.loads((root/'plan.json').read_text());status=json.loads((run/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=80:raise ValueError('all80 numerical cases required; no partial selection')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('plan drift')
    for f,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(f)!=sha:raise ValueError('executed source/input drift '+f)
    inventory=json.loads((root/'inventory.json').read_text());index=json.loads((run/'case_index.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['plan_sha256']!=digest(root/'plan.json') or manifest['config']!=p['config'] or manifest['models']!=p['models']:
        raise ValueError('manifest differs from executed plan/config/models')
    if manifest['cases']!=inventory or manifest['source']!=p['initial_source']:raise ValueError('manifest source/cases mismatch')
    if len(index)!=80 or len({c['case_id'] for c in index})!=80:raise ValueError('fixed complete inventory required')
    out.mkdir(exist_ok=False)
    helper=Path(__file__).with_name('audit_native_goal_initial.py')
    subprocess.run([sys.executable,'-B',str(helper),str(root),str(out/'initial_carry_review.json')],check=True)
    initial_review=json.loads((out/'initial_carry_review.json').read_text())
    paths={};goals={};trees={};raws={};rows=[];ticks=0
    for route in {c['route'] for c in index}:
        with np.load(run/(route+'_path.npz')) as z:path={k:z[k] for k in z.files}
        paths[route]=path
        goals[route]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,a]) for a in (0,1)])
        trees[route]=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[route]]))
    for i,c in enumerate(index):
        if any(c.get(k)!=v for k,v in inventory[i].items()):raise ValueError('case identity drift')
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('raw drift')
        np.testing.assert_allclose(c['original_goal_xy'],goals[c['route']],atol=3e-6,rtol=0)
        np.testing.assert_allclose(manifest['original_goals_xy'][i],goals[c['route']],atol=3e-6,rtol=0)
        with np.load(c['raw_file']) as z:raw={k:z[k] for k in z.files}
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum());ticks+=n
        if not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('active prefix broken')
        # Actual time was logged pre-tick. Derive, then verify the integer
        # clock rather than inventing a consecutive clock after flattening.
        clock=np.rint(raw['time']/.005).astype(np.int64)
        np.testing.assert_array_equal(clock[active],np.arange(n))
        np.testing.assert_allclose(raw['time'][active],np.arange(n)*.005,atol=3e-6,rtol=0)
        np.testing.assert_allclose(raw['v_user'][active],c['speed'],atol=2e-7,rtol=0)
        data=dict(raw,physical_tick=clock,policy_fault=raw['policy_fault']|raw['continuation_geometry_fault'])
        pulse=audit_pulse(data,dict(force_N=c['force_N'],world_direction=[0.,1.,0.],
             pulse_onset_task_tick=round(c['onset_s']/.005),pulse_duration_s=c['duration_s']))
        aliases=dict(data,issued_latent=raw['latent_z'],nominal_reference=raw['nominal'],
                     reference=raw['governed'],correction_offset=raw['offsets'])
        chain=verify_correction_chain(aliases,p['config'],initial_filtered=initial_review['initial_filtered'][i],
                                      initial_offsets=initial_review['initial_offsets'][i])
        blocked=active&~raw['reference_target_permitted']
        np.testing.assert_array_equal(raw['latent_z'][blocked],0)
        if c['mode']=='zero':np.testing.assert_array_equal(raw['latent_z'][active],0)
        horizons={}
        fault=data['policy_fault']|raw['nonfinite']|raw['lower_fault']
        for horizon in (10,20):
            cut={k:v[:horizon*200] for k,v in data.items()}
            endpoint=goal_arrival_metrics(cut,goal_xy=goals[c['route']],radius_m=1.,dt=.005,horizon_s=horizon)
            outcome=physical_outcome(cut['active_tick'],cut['contact_failure'],
                      cut['policy_fault']|cut['nonfinite']|cut['lower_fault'],horizon*200)
            horizons[str(horizon)]=dict(**endpoint,physical_outcome=outcome)
        fatal=np.flatnonzero(active&(raw['contact_failure']|fault))
        stop=int(fatal[0]) if len(fatal) else n
        arrival=horizons['20']['first_arrival_s']
        task_stop=min(stop,round(arrival/.005)) if arrival is not None else stop
        xy=raw['actual_xy'][:task_stop];speed=raw['actual_forward_speed'][:task_stop]
        distance=trees[c['route']].query(xy)[0] if task_stop else np.array([])
        path=paths[c['route']]
        prefix=dict(ticks=task_stop,observed_s=task_stop*.005,completed_task=arrival is not None,
            route_sample_distance_rmse_m=float(np.sqrt(np.mean(distance**2))) if task_stop else None,
            speed_rmse_m_s=float(np.sqrt(np.mean((speed-c['speed'])**2))) if task_stop else None,
            speed_mae_m_s=float(np.mean(abs(speed-c['speed']))) if task_stop else None,
            path_length_m=float(np.linalg.norm(np.diff(np.vstack([path['xy'][0],xy]),axis=0),axis=1).sum()),
            peak_roll_rad=float(raw['peak_roll'][:task_stop].max()) if task_stop else None,
            mean_abs_speed_reference_offset_m_s=float(np.mean(abs(raw['governed'][:task_stop,0]-raw['nominal'][:task_stop,0]))) if task_stop else None,
            mean_abs_steer_reference_offset_rad=float(np.mean(abs(raw['governed'][:task_stop,1]-raw['nominal'][:task_stop,1]))) if task_stop else None)
        replanned=run/f'replanned_path_{i:02d}.npz'
        if replanned.exists()!=c['replanned']:raise ValueError('replan file/flag mismatch')
        if replanned.exists():
            with np.load(replanned) as z:
                np.testing.assert_allclose([np.interp(float(z['goal']),z['s'],z['xy'][:,a]) for a in (0,1)],goals[c['route']],atol=3e-6,rtol=0)
        rows.append(dict(c,horizons=horizons,prefix=prefix,active_lower_ticks=n,pulse=pulse,correction_chain=chain,
            pulse_started_after_first_arrival=bool(arrival is not None and arrival<=c['onset_s']) if c['force_N'] else None,
            diagnostic_current_path_scored_cost=float(-raw['scored_tick_reward'][active].sum())))
        raws[c['case_id']]=raw
    if ticks!=status['active_lower_ticks']:raise ValueError('raw charge mismatch')
    summary={}
    for horizon in ('10','20'):
        summary[horizon]={}
        for mode in MODES:
            summary[horizon][mode]={}
            for subset in ('all','nominal','disturbed'):
                group=[r for r in rows if r['mode']==mode and (subset=='all' or bool(r['force_N'])==(subset=='disturbed'))]
                hs=[r['horizons'][horizon] for r in group]
                summary[horizon][mode][subset]=dict(n=len(group),arrived=sum(h['goal_arrived'] is True for h in hs),
                    survived=sum(h['physical_outcome']=='survived' for h in hs),contact=sum(h['physical_outcome']=='contact' for h in hs),
                    unknown=sum(h['physical_outcome']=='unknown' or h['goal_arrived'] is None for h in hs),
                    arrived_and_survived=sum(h['goal_arrived'] is True and h['physical_outcome']=='survived' for h in hs),
                    arrived_then_contact=sum(h['goal_arrived'] is True and h['physical_outcome']=='contact' for h in hs))
    paired=[]
    key=lambda r:(r['route'],r['alpha'],r['event'])
    for candidate,baseline in [('H75','zero'),('H75','equilibrium'),('current','H75'),('current','zero'),('current','equilibrium')]:
        candidates={key(r):r for r in rows if r['mode']==candidate};baselines={key(r):r for r in rows if r['mode']==baseline}
        details=[]
        for event,c in candidates.items():
            b=baselines[event];ch=c['horizons']['20'];bh=b['horizons']['20']
            known=ch['goal_arrived'] is not None and bh['goal_arrived'] is not None and 'unknown' not in (ch['physical_outcome'],bh['physical_outcome'])
            details.append(dict(key=event,force_N=c['force_N'],candidate=c['case_id'],baseline=b['case_id'],
                arrival_change=int(ch['goal_arrived'])-int(bh['goal_arrived']) if known else None,
                survival_change=int(ch['physical_outcome']=='survived')-int(bh['physical_outcome']=='survived') if known else None,
                common_success=bool(ch['goal_arrived'] and bh['goal_arrived'] and ch['physical_outcome']==bh['physical_outcome']=='survived'),
                prefix_delta={k:c['prefix'][k]-b['prefix'][k] for k in ('route_sample_distance_rmse_m','speed_rmse_m_s','path_length_m','peak_roll_rad')}
                    if c['prefix']['ticks'] and b['prefix']['ticks'] else None))
        paired.append(dict(candidate=candidate,baseline=baseline,details=details))
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),
        all_cases=80,actual_lower_ticks=ticks,allocated_lower_slots=status['allocated_lower_slots'],optimizer_updates=0,
        new_physics_steps=0,checked='All raw SHA, actual task-clock prefix, original goals, user speed, world pulse and correction chain; zero/masked targets',
        initial_audit_source_sha256=digest(helper),initial_carry_sha256=initial_review['initial_carry_sha256'],
        limitations='Shared-ancestor fixed DEV, no independent training repeat. Prefix includes first-arrival tick but excludes first contact/fault tick; prefix peakroll is not full contact-process peakroll. Prefix lengths differ by first arrival. Original-route deviation uses nearest samples on goal segment, not exact native V3 geometry projection. Current-path scored cost after replan is diagnostic.'))
    render(root,rows,raws,paths,goals,manifest['reward_components'])
    lines=['# 新情境80条：原目标到达与真实触地','',
       '四法共用新20秒/contact-only/因果一次重规划合同。每法20（4无扰、16受扰），相关DEV，不是独立训练重复。',
       '', '|方法|20s到达/20|整窗不触地/20|触地|未知|', '|---|---:|---:|---:|---:|']
    for mode in MODES:
        a=summary['20'][mode]['all'];lines.append(f'|{mode}|{a["arrived"]}|{a["survived"]}|{a["contact"]}|{a["unknown"]}|')
    lines+=['', '首达后继续运动的末点不是没到过目标。先看task_xy，再看full_xy；全部条件/所有增失见results.json、paired.json。',
       '不以当前路径V3代价替代原目标、原用户速度或原路线偏离。未完成轨迹的短前缀不能当成更便宜的完成任务。',
       '不同策略在起扰前已经形成不同状态；本组不能单独识别进入状态与受扰后动作的因果贡献。',
       '图无平滑/虚构插值；XY等比例，原路径到goal段固定，三角为首达，x为真实触地，方形为完整窗末点。',
       '过程图的reward和cost components均为当前路径诊断，参考更换会改变其路径定义；不据此宣称原用户效用排序。',
       'scientific-visualization用于内部证据图检查，沿论文既有Scientific Agent Skills引用；未核具体期刊图规或人类作者批准。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


def render(root,rows,raws,paths,goals,keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=root/'analysis';events=list(dict.fromkeys(r['event'] for r in rows));routes=sorted(paths)
    for event,task in itertools.product(events,(True,False)):
        fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
        for ax,(route,alpha) in zip(axes.flat,itertools.product(routes,(0.,1.))):
            path=paths[route];goal=goals[route]
            ax.plot(*path['xy'][path['s']<=path['goal']].T,color='black',lw=1,label='Original route')
            ax.add_patch(Circle(goal,1,fill=False,ec='black',lw=.8));ax.scatter(*goal,c='black',marker='*',s=45)
            for mode,color,style in zip(MODES,COLORS,STYLES):
                row=next(r for r in rows if (r['route'],r['alpha'],r['event'],r['mode'])==(route,alpha,event,mode))
                raw=raws[row['case_id']]
                n=row['prefix']['ticks'] if task and row['horizons']['20']['goal_arrived'] else row['active_lower_ticks']
                xy=raw['actual_xy'][:n]
                if not n:continue
                ax.plot(*xy.T,c=color,ls=style,lw=1.4,label=mode)
                end=row['horizons']['20'];marker='x' if end['physical_outcome']=='contact' and not (task and end['goal_arrived']) else 's'
                ax.scatter(*xy[-1],c=color,marker=marker,s=25)
                if end['first_arrival_s'] is not None:
                    i=round(end['first_arrival_s']/.005)-1
                    if i<n:ax.scatter(*raw['actual_xy'][i],c=color,marker='^',s=45)
            ax.set(title=f'{route} / user alpha={alpha:g}',xlabel='World X (m)',ylabel='World Y (m)')
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15);ax.legend(fontsize=8)
        fig.suptitle(f'{event} / '+('prefix to first arrival or failure' if task else 'whole20s window, contact stops'))
        fig.savefig(out/f'{"task" if task else "full"}_xy_{event}.png',dpi=150);plt.close(fig)
    diag=out/'diagnostics';diag.mkdir()
    for route,alpha,event in itertools.product(routes,(0.,1.),events):
        fig,axes=plt.subplots(6,1,figsize=(12,15),sharex=True,layout='constrained')
        comp,caxes=plt.subplots(4,1,figsize=(12,12),sharex=True,layout='constrained')
        for ax,mode,color,style in zip(caxes,MODES,COLORS,STYLES):
            row=next(r for r in rows if (r['route'],r['alpha'],r['event'],r['mode'])==(route,alpha,event,mode))
            raw=raws[row['case_id']];n=row['active_lower_ticks'];t=raw['time'][:n]+.005
            for target,data in [(axes[0],raw['actual_forward_speed']),(axes[1],raw['actual_delta']),
                (axes[2],raw['phi']),(axes[3],raw['phi_dot']),(axes[4],raw['scored_tick_reward']),
                (axes[5],np.cumsum(raw['scored_tick_reward']))]:target.plot(t,data[:n],c=color,ls=style,lw=1,label=mode)
            axes[0].plot(t,raw['wheel_speed_proxy'][:n],c=color,ls=':',lw=.6)
            axes[0].plot(t,raw['governed'][:n,0],c=color,ls='--',lw=.6,alpha=.6)
            axes[1].plot(t,raw['governed'][:n,1],c=color,ls=':',lw=.6)
            for k,name in enumerate(keys):ax.plot(t,raw['effective_costs'][:n,k],lw=.8,label=name)
            ax.set_ylabel(mode+' cost rate');ax.legend(ncol=5,fontsize=7);ax.grid(alpha=.15)
            if row['force_N']:
                for target in [*axes,ax]:target.axvspan(row['onset_s'],row['onset_s']+row['duration_s'],color='gray',alpha=.12)
        axes[0].axhline(2.2,color='black',lw=.7,label='Original user speed')
        for ax,label in zip(axes,['Speed (m/s)','Steer (rad)','Roll (rad)','Roll rate (rad/s)','Diagnostic tick reward','Diagnostic cumulative reward']):
            ax.set_ylabel(label);ax.grid(alpha=.15);ax.legend(fontsize=7,ncol=5)
        axes[-1].set_xlabel('Task post-tick time (s)');caxes[-1].set_xlabel('Task post-tick time (s)')
        title=f'{route} / original alpha={alpha:g} / {event}'
        fig.suptitle(title+'\nSpeed: actual=mode style, wheel proxy=dotted, governed reference=thin dashed; steer reference=dotted')
        comp.suptitle(title+' / current-path diagnostic components, not fixed original-route utility')
        name=f'{route}_a{alpha:g}_{event}'
        fig.savefig(diag/(name+'.png'),dpi=130);comp.savefig(diag/(name+'_components.png'),dpi=130)
        plt.close(fig);plt.close(comp)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
