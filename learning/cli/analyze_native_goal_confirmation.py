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
    p=json.loads((root/'plan.json').read_text())
    if p.get('mode')=='h75_channel_response':return analyze_response(root,p)
    status=json.loads((run/'status.json').read_text())
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


def response_method(case):
    pair=(float(case['speed_alpha']),float(case['steer_alpha']))
    names={(.5,.5):'fixed0.5',(1.,1.):'fixed1',(1.,.5):'speed1_steer0.5'}
    if pair not in names or case['mode']!=('hybrid' if pair==(1.,.5) else 'H75'):
        raise ValueError('undeclared response mapping')
    return names[pair]


def validate_response_inventory(inventory,protocol):
    """Validate the frozen Cartesian protocol without importing runner code."""
    actual=[]
    for c in inventory:
        response_method(c)
        if c['alpha']!=c['speed_alpha']:raise ValueError('state alpha identity differs from declared speed query')
        actual.append((c['route'],float(c['speed']),tuple((float(c['speed_alpha']),float(c['steer_alpha']))),
                       float(c['force_N']),float(c['onset_s']),float(c['duration_s'])))
    events=[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))
    expected=[(r['route'],float(r['speed']),tuple(pair),float(force),float(onset),float(protocol['duration_s']))
              for r,pair,(force,onset) in itertools.product(protocol['routes'],protocol['mappings'],events)]
    if sorted(actual)!=sorted(expected) or len(set(actual))!=len(actual):raise ValueError('response inventory differs from declared protocol')


def audit_response_channels(raw,case):
    active=np.asarray(raw['active_tick'],bool);n=int(active.sum())
    pair=(float(case['speed_alpha']),float(case['steer_alpha']))
    response_method(case)
    for field,width in (('queried_latent_alpha_half',2),('queried_latent_alpha_one',2),
                        ('composed_latent',2),('channel_alpha_pair',2),('latent_z',2),('query_observation',351)):
        values=np.asarray(raw[field])[active]
        if values.shape!=(n,width) or not np.isfinite(values).all():raise ValueError('invalid channel log '+field)
    np.testing.assert_array_equal(raw['channel_alpha_pair'][active],np.tile(pair,(n,1)))
    np.testing.assert_allclose(raw['query_observation'][active,336],case['alpha'],atol=1e-6,rtol=0)
    queries={.5:raw['queried_latent_alpha_half'],1.:raw['queried_latent_alpha_one']}
    composed=np.column_stack([queries[pair[0]][:,0],queries[pair[1]][:,1]])
    np.testing.assert_allclose(raw['composed_latent'][active],composed[active],atol=1e-6,rtol=0)
    permitted=active&np.asarray(raw['reference_target_permitted'],bool)
    np.testing.assert_allclose(raw['latent_z'][permitted],composed[permitted],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(raw['latent_z'][active&~permitted],0)
    # Each query and mask belongs to one pre-upper state, held over four lower ticks.
    for start in range(0,n,4):
        end=min(start+4,n)
        for field in ('queried_latent_alpha_half','queried_latent_alpha_one','composed_latent',
                      'query_observation','pre_upper_yaw','reference_target_permitted'):
            values=np.asarray(raw[field])[start:end]
            np.testing.assert_array_equal(values,np.broadcast_to(values[0],values.shape))
        if 'clean_blocks_at_publication' in raw:
            clean=int(raw['clean_blocks_at_publication'][start])
            if not 0<=clean<=16:raise ValueError('invalid clean-block counter')
            previous_clean=0 if start==0 else int(raw['clean_blocks_at_end'][start-1])
            if clean!=previous_clean:raise ValueError('clean-block publication history changed')
            previous_exit=False if start==0 else bool(raw['cumulative_domain_exit'][start-1])
            if previous_exit and clean<16 and np.any(permitted[start:end]):
                raise ValueError('publication before sixteen complete clean blocks')
    return dict(actual_permitted_ticks=int(permitted.sum()),masked_ticks=int((active&~permitted).sum()),
                speed_alpha=pair[0],steer_alpha=pair[1],state_alpha=case['alpha'],
                checked='Same pre-upper observation except queried alpha slot; composed target published only under actual permission mask')


def onset_frame_impulse(raw,case,pulse):
    """World impulse projected once into the logged ONSET upper-frame yaw."""
    active=np.asarray(raw['active_tick'],bool);onset=round(case['onset_s']/.005)
    actual=np.asarray(pulse['actual_world_impulse'],float)
    full=np.array([0.,case['force_N']*case['duration_s'],0.])
    exposed=np.flatnonzero(active&(np.linalg.norm(raw['applied_force_impulse'],axis=1)>1e-10))
    observed=bool(onset<len(active) and active[onset])
    yaw=float(raw['pre_upper_yaw'][onset]) if observed else None
    lateral=None if yaw is None else float(-np.sin(yaw)*actual[0]+np.cos(yaw)*actual[1])
    if not case['force_N']:exposure='nominal_no_force'
    elif np.allclose(actual,full,atol=max(1e-5,.01*abs(full[1])),rtol=0):exposure='full_event_dose'
    else:exposure='truncated_exposure'
    return dict(onset_pre_upper_yaw_rad=yaw,onset_observed=observed,
        signed_body_lateral_impulse_onset_frame_N_s=lateral,exposure=exposure,
        intended_world_impulse_N_s=full.tolist(),actual_world_impulse_N_s=actual.tolist(),
        actual_exposed_ticks=len(exposed),frame='worldY fixed force; lateral=-sin(yaw)*Jx+cos(yaw)*Jy',
        limitation='Onset pre-upper yaw approximation, not an exact body-frame substep impulse integral.')


def reconstruct_execution_path_reward(raw,case,original,replanned,replan_tick,config,controller,keys):
    """Rebuild native V3 task channels on the path actually active at each tick."""
    from analyze_native_frozen_validation import score
    n=int(np.asarray(raw['active_tick'],bool).sum());effective=np.asarray(raw['effective_costs'][:n],float).copy();user1=effective.copy()
    cuts=[(0,n,original)] if replanned is None else [(0,replan_tick,original),(replan_tick,n,replanned)]
    for start,end,path in cuts:
        if start>=end:continue
        segment={k:np.asarray(v)[start:end] for k,v in raw.items()}
        # Domain telemetry is not fatal in this new contact-only contract.
        segment['physical_failure']=np.zeros(end-start,bool);segment['domain_exit']=np.zeros(end-start,bool)
        _,data=score(segment,path,config,controller,keys,float(case['alpha']))
        np.testing.assert_allclose(effective[start:end],data['effective'],atol=4e-4,rtol=2e-5,
            err_msg='execution-path V3 reward components differ')
        effective[start:end]=data['effective']
        _,one=score(segment,path,config,controller,keys,1.)
        user1[start:end]=one['effective']
    running=-effective.sum(axis=1)*config['path_reward']['scale']*.005
    reward=running.copy();fatal=np.flatnonzero(np.asarray(raw['physical_failure'][:n],bool))
    if len(fatal):
        end=int(fatal[0]);block=end//4*4;remaining=round(10/.02)-end//4
        c=config['path_reward'];gamma=config['training_future_phase_C']['gamma']
        penalty=-c['failure_extra']-c['scale']*.02*c['failure_cost_rate_bound']*(1-gamma**remaining)/(1-gamma)
        reward[block:]=0.;reward[end]=penalty
    np.testing.assert_allclose(reward,raw['scored_tick_reward'][:n],atol=4e-5,rtol=2e-5,
        err_msg='execution-path per-step reward differs')
    user1_running=-user1.sum(axis=1)*config['path_reward']['scale']*.005
    user1_reward=user1_running.copy()
    if len(fatal):user1_reward[block:]=0.;user1_reward[end]=penalty
    return dict(reward=reward,running_components=-effective*config['path_reward']['scale']*.005,
                terminal_adjustment=reward-running,user1_reward=user1_reward,
                user1_running_components=-user1*config['path_reward']['scale']*.005,
                user1_terminal_adjustment=user1_reward-user1_running)


def response_summary(rows):
    methods=('fixed0.5','fixed1','speed1_steer0.5');summary={};paired=[]
    for method in methods:
        summary[method]={}
        for subset in ('all','nominal','disturbed'):
            group=[r for r in rows if r['method']==method and (subset=='all' or bool(r['force_N'])==(subset=='disturbed'))]
            h=[r['horizons']['10'] for r in group]
            summary[method][subset]=dict(n=len(h),arrived=sum(x['goal_arrived'] is True for x in h),
                survived=sum(x['physical_outcome']=='survived' for x in h),contact=sum(x['physical_outcome']=='contact' for x in h),
                unknown=sum(x['goal_arrived'] is None or x['physical_outcome']=='unknown' for x in h),
                arrived_and_survived=sum(x['goal_arrived'] is True and x['physical_outcome']=='survived' for x in h),
                nominal_replans=sum(bool(r['replanned']) for r in group if not r['force_N']),
                first_arrival_before_force=sum(r['pulse_started_after_first_arrival'] is True for r in group),
                truncated_exposure=sum(r['onset_frame']['exposure']=='truncated_exposure' for r in group))
    for candidate,baseline in itertools.combinations(methods,2):
        a={(r['route'],r['event']):r for r in rows if r['method']==candidate}
        b={(r['route'],r['event']):r for r in rows if r['method']==baseline}
        if set(a)!=set(b):raise ValueError('incomplete cross-method route/event pairs')
        details=[]
        for key,c in a.items():
            d=b[key];ch=c['horizons']['10'];bh=d['horizons']['10']
            known=ch['goal_arrived'] is not None and bh['goal_arrived'] is not None and 'unknown' not in (ch['physical_outcome'],bh['physical_outcome'])
            details.append(dict(route=key[0],event=key[1],candidate=c['case_id'],baseline=d['case_id'],
                candidate_state_alpha=c['alpha'],baseline_state_alpha=d['alpha'],
                arrival_change=int(ch['goal_arrived'])-int(bh['goal_arrived']) if known else None,
                survival_change=int(ch['physical_outcome']=='survived')-int(bh['physical_outcome']=='survived') if known else None,
                candidate_onset_frame=c['onset_frame'],baseline_onset_frame=d['onset_frame'],
                diagnostic_execution_path_user1_cost_difference=c['diagnostic_execution_path_user1_cost']-d['diagnostic_execution_path_user1_cost'],
                original_route_sample_distance_rmse_difference_m=c['original_route_sample_distance_rmse_m']-d['original_route_sample_distance_rmse_m'],
                speed_rmse_difference_m_s=c['full_observed_speed_rmse_m_s']-d['full_observed_speed_rmse_m_s']))
        paired.append(dict(candidate=candidate,baseline=baseline,details=details))
    return summary,paired


def analyze_response(root,p):
    run=root/'evaluation';out=root/'analysis';status=json.loads((run/'status.json').read_text())
    inventory=json.loads((root/'inventory.json').read_text());index=json.loads((run/'case_index.json').read_text())
    if p['horizon_s']!=10. or p['cases']!=54 or len(inventory)!=54 or len(index)!=54 or status['phase']!='completed' or status['completed_cases']!=54:
        raise ValueError('complete declared54 H10 cases required')
    if len({c['case_id'] for c in index})!=54:raise ValueError('duplicate case identity')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('plan drift')
    for file,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(file)!=sha:raise ValueError('executed source/input drift '+file)
    manifest=json.loads((run/'manifest.json').read_text())
    for key,expected in (('plan_sha256',digest(root/'plan.json')),('config',p['config']),('models',p['models']),('cases',inventory),('source',p['initial_source'])):
        if manifest[key]!=expected:raise ValueError('manifest identity mismatch '+key)
    # Exactly three methods per route/event; nominal lanes are new physical trajectories.
    validate_response_inventory(inventory,p['protocol'])
    expected={(r['route'],r['event'],response_method(r)) for r in inventory}
    if len(expected)!=54 or len({r['route'] for r in inventory})!=2 or sum(not r['force_N'] for r in inventory)!=6:
        raise ValueError('declared response inventory invalid')
    out.mkdir(exist_ok=False);helper=Path(__file__).with_name('audit_native_goal_initial.py')
    subprocess.run([sys.executable,'-B',str(helper),str(root),str(out/'initial_carry_review.json')],check=True)
    initial=json.loads((out/'initial_carry_review.json').read_text())
    paths={};goals={};trees={};raws={};rows=[];ticks=0;rebuilt={};executed={}
    decisions=json.loads((run/'replanning_decisions.json').read_text());by_case={}
    for decision in decisions:
        case=decision['case_id']
        if case in by_case:raise ValueError('more than one replan')
        tick=decision['task_tick']
        if tick not in range(200,1801,200):raise ValueError('replan outside declared1..9s boundaries')
        by_case[case]=decision
    for route in {r['route'] for r in inventory}:
        with np.load(run/(route+'_path.npz')) as z:paths[route]=dict(z)
        path=paths[route];goals[route]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,a]) for a in (0,1)])
        with np.load(Path(p['cached_source'])/'evaluation'/(route+'_path.npz')) as z:
            for field in ('s','xy','heading','curvature'):
                np.testing.assert_array_equal(path[field][:len(z[field])],z[field],err_msg='original path prefix changed')
            for field in ('goal','turn_end'):np.testing.assert_array_equal(path[field],z[field])
        trees[route]=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[route]]))
    for i,c in enumerate(index):
        if any(c.get(k)!=v for k,v in inventory[i].items()):raise ValueError('case identity drift')
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('raw SHA drift')
        goal=goals[c['route']]
        np.testing.assert_allclose(c['original_goal_xy'],goal,atol=3e-6,rtol=0)
        np.testing.assert_allclose(manifest['original_goals_xy'][i],goal,atol=3e-6,rtol=0)
        with np.load(c['raw_file']) as z:raw=dict(z)
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum());ticks+=n
        if not 0<n<=2000 or not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('invalid actual clock prefix')
        clock=np.rint(raw['time']/.005).astype(np.int64)
        np.testing.assert_array_equal(clock[active],np.arange(n));np.testing.assert_allclose(raw['time'][active],np.arange(n)*.005,atol=3e-6,rtol=0)
        np.testing.assert_allclose(raw['v_user'][active],c['speed'],atol=2e-7,rtol=0)
        data=dict(raw,physical_tick=clock,policy_fault=raw['policy_fault']|raw['continuation_geometry_fault'])
        pulse=audit_pulse(data,dict(force_N=c['force_N'],world_direction=[0.,1.,0.],pulse_onset_task_tick=round(c['onset_s']/.005),pulse_duration_s=c['duration_s']))
        channels=audit_response_channels(raw,c)
        chain=verify_correction_chain(dict(data,issued_latent=raw['latent_z'],nominal_reference=raw['nominal'],reference=raw['governed'],correction_offset=raw['offsets']),p['config'],initial_filtered=initial['initial_filtered'][i],initial_offsets=initial['initial_offsets'][i])
        cut={k:v[:2000] for k,v in data.items()}
        endpoint=goal_arrival_metrics(cut,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=10.)
        endpoint['physical_outcome']=physical_outcome(cut['active_tick'],cut['contact_failure'],cut['policy_fault']|cut['nonfinite']|cut['lower_fault'],2000)
        fault=data['policy_fault']|raw['nonfinite']|raw['lower_fault'];fatal=np.flatnonzero(active&(raw['contact_failure']|fault))
        stop=int(fatal[0]) if len(fatal) else n;arrival=endpoint['first_arrival_s'];task_stop=min(stop,round(arrival/.005)) if arrival is not None else stop
        path=paths[c['route']];distance=trees[c['route']].query(raw['actual_xy'][:n])[0]
        replan=run/f'replanned_path_{i:02d}.npz';decision=by_case.get(c['case_id']);changed=None;replan_tick=n
        if replan.exists()!=bool(c['replanned']) or bool(decision)!=bool(c['replanned']):raise ValueError('replan evidence mismatch')
        if decision:
            if decision['case_index']!=i:raise ValueError('replan case index mismatch')
            replan_tick=decision['task_tick']
            from sttw_control.goal_replanning import replan_requested
            if not replan_requested(xy=np.asarray(decision['measured_pose'])[:2],goal_xy=goal,
                cross_track_m=decision['cross_track_m'],follower_invalid=decision['follower_invalid'],
                already_replanned=False,contact=decision['contact'],fault=decision['fault'],ever_arrived=decision['ever_arrived']):
                raise ValueError('replan lacks declared measured trigger')
            np.testing.assert_allclose(decision['measured_pose'][:2],raw['actual_xy'][replan_tick-1],atol=3e-6,rtol=0)
            np.testing.assert_allclose(decision['cross_track_m'],raw['path_cross_track'][replan_tick-1],atol=3e-6,rtol=0)
            with np.load(replan) as z:changed=dict(z)
            np.testing.assert_allclose([np.interp(float(changed['goal']),changed['s'],changed['xy'][:,a]) for a in (0,1)],goal,atol=3e-6,rtol=0)
            executed[c['case_id']]=changed
        rebuilt[c['case_id']]=reconstruct_execution_path_reward(raw,c,path,changed,replan_tick,p['config'],p['initial_source']['controller'],manifest['reward_components'])
        frame=onset_frame_impulse(raw,c,pulse)
        rows.append(dict(c,method=response_method(c),horizons={'10':endpoint},active_lower_ticks=n,pulse=pulse,onset_frame=frame,correction_chain=chain,channels=channels,
            prefix=dict(ticks=task_stop,observed_s=task_stop*.005),
            pulse_started_after_first_arrival=bool(arrival is not None and arrival<=c['onset_s']) if c['force_N'] else None,
            original_route_sample_distance_rmse_m=float(np.sqrt(np.mean(distance**2))),
            original_route_sample_distance_max_m=float(distance.max()),
            full_observed_speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed'][:n]-c['speed'])**2))),
            full_observed_mean_speed_m_s=float(np.mean(raw['actual_forward_speed'][:n])),
            domain_exit_telemetry_ticks=int(np.sum(active&raw['domain_exit'])),
            runtime_own_alpha_execution_path_cost=float(-raw['scored_tick_reward'][active].sum()),
            diagnostic_execution_path_user1_cost=float(-rebuilt[c['case_id']]['user1_reward'].sum()),
            replan_task_tick=None if decision is None else replan_tick,
            planning_at_4s_before_pulse_step=bool(decision and replan_tick==800 and c['onset_s']==4.),
            full_observed_peak_roll_rad=float(raw['peak_roll'][:n].max()),actual_final_xy=raw['actual_xy'][n-1].tolist()))
        raws[c['case_id']]=raw
        # Raw per-case frames retained beside rebuilt rewards and descriptive original distance.
        numeric={k:np.asarray(v)[:n] for k,v in raw.items()}
        numeric.update(original_route_sample_distance_m=distance,reconstructed_runtime_own_alpha_reward=rebuilt[c['case_id']]['reward'],
                       diagnostic_execution_path_user1_reward=rebuilt[c['case_id']]['user1_reward'],
                       diagnostic_execution_path_user1_cumulative_reward=np.cumsum(rebuilt[c['case_id']]['user1_reward']),
                       diagnostic_execution_path_user1_components=rebuilt[c['case_id']]['user1_running_components'],
                       diagnostic_execution_path_user1_cumulative_components=np.cumsum(rebuilt[c['case_id']]['user1_running_components'],axis=0),
                       reconstructed_reward_components=rebuilt[c['case_id']]['running_components'],cumulative_reward_components=np.cumsum(rebuilt[c['case_id']]['running_components'],axis=0),terminal_adjustment=rebuilt[c['case_id']]['terminal_adjustment'])
        np.savez_compressed(out/f'case_{i:02d}_diagnostics.npz',**numeric)
    if ticks!=status['active_lower_ticks']:raise ValueError('raw tick charge mismatch')
    summary,paired=response_summary(rows)
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(all_cases=54,horizons_s=[10],actual_lower_ticks=ticks,new_physics_steps=0,optimizer_updates=0,
        plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),initial_carry_sha256=initial['initial_carry_sha256'],
        checked='Raw SHA, full initial audit, config/model/goal/inventory, actual pulse/clock, masked channel composition, correction chain and reconstructed execution-path V3 channels/rewards.',
        limitation='Non-task reward components reused from authenticated raw logs, not independently rederived physics. Original-path distances are descriptive sampled nearest distances; execution-path V3 is not original-user utility after replan.'))
    render_response(root,rows,raws,paths,goals,executed,rebuilt,manifest['reward_components'])
    lines=['# H75通道受扰响应：54条H10','','原始XY与执行路径优先；两tight S×三通道映射×九事件。六条无扰轨迹为本合同重新执行，不复用旧原生停止合同。',
        '','主终点：原世界目标1m首达；整10s真实触地单列。早期domain仅遥测，不算触地；重规划后不声称严格完成原S。',
        '','相关熟悉DEV、同祖先；无新训练。各法在起扰前已有不同状态，不能把转向通道替换解释为独立于速度的因果效应。']
    for event in dict.fromkeys(r['event'] for r in rows):
        lines+=['',f'## {event}','',f'![Original and executed XY](full_xy_{event}.png)','',f'![First-arrival prefixes](task_xy_{event}.png)']
    lines+=['','执行路径V3图以相同user1重评分；运行本身alpha评分另行核验并保留。逐步/累计奖励和分项仅为诊断，不是重规划后原用户路径效用。原路线距离与执行路径误差分别保存；未完成短窗不能当更低完成代价。',
        '世界Y脉冲投影使用日志起扰前upper yaw作一次onset-frame近似，不是逐物理子步积分。完整剂量/截断暴露、首达早于起扰、名义重规划逐例列出。4s若重规划，先规划后执行含脉冲的步。',
        '[逐例结果](results.json) · [无扰/受扰汇总](summary.json) · [全部配对得失](paired.json) · [核验](receipt.json)',
        '全部54逐步原始帧、实际末点、重建奖励/累计分项、真速度/轮速代理与误差保留于case诊断NPZ；图见diagnostics。无20s数据或结论。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def render_response(root,rows,raws,paths,goals,executed,rebuilt,keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=root/'analysis';diag=out/'diagnostics';diag.mkdir();routes=sorted(paths)
    methods=('fixed0.5','fixed1','speed1_steer0.5');colors=('#0072B2','#D55E00','#008060')
    for event in dict.fromkeys(r['event'] for r in rows):
        for task in (False,True):
            fig,axes=plt.subplots(1,len(routes),figsize=(13,6),squeeze=False,layout='constrained')
            for ax,route in zip(axes.flat,routes):
                path=paths[route];goal=goals[route];ax.plot(*path['xy'][path['s']<=path['goal']].T,'k-',lw=1,label='Original S to goal')
                ax.add_patch(Circle(goal,1,fill=False,ec='black'));ax.scatter(*goal,c='black',marker='*')
                for method,color in zip(methods,colors):
                    r=next(r for r in rows if (r['route'],r['event'],r['method'])==(route,event,method));raw=raws[r['case_id']]
                    h=r['horizons']['10'];n=r['prefix']['ticks'] if task and h['goal_arrived'] else r['active_lower_ticks'];xy=raw['actual_xy'][:n]
                    if r['case_id'] in executed:
                        path2=executed[r['case_id']];ax.plot(*path2['xy'][path2['s']<=path2['goal']].T,c=color,ls='--',lw=.8,alpha=.65,label=method+' executed replan')
                    if n:
                        ax.plot(*xy.T,c=color,label=method);marker='^' if task and h['goal_arrived'] else 'x' if h['physical_outcome']=='contact' else 's' if h['physical_outcome']=='survived' else 'D'
                        ax.scatter(*xy[-1],c=color,marker=marker,s=30)
                    if h['first_arrival_s'] is not None:
                        k=round(h['first_arrival_s']/.005)-1
                        if k<n:ax.scatter(*raw['actual_xy'][k],c=color,marker='^')
                    if r['force_N']:
                        dose=np.flatnonzero(np.linalg.norm(raw['applied_force_impulse'][:n],axis=1)>1e-10)
                        if len(dose):ax.scatter(*raw['actual_xy'][dose].T,c=color,s=8,alpha=.5)
                ax.set(title=route,xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6);ax.grid(alpha=.15)
            fig.suptitle(event+' / '+('first-arrival/contact prefix' if task else 'full actual H10 endpoint')+'; dashed=executed replan; dots=actual worldY exposure')
            fig.savefig(out/f'{"task" if task else "full"}_xy_{event}.png',dpi=140);plt.close(fig)
        for route in routes:
            selected=[r for r in rows if r['route']==route and r['event']==event]
            path=paths[route];original_tree=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[route]]))
            fig,axes=plt.subplots(13,1,figsize=(12,28),sharex=True,layout='constrained')
            comp,cax=plt.subplots(len(keys)+1,2,figsize=(14,2.3*(len(keys)+1)),sharex=True,squeeze=False,layout='constrained')
            for method,color in zip(methods,colors):
                r=next(r for r in selected if r['method']==method);raw=raws[r['case_id']];n=r['active_lower_ticks'];t=raw['time'][:n]+.005;data=rebuilt[r['case_id']]
                original=np.linalg.norm(raw['actual_xy'][:n]-goals[route],axis=1)
                values=(raw['actual_forward_speed'][:n],raw['actual_forward_speed'][:n]-raw['wheel_speed_proxy'][:n],raw['actual_forward_speed'][:n]-raw['v_user'][:n],raw['path_cross_track'][:n],raw['path_heading_error'][:n],original,
                        raw['phi'][:n],raw['phi_dot'][:n],raw['actual_delta'][:n],raw['applied_force_impulse'][:n,1]/.005,
                        original_tree.query(raw['actual_xy'][:n])[0],data['user1_reward'],np.cumsum(data['user1_reward']))
                for ax,v in zip(axes,values):ax.plot(t,v,c=color,label=method,lw=.8);ax.scatter(t[-1],v[-1],c=color,marker='x' if r['horizons']['10']['physical_outcome']=='contact' else 's',s=12)
                axes[0].plot(t,raw['wheel_speed_proxy'][:n],c=color,ls=':',lw=.7);axes[0].plot(t,raw['v_user'][:n],c=color,ls='--',lw=.5)
                components=np.column_stack([data['user1_running_components'],data['user1_terminal_adjustment']])
                for k,name in enumerate(list(keys)+['terminal replacement']):
                    cax[k,0].plot(t,components[:,k],c=color,label=method,lw=.8);cax[k,1].plot(t,np.cumsum(components[:,k]),c=color,lw=.8)
                if r['force_N']:
                    for ax in [*axes,*cax.flat]:ax.axvspan(r['onset_s'],r['onset_s']+r['duration_s'],color='gray',alpha=.07)
                if r['replan_task_tick'] is not None:
                    for ax in [*axes,*cax.flat]:ax.axvline(r['replan_task_tick']*.005,c=color,ls=':',lw=.6)
            for ax,label in zip(axes,('Speed: true/dotted wheel/dashed user (m/s)','True - wheel proxy (m/s)','True - original request (m/s)','Execution-path cross-track (m)','Execution-path heading error (rad)','Original goal distance (m)','Roll (rad)','Roll rate (rad/s)','Steer (rad)','Actual worldY mean force/tick (N)','Original route sample distance (m)','Diagnostic user1 reward / actual step','Diagnostic user1 cumulative reward')):
                ax.set_ylabel(label);ax.grid(alpha=.15);ax.legend(fontsize=6)
            for k,name in enumerate(list(keys)+['terminal replacement']):cax[k,0].set_ylabel(name);cax[k,1].set_ylabel('Cumulative '+name)
            cax[0,0].legend(fontsize=6)
            for ax in [axes[-1],*cax[-1]]:ax.set_xlabel('Actual post-tick time (s)')
            title=route+' / '+event+' / H10 actual endpoints; vertical lines=replan, gray=intended pulse; no new training'
            fig.suptitle(title+'\nExecution-path errors/rewards are diagnostic; original goal unchanged')
            comp.suptitle(title+'\nSigned per-step and cumulative user1 execution-path reward components; no original-user utility ranking')
            fig.savefig(diag/f'{route}_{event}.png',dpi=125);comp.savefig(diag/f'{route}_{event}_components.png',dpi=125);plt.close(fig);plt.close(comp)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
