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


def analyze(root,include_heldout_labels=False,include_confirmation_labels=False):
    root=root.resolve();run=root/'evaluation';out=root/'analysis'
    p=json.loads((root/'plan.json').read_text())
    if p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192'):return analyze_bank144(root,p,include_heldout_labels=include_heldout_labels,include_confirmation_labels=include_confirmation_labels)
    if p.get('mode')=='h75_samestate_tolerance72':return analyze_samestate(root,p)
    if p.get('mode') in ('h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100','tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):return analyze_response(root,p)
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


def analysis_path_key(case):
    return case.get('path_key',case['route'])


def response_method(case):
    if case.get('effective_user_alpha') is not None:return case.get('method',case['mode'])
    if case.get('path_key'):
        return {'fixed_half':'fixed0.5'}.get(case['mode'],case['mode'])
    if case['mode']=='online_scalar':return 'online_scalar'
    if case['mode'].startswith('switch_'):
        if case['origin'] not in ('fixed1','hybrid') or case['mode']!='switch_'+case['origin'] or case['alpha']!=1.:raise ValueError('invalid switching origin')
        return f"switch_{case['origin']}@{case['switch_s']:g}s"
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
    if case['mode']=='online_scalar' or 'path_key' in case:return audit_online_publication(raw,case)
    active=np.asarray(raw['active_tick'],bool);n=int(active.sum())
    pair=(float(case['speed_alpha']),float(case['steer_alpha']))
    response_method(case)
    for field,width in (('queried_latent_alpha_half',2),('queried_latent_alpha_one',2),
                        ('composed_latent',2),('channel_alpha_pair',2),('latent_z',2),('query_observation',351)):
        values=np.asarray(raw[field])[active]
        if values.shape!=(n,width) or not np.isfinite(values).all():raise ValueError('invalid channel log '+field)
    expected_pair=np.tile(pair,(len(active),1))
    if case['mode'].startswith('switch_'):
        tick=round(case['switch_s']/.005)
        if tick%4 or abs(tick*.005-case['switch_s'])>1e-8:raise ValueError('switch not aligned to true upper boundary')
        switched=np.arange(len(active))>=tick
        expected_pair[switched]=[.5,.5]
        np.testing.assert_array_equal(raw['origin_channel_alpha_pair'][active],np.tile(pair,(n,1)))
        np.testing.assert_array_equal(raw['switch_tick'][active],np.full(n,tick))
        np.testing.assert_array_equal(raw['switch_active'][active],switched[active])
    np.testing.assert_array_equal(raw['channel_alpha_pair'][active],expected_pair[active])
    np.testing.assert_allclose(raw['query_observation'][active,336],case['alpha'],atol=1e-6,rtol=0)
    queries={.5:raw['queried_latent_alpha_half'],1.:raw['queried_latent_alpha_one']}
    composed=np.column_stack([np.where(expected_pair[:,0]==.5,queries[.5][:,0],queries[1.][:,0]),
                              np.where(expected_pair[:,1]==.5,queries[.5][:,1],queries[1.][:,1])])
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


def reconstruct_execution_path_reward(raw,case,original,replanned,replan_tick,config,controller,keys,start_tick=0):
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
        _,one=score(segment,path,config,controller,keys,float(case.get('effective_user_alpha',1.)))
        user1[start:end]=one['effective']
    running=-effective.sum(axis=1)*config['path_reward']['scale']*.005
    reward=running.copy();fatal=np.flatnonzero(np.asarray(raw['physical_failure'][:n],bool))
    if len(fatal):
        end=int(fatal[0]);block=end//4*4;remaining=round(10/.02)-(start_tick+end)//4
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


def validate_confirmation_inventory(inventory,p):
    actual=[]
    for c in inventory:
        if c['alpha']!=1. or c['mode'] not in p['arms'] or c['path_key']!=f"bank{c['bank_index']}_"+c['route']:raise ValueError('invalid bank-aware confirmation identity')
        if c['initial_fingerprint']!=p['bank_state_fingerprints'][str(c['bank_index'])]:raise ValueError('case bank fingerprint mismatch')
        expected_query={'fixed0':0.,'fixed_half':.5}.get(c['mode'],1.)
        if c['query_alpha']!=expected_query:raise ValueError('query alpha differs from arm identity')
        actual.append((c['bank_index'],c['route'],c['speed'],c['mode'],c['force_N'],c['onset_s'],c['duration_s']))
    events=[(0.,p['zero_onset_s'])]+list(itertools.product(p['forces_N'],p['onsets_s']))
    expected=[(b,r['route'],r['speed'],arm,f,t,p['duration_s']) for b,r,arm,(f,t) in itertools.product(p['bank_indices'],p['routes'],p['arms'],events)]
    if sorted(actual)!=sorted(expected) or len(set(actual))!=len(actual):raise ValueError('confirmation inventory differs from bank/route/event/arm protocol')


def audit_floor_support(raw,case,config,floor,initial_nominal):
    active=np.asarray(raw['active_tick'],bool);n=int(active.sum());zero=case['mode']=='zero_with_existing_speed_floor'
    scale=config['residual_interface']['speed_positive_scale_m_s'];realized=[]
    for start in range(0,n,4):
        end=min(start+4,n);nominal=np.asarray(raw['support_nominal_reference'][start])
        np.testing.assert_allclose(nominal,np.asarray(initial_nominal) if start==0 else raw['nominal'][start],atol=4e-6,rtol=0)
        for field in ('support_nominal_reference','support_floor','support_requested_offset','support_realized_target_offset','support_active','neural_query_active'):
            a=np.asarray(raw[field])[start:end];np.testing.assert_array_equal(a,np.broadcast_to(a[0],a.shape))
        desired=max(0.,floor-float(nominal[0]));bounded=np.clip(desired/scale,-1.+1e-6,1.-1e-6);target=bounded*scale
        for field,value in (('support_floor',floor),('support_requested_offset',desired),('support_realized_target_offset',target),('support_active',target>0)):
            np.testing.assert_allclose(raw[field][start],value if zero else 0.,atol=3e-6,rtol=1e-5)
        if zero:
            np.testing.assert_allclose(raw['composed_latent'][start],[np.arctanh(bounded),0.],atol=3e-6,rtol=1e-5)
            realized.extend([target]*(end-start))
    return dict(neural_query=not zero,observed_support_active_ticks=int(np.sum(raw['support_active'][active])),
        maximum_realized_target_offset_m_s=float(max(realized,default=0.)),
        equivalent_to_zero_in_observed_run=bool(zero and not np.any(raw['support_active'][active])),
        interpretation='Existing pre-filter nominal speed floor; when inactive this arm is effectively zero neural correction in these conditions, not evidence excluding speed-feedback support.')


def confirmation_summary(rows):
    summaries={};pairs=[]
    for method in ('online_scalar','fixed0','fixed0.5','fixed1','zero_with_existing_speed_floor'):
        summaries[method]={}
        for bank in sorted({r['bank_index'] for r in rows}):
            summaries[method][str(bank)]={}
            for subset in ('all','nominal','disturbed'):
                group=[r for r in rows if r['method']==method and r['bank_index']==bank and (subset=='all' or bool(r['force_N'])==(subset=='disturbed'))]
                summaries[method][str(bank)][subset]=dict(n=len(group),arrived=sum(r['horizons']['10']['goal_arrived'] is True for r in group),
                    contact=sum(r['horizons']['10']['physical_outcome']=='contact' for r in group),unknown=sum(r['horizons']['10']['physical_outcome']=='unknown' or r['horizons']['10']['goal_arrived'] is None for r in group),
                    survived=sum(r['horizons']['10']['physical_outcome']=='survived' for r in group))
    online=[r for r in rows if r['mode']=='online_scalar']
    for method in ('fixed0','fixed0.5','fixed1','zero_with_existing_speed_floor'):
        baseline={(r['bank_index'],r['route'],r['event']):r for r in rows if r['method']==method};details=[]
        for c in online:
            b=baseline[c['bank_index'],c['route'],c['event']];a=c['horizons']['10'];h=b['horizons']['10'];known=a['goal_arrived'] is not None and h['goal_arrived'] is not None and 'unknown' not in (a['physical_outcome'],h['physical_outcome'])
            details.append(dict(bank_index=c['bank_index'],route=c['route'],event=c['event'],force_N=c['force_N'],candidate=c['case_id'],baseline=b['case_id'],
                candidate_outcome=a,baseline_outcome=h,candidate_prefix=c['prefix'],baseline_prefix=b['prefix'],
                contact_rescue=h['physical_outcome']=='contact' and a['physical_outcome']=='survived',new_contact=h['physical_outcome']!='contact' and a['physical_outcome']=='contact',
                new_fault_or_unknown=a['physical_outcome']=='unknown' or a['goal_arrived'] is None,goal1m_gain=known and bool(a['goal_arrived'] and not h['goal_arrived']),goal1m_loss=known and bool(h['goal_arrived'] and not a['goal_arrived']),
                arrival_time_difference_s=a['first_arrival_s']-h['first_arrival_s'] if a['first_arrival_s'] is not None and h['first_arrival_s'] is not None else None,
                speed_rmse_difference_m_s=c['full_observed_speed_rmse_m_s']-b['full_observed_speed_rmse_m_s'],
                diagnostic_execution_path_user1_cost_difference=c['diagnostic_execution_path_user1_cost']-b['diagnostic_execution_path_user1_cost']))
        pairs.append(dict(candidate='online_scalar',baseline=method,details=details,
            nominal_no_new_contact_fault_or_goal_loss=all(not(x['new_contact'] or x['new_fault_or_unknown'] or x['goal1m_loss']) for x in details if not x['force_N'])))
    return dict(method_bank_subset=summaries,new_cases=len(rows),cached_cases=0,
        unreturned_last_activations=[dict(case_id=r['case_id'],alpha_rule=r['alpha_rule']) for r in online if r['alpha_rule']['unreturned_at_actual_endpoint']],
        floor_support=[dict(case_id=r['case_id'],**r['floor_support']) for r in rows if r['mode']=='zero_with_existing_speed_floor'],
        adoption='not_automatically_adopted',data_role='Two prepared bank states, related DEV, not independent training seeds; all100 physics newly executed.'),pairs


def validate_online_inventory(inventory,p):
    actual=[]
    for c in inventory:
        if c['mode']!='online_scalar' or c['alpha']!=1. or c['speed_alpha']!=1. or c['steer_alpha']!=1.:raise ValueError('invalid online user/state identity')
        actual.append((c['route'],c['speed'],c['force_N'],c['onset_s'],c['duration_s']))
    events=[(0.,p['zero_onset_s'])]+list(itertools.product(p['forces_N'],p['onsets_s']))
    expected=[(r['route'],r['speed'],f,t,p['duration_s']) for r,(f,t) in itertools.product(p['routes'],events)]
    if sorted(actual)!=sorted(expected) or len(set(actual))!=len(actual):raise ValueError('online inventory differs from declared protocol')


def audit_online_publication(raw,case):
    active=np.asarray(raw['active_tick'],bool);n=int(active.sum());alpha=np.asarray(raw['alpha_exec'])
    for field in ('alpha_user','alpha_exec','rule_latched','rule_preview_max_curvature','rule_accel_estimate','rule_alpha_target'):
        values=np.asarray(raw[field])[active]
        if values.shape!=(n,) or not np.isfinite(values).all():raise ValueError('invalid online scalar log '+field)
    np.testing.assert_array_equal(raw['alpha_user'][active],np.full(n,case.get('source_user_alpha',1.)))
    if case['mode']=='online_scalar':
        if not np.isin(alpha[active],[.5,1.]).all():raise ValueError('scalar alpha has undeclared slew/intermediate value')
    else:
        np.testing.assert_array_equal(alpha[active],np.full(n,case['query_alpha']))
        np.testing.assert_array_equal(raw['rule_latched'][active],np.zeros(n,bool))
    if 'path_key' in case:
        np.testing.assert_array_equal(raw['neural_query_active'][active],np.full(n,case['mode']!='zero_with_existing_speed_floor'))
    np.testing.assert_array_equal(raw['channel_alpha_pair'][active],np.column_stack([alpha[active],alpha[active]]))
    np.testing.assert_allclose(raw['query_observation'][active,336],alpha[active],atol=1e-6,rtol=0)
    permitted=active&np.asarray(raw['reference_target_permitted'],bool)
    for field,width in (('query_observation',351),('composed_latent',2),('latent_z',2)):
        values=np.asarray(raw[field])[active]
        if values.shape!=(n,width) or not np.isfinite(values).all():raise ValueError('invalid actual scalar query '+field)
    np.testing.assert_allclose(raw['latent_z'][permitted],raw['composed_latent'][permitted],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(raw['latent_z'][active&~permitted],0)
    for start in range(0,n,4):
        end=min(start+4,n)
        for field in ('alpha_exec','rule_latched','rule_preview_max_curvature','rule_accel_estimate','rule_alpha_target','query_observation','composed_latent','reference_target_permitted'):
            values=np.asarray(raw[field])[start:end];np.testing.assert_array_equal(values,np.broadcast_to(values[0],values.shape))
        if 'clean_blocks_at_publication' in raw:
            clean=int(raw['clean_blocks_at_publication'][start]);prior=0 if start==0 else int(raw['clean_blocks_at_end'][start-1])
            if clean!=prior or not 0<=clean<=16:raise ValueError('online clean counter changed')
            if start and raw['cumulative_domain_exit'][start-1] and clean<16 and np.any(permitted[start:end]):raise ValueError('online publication before clean handback')
    return dict(actual_permitted_ticks=int(permitted.sum()),masked_ticks=int((active&~permitted).sum()),state_alpha=1.,
        checked=('No neural query; logged existing speed-floor support latent with actual native publication mask/filter chain' if case['mode']=='zero_with_existing_speed_floor' else 'One scalar-alpha query in slot336 for both channels; actual native publication mask/filter chain retained'))


def audit_online_rule(raw,case,original,replanned,rule):
    """Independent NumPy closed-interval knot/endpoint preview and latch recursion."""
    n=int(np.asarray(raw['active_tick'],bool).sum());latched=False;departures=[];returns=[];dwell=[];start_active=None;max_error=0.
    replan_tick=case.get('replan_task_tick');path=original
    for tick in range(0,n,4):
        rebound=replanned is not None and replan_tick==tick
        if rebound:path=replanned
        progress=0. if tick==0 or rebound else float(raw['path_progress'][tick-1])
        if 'rule_pre_upper_progress' in raw:
            np.testing.assert_allclose(raw['rule_pre_upper_progress'][tick],progress,atol=1e-4,rtol=1e-6,err_msg='rule pre-upper progress differs from physical/rebound state')
        lo=np.clip(np.float32(progress),path['s'][0],path['s'][-1]);hi=np.minimum(np.float32(lo+np.float32(rule['preview_m'])),path['s'][-1])
        covered=(path['s']>=lo)&(path['s']<=hi)
        values=[abs(np.interp(lo,path['s'],path['curvature'])),abs(np.interp(hi,path['s'],path['curvature']))]
        if np.any(covered):values.append(float(np.max(abs(path['curvature'][covered]))))
        maximum=np.float32(max(values));proxy=np.float32(np.float32(case['speed'])**2*maximum)
        next_latched=bool(proxy>=rule['release_accel_m_s2']) if latched else bool(proxy>rule['engage_accel_m_s2'])
        expected=rule['active_alpha'] if next_latched else 1.
        for field,value in (('rule_preview_max_curvature',maximum),('rule_accel_estimate',proxy),('rule_alpha_target',expected),('alpha_exec',expected)):
            np.testing.assert_allclose(raw[field][tick],value,atol=2e-5,rtol=2e-5,err_msg='independent online rule differs '+field)
        np.testing.assert_array_equal(bool(raw['rule_latched'][tick]),next_latched)
        max_error=max(max_error,abs(float(raw['rule_accel_estimate'][tick])-float(proxy)))
        if next_latched and not latched:departures.append(tick*.005);start_active=tick*.005
        if latched and not next_latched:returns.append(tick*.005);dwell.append(tick*.005-start_active);start_active=None
        latched=next_latched
    speed=np.asarray(raw['actual_forward_speed'][:n])-raw['v_user'][:n];observed=n*.005
    return dict(departure_from_one_times_s=departures,return_to_one_times_s=returns,departure_count=len(departures),return_count=len(returns),
        completed_dwell_s=dwell,unreturned_at_actual_endpoint=latched,
        first_return_to_one_s=returns[0] if returns else None,last_departure_from_one_s=departures[-1] if departures else None,
        returns_before_force_onset_s=[t for t in returns if t<case.get('onset_s',float('-inf'))],open_dwell_observed_s=observed-start_active if latched else 0.,
        active_alpha_observed_duration_s=float(np.sum(np.asarray(raw['alpha_exec'][:n])==rule['active_alpha'])*.005),
        speed_error_rmse_m_s=float(np.sqrt(np.mean(speed**2))),speed_error_mae_m_s=float(abs(speed).mean()),
        maximum_independent_accel_error_m_s2=max_error,latch_preserved_across_replan=True,
        preprogress_source='Previous actual post-tick projection; task0/rebound progress0 because native project(path,current_xy,0,0) begins at the current-pose path origin. Pre-upper progress was not separately logged.',
        inputs='Current execution path and pre-upper physical projection progress, original speed request only; no event/onset/force labels or task clock in decision formula.')


def online_summary(rows):
    online=[r for r in rows if r['method']=='online_scalar'];pairs=[];summary={}
    for comparator in ('fixed0.5','fixed1'):
        base={(r['route'],r['event']):r for r in rows if r['method']==comparator}
        details=[]
        for r in online:
            b=base[r['route'],r['event']];a=r['horizons']['10'];h=b['horizons']['10']
            known=a['goal_arrived'] is not None and h['goal_arrived'] is not None and 'unknown' not in (a['physical_outcome'],h['physical_outcome'])
            details.append(dict(route=r['route'],event=r['event'],force_N=r['force_N'],candidate=r['case_id'],baseline=b['case_id'],
                contact_rescue=h['physical_outcome']=='contact' and a['physical_outcome']=='survived',new_contact=h['physical_outcome']!='contact' and a['physical_outcome']=='contact',
                new_fault_or_unknown=a['physical_outcome']=='unknown' or a['goal_arrived'] is None,
                goal1m_gain=known and bool(a['goal_arrived'] and not h['goal_arrived']),goal1m_loss=known and bool(h['goal_arrived'] and not a['goal_arrived']),
                candidate_outcome=a,baseline_outcome=h,candidate_observed_ticks=r['active_lower_ticks'],baseline_observed_ticks=b['active_lower_ticks'],alpha_rule=r['alpha_rule'],candidate_prefix=r['prefix'],baseline_prefix=b['prefix'],
                diagnostic_execution_path_user1_cost_difference=r['diagnostic_execution_path_user1_cost']-b['diagnostic_execution_path_user1_cost'],
                speed_rmse_difference_m_s=r['full_observed_speed_rmse_m_s']-b['full_observed_speed_rmse_m_s']))
        nominal=[x for x in details if not x['force_N']]
        summary[comparator]=dict(n=len(details),contact_rescues=sum(x['contact_rescue'] for x in details),new_contacts=sum(x['new_contact'] for x in details),
            nominal_no_new_contact_fault_or_goal_loss=all(not(x['new_contact'] or x['new_fault_or_unknown'] or x['goal1m_loss']) for x in nominal),nominal=nominal)
        pairs.append(dict(candidate='online_scalar',baseline=comparator,details=details))
    return dict(comparisons=summary,new_cases=len(online),cached_cases=len(rows)-len(online),
        unreturned_last_activations=[dict(case_id=r['case_id'],route=r['route'],event=r['event'],alpha_rule=r['alpha_rule']) for r in online if r['alpha_rule']['unreturned_at_actual_endpoint']],
        early_return_cases=[dict(case_id=r['case_id'],returns_before_force_onset_s=r['alpha_rule']['returns_before_force_onset_s']) for r in online if r['alpha_rule']['returns_before_force_onset_s']],adoption='not_automatically_adopted',
        scope='Causal geometric heuristic on familiar DEV; no onset label input, threshold search, new training or independent speed/steering causality claim.'),pairs


def validate_switching_inventory(inventory,protocol):
    actual=[]
    for c in inventory:
        response_method(c)
        actual.append((c['route'],c['speed'],c['origin'],c['speed_alpha'],c['steer_alpha'],c['switch_s'],c['force_N'],c['onset_s'],c['duration_s']))
    events=[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))
    expected=[(r['route'],r['speed'],o['origin'],o['speed_alpha'],o['steer_alpha'],t,f,onset,protocol['duration_s'])
              for r,o,t,(f,onset) in itertools.product(protocol['routes'],protocol['origins'],protocol['switch_times_s'],events)]
    if sorted(actual)!=sorted(expected) or len(set(actual))!=len(actual):raise ValueError('switching inventory differs from declared protocol')


def load_switching_cache(root,p,initial):
    online=p.get('mode')=='h75_online_scalar'
    source=Path(p['stay_source']);run=source/'evaluation'
    plan=json.loads((source/'plan.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256'] or plan['mode']!='h75_channel_response':raise ValueError('stay source seal/mode mismatch')
    status=json.loads((run/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=54:raise ValueError('complete54 stay source required')
    for key in ('config','initial_source','models','native_runtime_root'):
        if plan[key]!=p[key]:raise ValueError('stay source interface mismatch '+key)
    for key,expected in (('plan_sha256',digest(source/'plan.json')),('config',p['config']),('models',p['models']),('source',p['initial_source'])):
        if manifest[key]!=expected:raise ValueError('stay manifest mismatch '+key)
    source_inventory=json.loads((source/'inventory.json').read_text());validate_response_inventory(source_inventory,plan['protocol'])
    if manifest['cases']!=source_inventory:raise ValueError('stay manifest inventory mismatch')
    allrows=json.loads((run/'case_index.json').read_text())
    if len(allrows)!=54:raise ValueError('complete stay case index required')
    stored=json.loads((root/'cached_stay_records.json').read_text());selected=[];background=[]
    for lane,row in enumerate(allrows):
        if any(row.get(k)!=v for k,v in source_inventory[lane].items()):raise ValueError('stay source case identity drift')
        if row['force_N'] not in (0.,*p['protocol']['forces_N']) or row['onset_s'] not in ([p['protocol']['zero_onset_s'],*p['protocol']['onsets_s']] if online else [p['protocol']['zero_onset_s']]):continue
        if online:
            if row['mode']=='H75' and row['alpha'] in (.5,1.):selected.append(dict(row,origin='fixed'+format(row['alpha'],'g')))
            continue
        if row['alpha']==1. and row['mode'] in ('H75','hybrid'):
            selected.append(dict(row,origin='fixed1' if row['mode']=='H75' else 'hybrid'))
        elif row['alpha']==.5 and row['mode']=='H75':background.append(dict(row,origin='background',_cache_index=lane))
    if stored!=selected or len(selected)!=(36 if online else 12) or len(background)!=(0 if online else 6):raise ValueError('exact12 stays and6 background cases required')
    by_id={r['case_id']:i for i,r in enumerate(allrows)}
    cached=[dict(r,_cache_index=by_id[r['case_id']]) for r in selected]+background
    for r in cached:
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('cached trace SHA drift')
    return cached,manifest,dict(initial_filtered=initial['cached_initial_filtered'],initial_offsets=initial['cached_initial_offsets'])


def audit_switch_prefixes(rows,raws,protocol):
    stays={(r['route'],r['origin'],r['event']):r for r in rows if r.get('method','').startswith('stay_')}
    result=[]
    for r in rows:
        if not r['mode'].startswith('switch_'):continue
        other=stays[r['route'],r['origin'],r['event']];raw=raws[r['case_id']];old=raws[other['case_id']]
        tick=round(r['switch_s']/.005)
        if not np.asarray(raw['active_tick'][:tick]).all() or not np.asarray(old['active_tick'][:tick]).all():raise ValueError('complete physical prefix to switch required')
        maximum={};exceeded={}
        descriptive=protocol['prefix_comparison'].get('floating_descriptive_only',False)
        for field in protocol['prefix_comparison']['fields']:
            a=np.asarray(raw[field][:tick]);b=np.asarray(old[field][:tick])
            if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('own-switch prefix dtype/shape mismatch '+field)
            maximum[field]=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
            if np.issubdtype(a.dtype,np.inexact):
                if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite switch prefix '+field)
                atol=protocol['prefix_comparison'].get('field_absolute_tolerance',{}).get(field,protocol['prefix_comparison']['absolute_tolerance'])
                outside=~np.isclose(a,b,atol=atol,rtol=protocol['prefix_comparison']['relative_tolerance'])
                if np.any(outside):exceeded[field]=int(outside.sum())
                if not descriptive:np.testing.assert_allclose(a,b,atol=atol,rtol=protocol['prefix_comparison']['relative_tolerance'],err_msg='own-switch stay prefix mismatch '+field)
            else:np.testing.assert_array_equal(a,b,err_msg='own-switch discrete prefix mismatch '+field)
        result.append(dict(case_id=r['case_id'],stay_case_id=other['case_id'],physical_prefix_ticks=tick,
            switch_task_s=r['switch_s'],floating_fields_within_declared_tolerance=not bool(exceeded),floating_descriptive_only=descriptive,tolerance_exceedance_element_counts=exceeded,discrete_fields_exact=True,maximum_absolute_difference=maximum,fields=protocol['prefix_comparison']['fields']))
    if len(result)!=36:raise ValueError('all36 own-switch prefixes required')
    return result


def audit_continuation_prefix(root,p,rows,raws):
    """Each lane keeps its own authenticated source history, without replay."""
    source=Path(p['continuation_source']['source']);run=root/'evaluation'
    provenance=json.loads((run/'prefix_provenance.json').read_text())
    if provenance['source']!=str(source) or provenance['source_indices']!=list(range(36)) or provenance['reused_lower_slots']!=7200 or provenance['new_physics_ticks']!=0:
        raise ValueError('own-source prefix provenance mismatch')
    new=[r for r in rows if r['mode'].startswith('switch_')]
    if len(new)!=36:raise ValueError('36 own-source prefixes required')
    with np.load(source/'evaluation/chunk_00.npz') as original,np.load(run/'reused_source_prefix.npz') as reused:
        if set(original.files)!=set(reused.files):raise ValueError('reused own-prefix fields differ')
        for field in original.files:
            np.testing.assert_array_equal(reused[field],original[field],err_msg='own source prefix was changed')
            for lane,row in enumerate(new):
                np.testing.assert_array_equal(raws[row['case_id']][field][:200],original[field][:,lane],err_msg='case history differs from own source prefix')
    receipt=json.loads((run/'source_prefix_audit.json').read_text())
    if not receipt.get('discrete_exact') or (not receipt.get('all_fields_within_tolerance') and not (p['protocol']['prefix_comparison'].get('floating_descriptive_only') and receipt.get('floating_descriptive_only'))):raise ValueError('source-prefix stay audit incomplete')
    return dict(source=str(source),source_indices=list(range(36)),own_prefix_ticks=200,reused_lower_ticks=7200,
                all_reused_source_fields_exact=True,new_physics_steps=0,
                interpretation='Own source prefixes are exact; cross-lane origin states are only approximately matched under the user-authorized amendment.')


def switching_summary(rows):
    stays={(r['route'],r['origin'],r['event']):r for r in rows if r['method'].startswith('stay_')}
    groups={};paired=[]
    for r in rows:
        if not r['mode'].startswith('switch_'):continue
        b=stays[r['route'],r['origin'],r['event']];a=r['horizons']['10'];h=b['horizons']['10']
        known=a['goal_arrived'] is not None and h['goal_arrived'] is not None and 'unknown' not in (a['physical_outcome'],h['physical_outcome'])
        item=dict(route=r['route'],origin=r['origin'],event=r['event'],switch_s=r['switch_s'],force_N=r['force_N'],candidate=r['case_id'],stay=b['case_id'],
            contact_rescue=h['physical_outcome']=='contact' and a['physical_outcome']=='survived',
            new_contact=h['physical_outcome']!='contact' and a['physical_outcome']=='contact',
            new_fault_or_unknown=a['physical_outcome']=='unknown' or a['goal_arrived'] is None,
            goal1m_gain=known and a['goal_arrived'] and not h['goal_arrived'],
            goal1m_loss=known and h['goal_arrived'] and not a['goal_arrived'],
            arrival_time_difference_s=(a['first_arrival_s']-h['first_arrival_s']) if a['first_arrival_s'] is not None and h['first_arrival_s'] is not None else None,
            diagnostic_execution_path_user1_cost_difference=r['diagnostic_execution_path_user1_cost']-b['diagnostic_execution_path_user1_cost'],
            speed_rmse_difference_m_s=r['full_observed_speed_rmse_m_s']-b['full_observed_speed_rmse_m_s'],
            original_route_sample_distance_rmse_difference_m=r['original_route_sample_distance_rmse_m']-b['original_route_sample_distance_rmse_m'],
            first_switch_at_force_onset=r['switch_s']==r['onset_s'],candidate_observed_ticks=r['active_lower_ticks'],stay_observed_ticks=b['active_lower_ticks'],
            candidate_prefix=r['prefix'],stay_prefix=b['prefix'],candidate_outcome=a,stay_outcome=h)
        paired.append(item);key=r['route']+'/'+r['origin'];groups.setdefault(key,[]).append(item)
    if len(paired)!=36 or len(groups)!=4:raise ValueError('four full source groups required')
    report={}
    for key,items in groups.items():
        nominal=[r for r in items if not r['force_N']];disturbed=[r for r in items if r['force_N']]
        report[key]=dict(n=len(items),nominal=nominal,disturbed=disturbed,
            nominal_without_new_contact_fault_or_goal_loss=all(not(r['new_contact'] or r['new_fault_or_unknown'] or r['goal1m_loss']) for r in nominal),
            contact_rescues=sum(r['contact_rescue'] for r in disturbed),new_contacts=sum(r['new_contact'] for r in disturbed),goal_losses=sum(r['goal1m_loss'] for r in items))
    return dict(source_groups=report,new_cases=36,cached_stays=12,cached_fixed_half_background=6,
        adoption='not_automatically_adopted',cost_rule='No percentage improvement gate; execution-path user1 cost remains descriptive.',
        scope='Known preparation times;1.6s switch precedes first pulse step and is not reactive. Fixed.5 background is not a same-state comparator.'),paired


def response_plot_groups(rows):
    if rows and all('effective_user_alpha' in r for r in rows):return list(dict.fromkeys(r['event']+'__bank'+str(r['bank_index'])+'__route'+r['route']+'__user'+str(r['effective_user_alpha']) for r in rows))
    if any(r.get('mode') in TOLERANCE_ARMS[:4]+TEMPORAL_TOLERANCE_ARMS+USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS for r in rows):
        return list(dict.fromkeys(r['event']+'__bank'+str(r['bank_index'])+'__route'+r['route'] for r in rows))
    if any(r.get('mode')=='samestate_scalar' for r in rows):
        return list(dict.fromkeys(r['event']+'__source'+str(r['source_index']) for r in rows))
    if any('path_key' in r for r in rows):
        return list(dict.fromkeys(r['event']+'__bank'+str(r['bank_index']) for r in rows))
    if any(r.get('mode','').startswith('switch_') for r in rows):
        return list(dict.fromkeys(r['event']+'__'+r['origin'] for r in rows if r['mode'].startswith('switch_')))
    return list(dict.fromkeys(r['event'] for r in rows))


def response_plot_selection(rows,label):
    if rows and all('effective_user_alpha' in r for r in rows):
        base,user=label.rsplit('__user',1);base,route=base.rsplit('__route',1);event,bank=base.rsplit('__bank',1)
        return [r for r in rows if r['event']==event and r['bank_index']==int(bank) and r['route']==route and r['effective_user_alpha']==float(user)]
    if any(r.get('mode') in TOLERANCE_ARMS[:4]+TEMPORAL_TOLERANCE_ARMS+USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS for r in rows):
        base,route=label.rsplit('__route',1);event,bank=base.rsplit('__bank',1)
        return [r for r in rows if r['event']==event and r['bank_index']==int(bank) and r['route']==route]
    if any(r.get('mode')=='samestate_scalar' for r in rows):
        event,source=label.rsplit('__source',1)
        return [r for r in rows if r['event']==event and r['source_index']==int(source)]
    if any('path_key' in r for r in rows):
        event,bank=label.rsplit('__bank',1)
        return [r for r in rows if r['event']==event and r['bank_index']==int(bank)]
    if any(r.get('mode','').startswith('switch_') for r in rows):
        event,origin=label.rsplit('__',1)
        return [r for r in rows if r['event']==event and (r.get('origin')==origin or r.get('origin')=='background')]
    return [r for r in rows if r['event']==label]


def validate_samestate_inventory(rows,p):
    actual=[]
    for c in rows:
        if c['mode']!='samestate_scalar' or c['alpha']!=c.get('source_user_alpha',1.) or c['query_alpha']!=c['candidate_alpha']:raise ValueError('same-state user/query identity drift')
        if not np.isclose(c['onset_s'],p['source_task_s']+c['preparation_s'],atol=1e-8,rtol=0):raise ValueError('preparation/onset mismatch')
        if not np.isclose(c['primary_window_end_s'],c['onset_s']+c['duration_s']+p['primary_after_pulse_end_s']):raise ValueError('primary window mismatch')
        src=next(r for r in p['source_rows'] if r['source_index']==c['source_index'])
        if (c['route'],c['source_origin'],c['source_fingerprint'])!=(src['route'],src['origin'],src['fingerprint']):raise ValueError('source metadata mismatch')
        actual.append((c['source_index'],c['candidate_alpha'],c['preparation_s'],c['force_N']))
    expected=list(itertools.product([r['source_index'] for r in p['source_rows']],p['candidate_alphas'],p['preparation_s'],p['forces_N']))
    if sorted(actual)!=sorted(expected) or len(set(actual))!=72:raise ValueError('complete declared72 conditional suffixes required')


def samestate_primary_label(raw,case):
    """Primary contact label includes preparation, not merely post-onset recovery."""
    active=np.asarray(raw['active_tick'],bool);clock=np.rint(np.asarray(raw['time'])/.005).astype(int)
    begin=round(case['primary_window_start_s']/.005);end=round(case['primary_window_end_s']/.005)
    window=active&(clock>=begin)&(clock<end)
    faults=np.asarray(raw['policy_fault'],bool)|raw['lower_fault']|raw['nonfinite']|raw['continuation_geometry_fault']
    fault=bool(np.any(window&faults));contact=bool(np.any(window&raw['contact_failure']));physical=bool(np.any(window&raw['physical_failure']))
    complete=np.array_equal(clock[window],np.arange(begin,end))
    valid=not fault and (complete or contact or physical)
    return dict(no_fall_through_window=bool(not(contact or physical)) if valid else None,
        label_status='invalid_fault' if fault else 'fall' if contact or physical else 'no_fall' if complete else 'invalid_incomplete',
        window_start_s=case['primary_window_start_s'],window_end_s=case['primary_window_end_s'],
        observed_window_ticks=int(window.sum()),complete_window=complete,contact_in_window=contact,physical_failure_in_window=physical,
        prep_contact=bool(np.any(window&(clock<round(case['onset_s']/.005))&raw['contact_failure'])),
        prep_physical_failure=bool(np.any(window&(clock<round(case['onset_s']/.005))&raw['physical_failure'])),
        fault_in_window=fault,contract=f"Source task{case['primary_window_start_s']:g} through pulse END+3s; preparation fall is negative; faults/nonfinite invalid, missed goal is never a fall.")


def audit_samestate_query(raw,case):
    active=np.asarray(raw['active_tick'],bool);n=int(active.sum());alpha=case['candidate_alpha']
    np.testing.assert_array_equal(raw['alpha_user'][active],np.full(n,case.get('source_user_alpha',1.)));np.testing.assert_array_equal(raw['alpha_exec'][active],np.full(n,alpha))
    np.testing.assert_allclose(raw['query_observation'][active,336],alpha,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(raw['channel_alpha_pair'][active],np.full((n,2),alpha))
    permitted=active&np.asarray(raw['reference_target_permitted'],bool)
    np.testing.assert_allclose(raw['latent_z'][permitted],raw['composed_latent'][permitted],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(raw['latent_z'][active&~permitted],0)
    for field in ('query_observation','composed_latent','latent_z'):
        if not np.isfinite(np.asarray(raw[field])[active]).all():raise ValueError('nonfinite candidate evidence '+field)
    for start in range(0,n,4):
        for field in ('query_observation','composed_latent','alpha_exec','reference_target_permitted'):
            a=np.asarray(raw[field])[start:min(n,start+4)];np.testing.assert_array_equal(a,np.broadcast_to(a[0],a.shape))
    return dict(candidate_alpha=alpha,state_user_alpha=case.get('source_user_alpha',1.),candidate_requested_at_source_s=case.get('primary_window_start_s',1.),
        first_actual_permitted_query_s=float(raw['time'][np.flatnonzero(permitted)[0]]) if permitted.any() else None,
        permitted_ticks=int(permitted.sum()),masked_ticks=int((active&~permitted).sum()),checked='Scalar query held from common source; actual publication and native filters remain auditable')


def samestate_summary(rows):
    groups={};paired=[]
    for source in sorted({r['source_index'] for r in rows}):
        group=[r for r in rows if r['source_index']==source]
        groups[str(source)]=dict(source_user_alpha=group[0].get('source_user_alpha',1.),source_origin=group[0]['source_origin'],route=group[0]['route'],n_conditional_suffixes=len(group),
            primary_no_fall=sum(r['primary']['no_fall_through_window'] is True for r in group),
            primary_fall=sum(r['primary']['no_fall_through_window'] is False for r in group),invalid=sum(r['primary']['no_fall_through_window'] is None for r in group),
            conditions=[dict(case_id=r['case_id'],alpha=r['candidate_alpha'],prep_s=r['preparation_s'],force_N=r['force_N'],primary=r['primary'],full_H10=r['horizons']['10']) for r in group])
        for prep,force in itertools.product(sorted({r['preparation_s'] for r in group}),sorted({r['force_N'] for r in group})):
            conditions=[r for r in group if r['preparation_s']==prep and r['force_N']==force]
            for a,b in itertools.combinations(conditions,2):
                known=a['primary']['no_fall_through_window'] is not None and b['primary']['no_fall_through_window'] is not None
                paired.append(dict(source_index=source,preparation_s=prep,force_N=force,candidate=a['case_id'],comparator=b['case_id'],candidate_alpha=a['candidate_alpha'],comparator_alpha=b['candidate_alpha'],
                    primary_no_fall_change=int(a['primary']['no_fall_through_window'])-int(b['primary']['no_fall_through_window']) if known else None,
                    candidate_primary=a['primary'],comparator_primary=b['primary'],candidate_full_H10=a['horizons']['10'],comparator_full_H10=b['horizons']['10']))
    dimensions=[sorted({r[k] for r in rows}) for k in ('candidate_alpha','preparation_s','force_N')]
    expected=set(itertools.product(*dimensions))
    for source in groups:
        actual=[(r['candidate_alpha'],r['preparation_s'],r['force_N']) for r in rows if str(r['source_index'])==source]
        if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('complete same-state Cartesian source conditions required')
    return dict(source_groups=groups,conditional_suffixes=len(rows),independent_source_states=len(groups),
        adoption='no_learner_trained',interpretation='Familiar shared-ancestor state-conditional preparation/hypothetical-onset response only; preparation also changes absolute pulse onset and encountered geometry, so it does not isolate pure timing at the same force phase.Conditional suffixes are not independent source states. No automatic force/state search or classifier training.'),paired


def validate_bank144_inventory(rows,p):
    fresh=p.get('source_initialization')=='prepared_bank_all_prefixes'
    late=p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192')
    groups={g['source_index']:g for g in p['source_groups']};actual=[]
    for c in rows:
        group=groups[c['source_index']]
        if any(c.get(k)!=v for k,v in group.items()):raise ValueError('bank source metadata drift')
        if c['split']!=('late_user_development' if p.get('mode')=='h75_user_late_state192' else 'late_state_development' if late else 'new_confirmation' if fresh else 'development' if c['bank_index']==0 else 'label_heldout') or c['alpha']!=c.get('source_user_alpha',1.) or c['query_alpha']!=c['candidate_alpha']:raise ValueError('bank split/user/query identity mismatch')
        if not np.isclose(c['onset_s'],p.get('source_task_s',1.)+c['preparation_s']) or not np.isclose(c['primary_window_end_s'],c['onset_s']+.1+3.):raise ValueError('bank conditional timing mismatch')
        actual.append((c['source_index'],c['candidate_alpha'],c['preparation_s'],c['force_N']))
    expected=list(itertools.product(groups,p['candidate_alphas'],p['preparation_s'],p['forces_N']))
    if sorted(actual)!=sorted(expected) or len(set(actual))!=p.get('expected_cases',144):raise ValueError('complete declared conditional inventory required')


def select_bank_label_lanes(index,include_heldout_labels=False,include_confirmation_labels=False):
    """Select BEFORE any suffix raw open, scoring, plotting or label export."""
    fresh=any(row.get('split')=='new_confirmation' for row in index)
    if fresh:
        if not include_confirmation_labels or include_heldout_labels:raise ValueError('new confirmation labels require explicit confirmation flag only')
        if len(index)!=144 or {row['bank_index'] for row in index}!={2,5} or any(row['split']!='new_confirmation' for row in index):raise ValueError('new confirmation identity mismatch')
        return list(enumerate(index))
    if include_confirmation_labels:raise ValueError('confirmation flag cannot unseal historical labels')
    selected=[]
    for lane,row in enumerate(index):
        expected='development' if row['bank_index']==0 else 'label_heldout'
        if row['split']!=expected:raise ValueError('bank label seal metadata mismatch')
        if row['bank_index']==0 or include_heldout_labels:selected.append((lane,row))
    if len(selected)!=(144 if include_heldout_labels else 72):raise ValueError('declared bank label selection incomplete')
    return selected


def enriched_bank_runtime_inventory(inventory,sources):
    """Runtime adds only the generated, sealed complete-source fingerprint."""
    by_id={r['source_index']:r for r in sources}
    if len(by_id)!=len(sources):raise ValueError('duplicate generated source identity')
    expected=[]
    for case in inventory:
        source=by_id[case['source_index']]
        for key in ('bank_index','route','origin','path_key','split','source_group_id'):
            if case[key]!=source[key]:raise ValueError('generated source metadata differs '+key)
        fingerprint=source['source_fingerprint']
        if not isinstance(fingerprint,str) or len(fingerprint)!=64:raise ValueError('invalid generated source fingerprint')
        if 'source_fingerprint' in case and case['source_fingerprint']!=fingerprint:raise ValueError('sealed/runtime source fingerprint conflict')
        expected.append(dict(case,source_fingerprint=fingerprint))
    return expected


def bank144_original_path_directory(root,p):
    """New banks own their generated paths; legacy banks inherit saved paths."""
    if p['protocol'].get('source_initialization')=='prepared_bank_all_prefixes' or p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192'):return root/'evaluation'
    return Path(p['samestate_source']['source'])/'evaluation'


def bank144_analysis_audit_target(out):
    """Resume only the known pre-results failure, preserving its audit artifact."""
    if not out.exists():
        out.mkdir();return out/'initial_carry_review.json'
    if {x.name for x in out.iterdir()}!={'initial_carry_review.json'}:raise ValueError('analysis output already contains derived results; use a fresh output')
    return out/'initial_carry_review_recheck.json'


def analyze_bank144(root,p,*,include_heldout_labels=False,include_confirmation_labels=False):
    late=p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192')
    fresh=p['protocol'].get('source_initialization')=='prepared_bank_all_prefixes' and not late
    count=p['protocol'].get('expected_cases',144);source_s=p['protocol'].get('source_task_s',1.);start_tick=round(source_s/.005);suffix_s=p['protocol'].get('new_horizon_s',9.);max_ticks=round(suffix_s/.005)
    # Reject before opening suffix status/index/raw or generating any labels.
    if fresh and (not include_confirmation_labels or include_heldout_labels):raise ValueError('explicit --include-confirmation-labels required')
    if not fresh and not late and include_confirmation_labels:raise ValueError('confirmation flag cannot unseal historical labels')
    run=root/'evaluation';out=root/('analysis_late' if late else 'analysis_confirmation' if fresh else 'analysis_final' if include_heldout_labels else 'analysis_development')
    status=json.loads((run/'status.json').read_text());inventory=json.loads((root/'inventory.json').read_text());index=json.loads((run/'case_index.json').read_text())
    validate_bank144_inventory(inventory,p['protocol'])
    if status['phase']!='completed' or status['completed_cases']!=count or len(index)!=count:raise ValueError('completed declared suffix collection required')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('bank plan drift')
    for file,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(file)!=sha:raise ValueError('bank collection source/input drift '+file)
    manifest=json.loads((run/'manifest.json').read_text())
    sources=json.loads((root/'sources/records.json').read_text());source_seal=json.loads((root/'sources/seal.json').read_text())
    records_file=str(root/'sources/records.json')
    if source_seal['plan_sha256']!=digest(root/'plan.json') or source_seal['files'].get(records_file)!=digest(records_file):raise ValueError('generated source records seal mismatch')
    runtime_cases=enriched_bank_runtime_inventory(inventory,sources)
    for key,expected in (('plan_sha256',digest(root/'plan.json')),('config',p['config']),('models',p['models']),('cases',runtime_cases),('source',p['initial_source'])):
        if manifest[key]!=expected:raise ValueError('bank manifest identity mismatch '+key)
    for row,expected in zip(index,runtime_cases):
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('bank runtime index/source fingerprint mismatch')
    selected=list(enumerate(index)) if late else select_bank_label_lanes(index,include_heldout_labels,include_confirmation_labels)
    # Nothing below opens unselected bank1 suffix traces or emits their labels.
    audit_target=bank144_analysis_audit_target(out);helper=Path(__file__).with_name('audit_native_goal_initial.py')
    subprocess.run([sys.executable,'-B',str(helper),str(root),str(audit_target)],check=True)
    if audit_target.name!='initial_carry_review.json' and audit_target.read_bytes()!=(out/'initial_carry_review.json').read_bytes():raise ValueError('rechecked initial audit differs from preserved audit')
    initial=json.loads(audit_target.read_text());sources=json.loads((root/'sources/records.json').read_text());source_by_id={r['source_index']:r for r in sources}
    original_paths=bank144_original_path_directory(root,p);paths={};goals={};traces={};executed={};rebuilt={};rows=[];ticks=0
    # Only selected bank paths/decisions are used for label-bearing diagnostics.
    for key in sorted({analysis_path_key(c) for _,c in selected}):
        with np.load(run/(key+'_path.npz')) as z:paths[key]=dict(z)
        with np.load(original_paths/(key+'_path.npz')) as z:
            for field in z.files:np.testing.assert_array_equal(paths[key][field],z[field])
        path=paths[key];goals[key]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,a]) for a in (0,1)])
    selected_ids={c['case_id'] for _,c in selected}
    decision_rows=[d for d in json.loads((run/'replanning_decisions.json').read_text()) if d['case_id'] in selected_ids];decisions={d['case_id']:d for d in decision_rows}
    if len(decisions)!=len(decision_rows):raise ValueError('more than one bank replan per case')
    with np.load(root/'sources/host_state.npz') as host:
        source_host={k:host[k].copy() for k in host.files}
        if not late and np.any(host['ever_arrived']|host['replanned']):raise ValueError('source antecedent must be original-path/unarrived')
    for lane,c in selected:
        if any(c.get(k)!=v for k,v in inventory[lane].items()):raise ValueError('selected bank inventory drift')
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('selected bank raw SHA drift')
        with np.load(c['raw_file']) as z:raw=dict(z)
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum());ticks+=n
        if not 0<n<=max_ticks or not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('bank suffix observation prefix broken')
        np.testing.assert_allclose(raw['time'][active],source_s+np.arange(n)*.005,atol=3e-6,rtol=0);np.testing.assert_allclose(raw['v_user'][active],c['speed'],atol=2e-7,rtol=0)
        key=analysis_path_key(c);goal=goals[key]
        np.testing.assert_allclose(c['original_goal_xy'],goal,atol=3e-6,rtol=0);np.testing.assert_allclose(manifest['original_goals_xy'][lane],goal,atol=3e-6,rtol=0)
        query=audit_samestate_query(raw,c);clock=np.rint(raw['time']/.005).astype(int)
        pulse=audit_pulse(dict(raw,physical_tick=clock),dict(force_N=c['force_N'],world_direction=[0.,1.,0.],pulse_onset_task_tick=round(c['onset_s']/.005),pulse_duration_s=.1))
        chain=verify_correction_chain(dict(raw,issued_latent=raw['latent_z'],nominal_reference=raw['nominal'],reference=raw['governed'],correction_offset=raw['offsets']),p['config'],initial_filtered=initial['initial_filtered'][lane],initial_offsets=initial['initial_offsets'][lane])
        physical=dict(raw,physical_tick=clock-start_tick,policy_fault=raw['policy_fault']|raw['continuation_geometry_fault'])
        outcome=goal_arrival_metrics(physical,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=suffix_s)
        relative_arrival=outcome['first_arrival_s']
        if relative_arrival is not None:outcome['first_arrival_s']=relative_arrival+source_s
        outcome.update(physical_outcome=physical_outcome(raw['active_tick'],raw['contact_failure'],physical['policy_fault']|raw['lower_fault']|raw['nonfinite'],max_ticks),
            observation_contract=f'Source task{source_s:g}; observed actual suffix to10, first arrival uses original task clock; prefix-goal flag retained separately.')
        source_id=c['source_index'];antecedent_replanned=bool(source_host['replanned'][source_id]);antecedent_arrived=bool(source_host['ever_arrived'][source_id])
        outcome['goal_arrived_before_source']=antecedent_arrived
        outcome['goal_arrived_in_suffix']=outcome['goal_arrived']
        if antecedent_arrived:outcome.update(goal_arrived=True,first_arrival_s=None)
        source_execution=paths[key]
        if late:
            with np.load(root/'sources'/f'executed_source_{source_id:02d}_path.npz') as z:source_execution=dict(z)
        decision=decisions.get(c['case_id']);replan=run/f'replanned_path_{lane:02d}.npz';changed=None;replan_tick=start_tick+n
        if (bool(decision) or antecedent_replanned)!=bool(c['replanned']) or replan.exists()!=bool(decision):raise ValueError('bank replan evidence mismatch')
        if decision:
            replan_tick=decision['task_tick']
            if decision['case_index']!=lane or replan_tick not in range(start_tick,1801,200):raise ValueError('bank replan clock mismatch')
            from sttw_control.goal_replanning import replan_requested
            if not replan_requested(xy=np.asarray(decision['measured_pose'])[:2],goal_xy=goal,cross_track_m=decision['cross_track_m'],follower_invalid=decision['follower_invalid'],already_replanned=False,contact=decision['contact'],fault=decision['fault'],ever_arrived=decision['ever_arrived']):raise ValueError('bank replan lacks measured trigger')
            with np.load(replan) as z:changed=dict(z)
            np.testing.assert_allclose([np.interp(float(changed['goal']),changed['s'],changed['xy'][:,a]) for a in (0,1)],goal,atol=3e-6,rtol=0);executed[c['case_id']]=changed
        data=reconstruct_execution_path_reward(raw,c,source_execution,changed,replan_tick-start_tick,p['config'],p['initial_source']['controller'],manifest['reward_components'],start_tick=start_tick)
        if late and changed is None and antecedent_replanned:executed[c['case_id']]=source_execution
        rebuilt[c['case_id']]=data;traces[c['case_id']]=raw
        fault=physical['policy_fault']|raw['lower_fault']|raw['nonfinite'];fatal=np.flatnonzero(active&(raw['contact_failure']|fault));stop=int(fatal[0]) if len(fatal) else n
        task_stop=min(stop,round(relative_arrival/.005)) if relative_arrival is not None else stop
        event=f"prep{c['preparation_s']:g}_F{c['force_N']:+g}"
        rows.append(dict(c,event=event,method=f"alpha{c['candidate_alpha']:g}",source_fingerprint=source_by_id[c['source_index']]['source_fingerprint'],
            source_feature_file='../evaluation/source_features.npz',source_feature_row=lane,diagnostic_file=f'case_{lane:03d}_diagnostics.npz',
            inherited_execution_path_file=str(root/'sources'/f'executed_source_{source_id:02d}_path.npz') if late else None,
            primary=samestate_primary_label(raw,c),horizons={'10':outcome},active_lower_ticks=n,new_suffix_lower_ticks=n,trace_start_s=source_s,prefix=dict(ticks=task_stop),
            pulse=pulse,query=query,correction_chain=chain,replan_task_tick=None if decision is None else replan_tick,
            diagnostic_execution_path_user1_cost=float(-data['user1_reward'].sum()),actual_final_xy=raw['actual_xy'][n-1].tolist()))
        numeric={k:v[:n] for k,v in raw.items()};numeric.update(user1_suffix_reward=data['user1_reward'],cumulative_user1_suffix_reward=np.cumsum(data['user1_reward']),
            user1_suffix_reward_components=data['user1_running_components'],cumulative_user1_suffix_components=np.cumsum(data['user1_running_components'],axis=0),terminal_adjustment=data['user1_terminal_adjustment'])
        np.savez_compressed(out/f'case_{lane:03d}_diagnostics.npz',**numeric)
    summary,paired=samestate_summary(rows)
    summary.update(label_selection=('user_late_same_state_all192' if p.get('mode')=='h75_user_late_state192' else 'late_same_state_all48') if late else 'new_confirmation_explicit_all144' if fresh else 'final_all_banks_explicitly_unsealed' if include_heldout_labels else 'bank0_development_only',heldout_labels_accessed=include_heldout_labels,
        data_role=p['data_role'] if fresh or late else 'Old bank3 training; bank0 development; bank1 new labels held until frozen fitting recipe. Prior bank1 control seen; not independent TEST.',confirmation_labels_accessed=fresh)
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),analyzed_suffix_cases=len(rows),
        analyzed_suffix_active_lower_ticks=ticks,active_new_hybrid_prefix_lower_ticks=initial['active_new_prefix_lower_ticks'],maximum_suffix_budget=p['protocol']['maximum_lower_ticks'],maximum_total_budget=p['protocol']['maximum_total_lower_ticks'],confirmation_labels_accessed=fresh,
        heldout_suffix_labels_accessed=include_heldout_labels,unselected_suffix_files_opened=0,new_physics_steps=0,optimizer_updates=0,
        limitation=f"Prefix{p['protocol']['maximum_prefix_lower_ticks']} is separate source collection. Two ancestors are not independent training seeds. No predictor fitting."))
    # Existing renderer expects root/analysis; pass an explicit output directory.
    render_response(root,rows,traces,paths,goals,executed,rebuilt,manifest['reward_components'],output_directory=out)
    lines=['# 分源容扰数据：'+('显式解封两bank最终评价' if include_heldout_labels else '仅bank0开发标签'),'',
        '默认不打开、不计算、不绘制、不导出bank1新后缀标签；bank1此前控制工况已见，不能称独立TEST。仅显式最终评价选项才解封。',
        '八个source身份可核查，但默认结果仅含bank0四source的72条件；不从全局活动步数反推或报告bank1失败。原bank3旧72用于训练，本轮bank0开发；未来按source祖先分组，禁止随机按行拆分。',
        '原source前缀不重跑也不拼接。XY仅展示task1..10实测后缀与原始/执行路径；主标签source至pulseEND+3s，准备期摔倒为负、fault为invalid。准备改变绝对起扰时刻与遭遇几何，不隔离纯时序。']
    if fresh:
        lines=['# 冻结预测器新来源确认：banks2/5 全144条件','','所有八个fixed1/hybrid前缀新生成，共1600lower；144后缀上限259200；无拟合、无bank1标签读取。两个祖先非独立训练种子。','XY仅展示task1..10实测后缀及固定原始路径；原任务/物理/主标签不变。']
    if late:lines=['# 晚期同态48后缀','','四个task6完整source×三alpha×两prep×两方向=48；16个同条件图每图三alpha。主窗source至pulseEND+3s，准备触地为负、fault为invalid。保持原worldgoal；source前已到达/接回标记保留；suffix仅展示真实task6..10轨迹，执行路径奖励为诊断。']
    if p.get('mode')=='h75_user_late_state192':lines=['# 用户0/.5晚期同态192后缀','','16个完整task6源×三候选alpha×两prep×两方向；64个条件同图比较三候选。source用户alpha0/.5保留用于评分；候选同时查询H75两通道。主窗source至pulseEND+3s，准备期触地为负，故障invalid；整task10另报。0新前缀物理，suffix上限153600lower，0训练。']
    for label in response_plot_groups(rows):lines+=['',f'## {label}','',f'![Selected suffix XY](full_xy_{label}.png)']
    lines+=['','[所选逐例标签](results.json) · [按source分组](summary.json) · [配对](paired.json) · [身份与单独费用](receipt.json)',
        '本轮不拟合预测器。351源观测含真实速度/历史部署依赖，尚非最终小特征集。执行路径user1奖励仅诊断；四固定1来源复用，一秒hybrid源仅新800lower，后缀物理单独计费。']
    if late:lines[-1]='四个固定half task6完整源从封存online120复用，0新前缀物理；48后缀上限38400；零训练，冻结q另行评价；不把48分支当独立祖先。'
    if p.get('mode')=='h75_user_late_state192':lines[-1]='全部16个用户0/.5的task6来源直接复用，真实新prefix=0；192后缀最多153600lower。冻结last slot1预测另行核验，非BCE；192分支不是192独立祖先。执行路径奖励按source用户评分，仅为诊断。'
    if fresh:lines[-1]='本轮只确认封存预测器；没有拟合或在线替换。全部八来源前缀1600lower，后缀单独计费；351源观测含真实速度部署依赖。'
    (out/'README.md').write_text('\n'.join(lines)+'\n');return summary


def analyze_samestate(root,p):
    run=root/'evaluation';out=root/'analysis';status=json.loads((run/'status.json').read_text())
    inventory=json.loads((root/'inventory.json').read_text());index=json.loads((run/'case_index.json').read_text())
    validate_samestate_inventory(inventory,p['protocol'])
    if status['phase']!='completed' or status['completed_cases']!=72 or len(index)!=72:raise ValueError('completed72 suffixes required; no partial scoring')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('same-state plan drift')
    for file,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(file)!=sha:raise ValueError('same-state source/input drift '+file)
    manifest=json.loads((run/'manifest.json').read_text())
    for key,expected in (('plan_sha256',digest(root/'plan.json')),('config',p['config']),('models',p['models']),('cases',inventory),('source',p['initial_source'])):
        if manifest[key]!=expected:raise ValueError('same-state manifest drift '+key)
    source=Path(p['samestate_source']['source']);out.mkdir(exist_ok=False);helper=Path(__file__).with_name('audit_native_goal_initial.py')
    subprocess.run([sys.executable,'-B',str(helper),str(root),str(out/'initial_carry_review.json')],check=True)
    initial=json.loads((out/'initial_carry_review.json').read_text());paths={};goals={};traces={};rebuilt={};executed={};rows=[];ticks=0
    provenance=json.loads((run/'prefix_provenance.json').read_text())
    if provenance['source']!=str(source) or provenance['source_indices']!=[c['source_index'] for c in inventory] or provenance['new_physics_ticks']!=0:raise ValueError('display-prefix provenance mismatch')
    with np.load(run/'reused_source_prefix.npz') as z:prefix=dict(z)
    with np.load(source/'evaluation/reused_source_prefix.npz') as z:
        for field in z.files:np.testing.assert_array_equal(prefix[field],z[field][:,[c['source_index'] for c in inventory]])
    for route in sorted({c['route'] for c in inventory}):
        with np.load(run/(route+'_path.npz')) as z:paths[route]=dict(z)
        with np.load(source/'evaluation'/(route+'_path.npz')) as z:
            for field in z.files:np.testing.assert_array_equal(paths[route][field],z[field])
        path=paths[route];goals[route]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,a]) for a in (0,1)])
    decision_rows=json.loads((run/'replanning_decisions.json').read_text());decisions={d['case_id']:d for d in decision_rows}
    if len(decisions)!=len(decision_rows):raise ValueError('more than one replan per same-state case')
    for lane,c in enumerate(index):
        if any(c.get(k)!=v for k,v in inventory[lane].items()):raise ValueError('same-state case inventory drift')
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('same-state raw drift')
        with np.load(c['raw_file']) as z:raw=dict(z)
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum());ticks+=n
        if not 0<n<=1800 or not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('suffix active prefix broken')
        np.testing.assert_allclose(raw['time'][active],1.+np.arange(n)*.005,atol=3e-6,rtol=0)
        np.testing.assert_allclose(raw['v_user'][active],c['speed'],atol=2e-7,rtol=0)
        goal=goals[c['route']];np.testing.assert_allclose(c['original_goal_xy'],goal,atol=3e-6,rtol=0);np.testing.assert_allclose(manifest['original_goals_xy'][lane],goal,atol=3e-6,rtol=0)
        query=audit_samestate_query(raw,c)
        physical_tick=np.rint(raw['time']/.005).astype(int)
        pulse=audit_pulse(dict(raw,physical_tick=physical_tick),dict(force_N=c['force_N'],world_direction=[0.,1.,0.],pulse_onset_task_tick=round(c['onset_s']/.005),pulse_duration_s=c['duration_s']))
        chain=verify_correction_chain(dict(raw,issued_latent=raw['latent_z'],nominal_reference=raw['nominal'],reference=raw['governed'],correction_offset=raw['offsets']),p['config'],initial_filtered=initial['initial_filtered'][lane],initial_offsets=initial['initial_offsets'][lane])
        full={field:np.concatenate([prefix[field][:,lane],value]) for field,value in raw.items() if field in prefix}
        full['physical_tick']=np.rint(full['time']/.005).astype(int);full['policy_fault']=full['policy_fault']|full['continuation_geometry_fault']
        outcome=goal_arrival_metrics(full,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=10.);outcome['physical_outcome']=physical_outcome(full['active_tick'],full['contact_failure'],full['policy_fault']|full['nonfinite']|full['lower_fault'],2000)
        path=paths[c['route']];decision=decisions.get(c['case_id']);changed=None;replan_tick=200+n
        replan=run/f'replanned_path_{lane:02d}.npz'
        if replan.exists()!=bool(c['replanned']) or bool(decision)!=bool(c['replanned']):raise ValueError('same-state replan evidence mismatch')
        if decision:
            replan_tick=decision['task_tick']
            if decision['case_index']!=lane or replan_tick not in range(200,1801,200):raise ValueError('same-state replan boundary mismatch')
            from sttw_control.goal_replanning import replan_requested
            if not replan_requested(xy=np.asarray(decision['measured_pose'])[:2],goal_xy=goal,cross_track_m=decision['cross_track_m'],follower_invalid=decision['follower_invalid'],already_replanned=False,contact=decision['contact'],fault=decision['fault'],ever_arrived=decision['ever_arrived']):raise ValueError('same-state replan lacks measured trigger')
            np.testing.assert_allclose(decision['measured_pose'][:2],full['actual_xy'][replan_tick-1],atol=3e-6,rtol=0)
            with np.load(replan) as z:changed=dict(z)
            np.testing.assert_allclose([np.interp(float(changed['goal']),changed['s'],changed['xy'][:,a]) for a in (0,1)],goal,atol=3e-6,rtol=0);executed[c['case_id']]=changed
        data=reconstruct_execution_path_reward(full,c,path,changed,replan_tick,p['config'],p['initial_source']['controller'],manifest['reward_components'])
        rebuilt[c['case_id']]=data;traces[c['case_id']]=full
        arrival=outcome['first_arrival_s'];fatal=np.flatnonzero(full['active_tick']&(full['contact_failure']|full['policy_fault']|full['nonfinite']|full['lower_fault']))
        stop=int(fatal[0]) if len(fatal) else 200+n;task_stop=min(stop,round(arrival/.005)) if arrival is not None else stop
        event=f"prep{c['preparation_s']:g}_F{c['force_N']:+g}"
        rows.append(dict(c,event=event,source_group_id=f"source{c['source_index']}",source_feature_file='../evaluation/source_features.npz',source_feature_row=lane,diagnostic_file=f'case_{lane:02d}_diagnostics.npz',method=f"alpha{c['candidate_alpha']:g}",primary=samestate_primary_label(raw,c),horizons={'10':outcome},active_lower_ticks=200+n,new_suffix_lower_ticks=n,
            prefix=dict(ticks=task_stop),pulse=pulse,query=query,correction_chain=chain,prefix_role='200 original ticks are reused display/history only; primary label and new cost charge use task1 suffix',
            replan_task_tick=None if decision is None else replan_tick,diagnostic_execution_path_user1_cost=float(-data['user1_reward'].sum()),
            actual_final_xy=raw['actual_xy'][n-1].tolist()))
        numeric={k:v[:n] for k,v in raw.items()};numeric.update(primary_window_mask=(physical_tick[:n]<round(c['primary_window_end_s']/.005)),
            reconstructed_user1_suffix_reward=data['user1_reward'][200:200+n],cumulative_user1_suffix_reward=np.cumsum(data['user1_reward'][200:200+n]),
            user1_suffix_reward_components=data['user1_running_components'][200:200+n],cumulative_user1_suffix_components=np.cumsum(data['user1_running_components'][200:200+n],axis=0),
            terminal_adjustment=data['user1_terminal_adjustment'][200:200+n])
        np.savez_compressed(out/f'case_{lane:02d}_diagnostics.npz',**numeric)
    if ticks!=status['active_lower_ticks'] or ticks>129600:raise ValueError('suffix physical charge mismatch')
    summary,paired=samestate_summary(rows);atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(new_suffix_cases=72,source_states=4,new_active_lower_ticks=ticks,reused_display_prefix_lower_ticks=14400,new_physics_steps=0,optimizer_updates=0,
        plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),initial_carry_sha256=initial['initial_carry_sha256'],
        checked='All raw/source/config/model identities, exact complete-state copies, source features/host state, display-prefix provenance, actual candidate query+mask/filter, world pulse and per-step user1 execution-path rewards.',
        limitation='Four shared-ancestor familiar source states; no learned predictor. Reused prefix is not newly sampled or72 independent ancestor states. Non-task components remain authenticated raw terms; execution-path costs are diagnostic after replan.'))
    render_response(root,rows,traces,paths,goals,executed,rebuilt,manifest['reward_components'])
    lines=['# 同完整状态容扰72后缀','','四个source task1完整状态，各18个alpha/准备时间/方向条件；不是72个独立初态。所有候选task1立即切换并持有，准备时间也改变绝对起扰时刻与遭遇几何，属于状态条件下的准备/假设起扰响应，不能隔离同一施力相位的纯时序效应。无reset、无新训练、无原前缀物理重跑。',
        '','主标签覆盖source至pulse END+3s（task4.1/4.3/4.7）；准备期触地为负，fault/nonfinite为invalid，未到目标不算摔倒。完整H10真实触地及原goal1m另列。',
        '','XY原前200步为注明来源的显示前缀；候选查询/实际发布/filter与后缀数据单独保留。执行路径user1奖励仅诊断，不是重规划后的原S效用。']
    for label in response_plot_groups(rows):lines+=['',f'## {label}','',f'![Full original/executed XY](full_xy_{label}.png)','',f'![First-arrival prefix](task_xy_{label}.png)']
    lines+=['','[全部逐例标签](results.json) · [四source分组](summary.json) · [同源alpha配对](paired.json) · [状态/发布/奖励核验](receipt.json)',
        '仅本固定试采，不自动扩大力值/状态或训练分类器。原72后缀与源prefix身份均保留；实际失败终点截止。',
        '源351维观测包含真实速度/历史等仿真部署依赖，尚未选择最终小特征集。未来训练/验证必须按source祖先分组，禁止随机按72行拆分造成同源泄漏；四源也不是独立训练种子。']
    (out/'README.md').write_text('\n'.join(lines)+'\n');return summary


TOLERANCE_ARMS=('q_rank','q_bce','task_guard_only','existing_geometry','fixed_half','fixed1')
TEMPORAL_TOLERANCE_ARMS=('temporal_rank_selected','temporal_rank_last')
BROADEN_TOLERANCE_ARMS=('temporal_rank_last','task_guard_only','fixed_half','fixed1','speedcap')
GUARD_FALLBACK_ARMS=('user0_guardfallback','userhalf_guardfallback','user1_guardfallback')
SPEED_QUERY_ARMS=('user0_speedquery','userhalf_speedquery','user1_speedquery')
FROZEN_CONFIRMATION_ARMS=('user0_taskaware','user0_guard','user0_fixed','userhalf_taskaware','userhalf_guard','userhalf_fixed','user1_taskaware','user1_guard','user1_fixed','user1_speedcap')
TASK_AWARE_ARMS=('user0_taskaware','userhalf_taskaware','user1_taskaware')
PRIOR_ARMS=('user0_prior','userhalf_prior','user1_prior')
USER_PREFERENCE_ARMS=('user0_last','user0_guard','user0_fixed','userhalf_last','userhalf_guard')


def tolerance_comparison_arms(rows):
    if any(is_speedcap_case(r) for r in rows):return BROADEN_TOLERANCE_ARMS
    return TEMPORAL_TOLERANCE_ARMS+TOLERANCE_ARMS if any(r.get('mode') in TEMPORAL_TOLERANCE_ARMS for r in rows) else TOLERANCE_ARMS


def load_tolerance_analysis_export(root,p):
    """Identity checks precede probability reads; no model selection or fitting."""
    directory=root/'q';receipt=json.loads((directory/'receipt.json').read_text())
    declared=p['frozen_predictor_export']
    expected_map={'temporal_rank_selected':0,'temporal_rank_last':1} if p['mode']=='tolerance_online40' else {'q_rank':1,'q_bce':0}
    expected_order=['temporal_rank_selected','temporal_rank_last'] if p['mode']=='tolerance_online40' else ['bce','bce_rank']
    if p['mode']=='user_preference120':
        expected_map={a:1 for a in USER_PREFERENCE_ARMS};expected_order=['temporal_rank_selected','temporal_rank_last']
    if p['mode'] in ('user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):
        expected_map={a:0 for a in (p['protocol']['arms'] if p['mode']=='frozen_transfer_evaluation' else GUARD_FALLBACK_ARMS if p['mode']=='guard_fallback_evaluation' else SPEED_QUERY_ARMS if p['mode']=='speed_query_online60' else FROZEN_CONFIRMATION_ARMS if p['mode']=='frozen_confirmation200' else TASK_AWARE_ARMS if p['mode']=='task_aware_fallback72' else PRIOR_ARMS)};expected_order=['speed_query_rank_last'] if p['mode']=='speed_query_online60' else ['prior_rank_last']
    if p['mode']=='tolerance_broaden120':
        expected_map={a:1 for a in BROADEN_TOLERANCE_ARMS};expected_order=['temporal_rank_selected','temporal_rank_last']
    if Path(declared['directory'])!=directory or declared['receipt_sha256']!=digest(directory/'receipt.json') or declared['arm_to_export_index']!=expected_map:raise ValueError('declared frozen export identity mismatch')
    if receipt['arm_order']!=expected_order or receipt['new_optimizer_updates'] or receipt['new_physics_steps']:raise ValueError('invalid frozen q export')
    if max(receipt['maximum_probability_errors'].values())>2e-6:raise ValueError('Torch/JAX numerical import failed')
    for name,value in receipt['files_sha256'].items():
        if digest(directory/name)!=value:raise ValueError('frozen q export drift '+name)
    if p['input_sha256'].get(str(directory/'receipt.json'))!=digest(directory/'receipt.json'):raise ValueError('q receipt not sealed in run')
    with np.load(directory/'parameters.npz') as z:return dict(z)


def is_speedcap_case(case):
    return case.get('method',case.get('mode'))=='speedcap'


def audit_tolerance_publication(raw,case):
    n=int(np.asarray(raw['active_tick'],bool).sum());alpha=np.asarray(raw['alpha_exec'][:n]);permission=np.asarray(raw['reference_target_permitted'][:n],bool)
    if case['mode'] not in TOLERANCE_ARMS+TEMPORAL_TOLERANCE_ARMS+USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS+FROZEN_CONFIRMATION_ARMS+SPEED_QUERY_ARMS+GUARD_FALLBACK_ARMS+('speedcap',) or not np.isin(alpha,[0.,.5,1.]).all():raise ValueError('undeclared tolerance alpha/arm')
    np.testing.assert_array_equal(raw['alpha_user'][:n],case.get('alpha',1.))
    np.testing.assert_array_equal(raw['coord_candidate_alpha'][:n],np.tile([0.,.5,1.],(n,1)))
    np.testing.assert_array_equal(raw['channel_alpha_pair'][:n],np.column_stack([alpha,alpha]))
    np.testing.assert_allclose(raw['query_observation'][:n,336],alpha,atol=1e-6,rtol=0)
    np.testing.assert_allclose(raw['latent_z'][:n][permission],raw['composed_latent'][:n][permission],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(raw['latent_z'][:n][~permission],0.)
    held=('alpha_exec','rule_latched','query_observation','composed_latent','reference_target_permitted')+tuple(k for k in raw if k.startswith('coord_'))
    for start in range(0,n,4):
        for field in held:
            x=np.asarray(raw[field])[start:min(start+4,n)]
            if not np.isfinite(x).all():raise ValueError('nonfinite coordinator log '+field)
            np.testing.assert_array_equal(x,np.broadcast_to(x[0],x.shape))
        selected=int(np.argmin(abs(np.array([0.,.5,1.])-alpha[start])))
        if not is_speedcap_case(case):np.testing.assert_allclose(raw['composed_latent'][start],raw['coord_latents'][start,selected],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(raw['coord_permission'][start],permission[start])
    return dict(actual_permitted_ticks=int(permission.sum()),masked_ticks=int((~permission).sum()),state_alpha=case.get('alpha',1.),checked='Scalar selected H75 candidate, exact block hold and native publication mask')


def audit_speedcap(raw,case,interface):
    """Independent NumPy inverse of native asymmetric permissions, no physics."""
    n=int(np.asarray(raw['active_tick'],bool).sum());z=np.asarray(raw['coord_latents'][:n,2]);nom=np.asarray(raw['coord_nominal'][:n,0]);k=np.asarray(raw['coord_max_abs_curvature'][:n])
    positive=interface['speed_positive_scale_m_s'];negative=interface['speed_negative_scale_m_s'];lo,hi=interface['speed_reference_bounds_m_s']
    a=np.tanh(z[:,0]);original=np.clip(nom+a*np.where(a>=0,positive,negative),lo,hi)
    cap=np.full(n,np.inf);np.sqrt(np.divide(3.,k,out=np.zeros(n),where=k>0),out=cap,where=k>0)
    requested=np.minimum(original,cap);desired=requested-nom;scale=np.where(desired>=0,positive,negative);bounded=np.clip(desired/scale,-1.+1e-6,1.-1e-6);reachable=np.clip(nom+bounded*scale,lo,hi)
    for field,expected in [('cap_m_s',cap),('requested_target_m_s',requested),('reachable_target_m_s',reachable),('h75_target_m_s',original)]:
        np.testing.assert_allclose(raw['speedcap_'+field][:n],expected,atol=3e-5,rtol=3e-5)
    permission=np.asarray(raw['reference_target_permitted'][:n],bool);active=(is_speedcap_case(case))&permission&(abs(reachable-original)>1e-6)
    np.testing.assert_array_equal(raw['speedcap_active'][:n],active)
    if is_speedcap_case(case):
        np.testing.assert_allclose(raw['composed_latent'][:n,0],np.arctanh(bounded),atol=2e-3,rtol=3e-5)
        np.testing.assert_allclose(raw['composed_latent'][:n,1],z[:,1],atol=1e-6,rtol=0)
    return dict(active_ticks=int(active.sum()),observed_ticks=n,permitted_ticks=int(permission.sum()),actual_activity_fraction=float(active.mean()),active_duration_s=float(active.sum()*.005),maximum_target_reduction_m_s=float(np.max(original-reachable)),filtered_reference='native governed log, correction chain audited; never directly overwritten')


def tolerance_numpy_probabilities(params,obs,latents,yaw,interface,arm):
    """Independent fixed41-feature/three-sigmoid-mean inference; no labels."""
    a=np.tanh(latents).astype(np.float32);offsets=np.stack([a[...,0]*np.where(a[...,0]>=0,interface['speed_positive_scale_m_s'],interface['speed_negative_scale_m_s']),a[...,1]*interface['steer_scale_rad']],axis=-1).astype(np.float32)
    features=np.empty((len(obs),3,4,41),np.float32)
    indices=list(range(300,320))+list(range(337,351));features[...,:34]=obs[:,None,None,indices]
    features[...,34]=np.array([0.,.5,1.])[None,:,None];features[...,35]=offsets[:,:,0,None];features[...,36]=offsets[:,:,1,None]/.2
    force=np.array([-1.,1.,-1.,1.],np.float32)
    features[...,37]=np.sin(yaw)[:,None,None]*force;features[...,38]=np.cos(yaw)[:,None,None]*force
    features[...,39]=1.;features[...,40]=np.array([.2,.2,.6,.6],np.float32)/.6
    x=features.reshape(-1,41);probs=[]
    for seed in range(3):
        h=np.maximum(x@params['w0'][arm,seed]+params['b0'][arm,seed],0.)
        h=np.maximum(h@params['w1'][arm,seed]+params['b1'][arm,seed],0.)
        logit=(h@params['w2'][arm,seed]+params['b2'][arm,seed])[:,0]
        probs.append(1./(1.+np.exp(-np.clip(logit,-80.,80.))))
    return np.mean(probs,axis=0).reshape(len(obs),3,4),offsets


def tolerance_alpha_statistics(raw,user_alpha=1.,controller_uses_risk_gate=True):
    n=int(np.asarray(raw['active_tick'],bool).sum());alpha=np.asarray(raw['alpha_exec'][:n]);previous=np.r_[user_alpha,alpha[:-1]]
    departures=np.flatnonzero((previous==user_alpha)&(alpha!=user_alpha));returns=np.flatnonzero((previous!=user_alpha)&(alpha==user_alpha));switch=np.flatnonzero(previous!=alpha)
    result=dict(observed_s=n*.005,ever_departed=bool(len(departures)),ever_returned_after_departure=bool(len(returns)),final_returned_to_user=bool(len(departures) and alpha[-1]==user_alpha),final_at_user=bool(alpha[-1]==user_alpha),final_alpha=float(alpha[-1]),switch_count=len(switch),switch_times_s=(switch*.005).tolist(),departure_times_s=(departures*.005).tolist(),return_times_s=(returns*.005).tolist(),nonuser_duration_s=float(np.sum(alpha!=user_alpha)*.005),degraded_duration_s=float(np.sum(raw['coord_degraded'][:n])*.005),permission_withheld_duration_s=float(np.sum(raw['coord_publication_withheld'][:n])*.005))


    # Being executed at the user's alpha is distinct from satisfying the
    # risk/task/publication gate that would restore that alpha normally.
    result['user_execution_under_degraded_s']=float(np.sum((alpha==user_alpha)&np.asarray(raw['coord_degraded'][:n],bool))*.005)
    required=('coord_latched_before','coord_qmin','coord_task_mask','coord_permission')
    available=all(k in raw for k in required)
    result.update(user_restoration_gate_observed=available,final_user_qmin=None,final_user_restore_threshold=None,final_user_task_feasible=None,final_publication_permitted=None,final_risk_gate_satisfied=None,final_qualified_user_restoration=None,fallback_selected_user_without_restore_s=None)
    if available:
        ui=int(round(user_alpha*2));q=np.asarray(raw['coord_qmin'][:n])[:,ui]
        threshold=np.where(np.asarray(raw['coord_latched_before'][:n],bool),.85,.8)
        task=np.asarray(raw['coord_task_mask'][:n],bool)[:,ui];permission=np.asarray(raw['coord_permission'][:n],bool)
        gate=(q>=threshold)&task&permission
        result.update(final_user_qmin=float(q[-1]),final_user_restore_threshold=float(threshold[-1]),final_user_task_feasible=bool(task[-1]),final_publication_permitted=bool(permission[-1]),final_risk_gate_satisfied=bool(gate[-1]),final_qualified_user_restoration=bool(alpha[-1]==user_alpha and gate[-1]))
        if 'coord_task_aware_selected' in raw:
            selected=np.asarray(raw['coord_task_aware_selected'][:n],bool)
            result['fallback_selected_user_without_restore_s']=float(np.sum(selected&(alpha==user_alpha)&~gate)*.005)
    if available and 'coord_guard_fallback_selected' in raw:
        selected=np.asarray(raw['coord_guard_fallback_selected'][:n],bool)
        result['guard_fallback_selected_user_without_restore_s']=float(np.sum(selected&(alpha==user_alpha)&~gate)*.005)
    result['restoration_gate_used_by_controller']=controller_uses_risk_gate
    if not controller_uses_risk_gate:
        result['diagnostic_only_final_risk_gate_satisfied']=result['final_risk_gate_satisfied']
        result['diagnostic_only_final_user_at_risk_gate']=result['final_qualified_user_restoration']
        result['final_risk_gate_satisfied']=None;result['final_qualified_user_restoration']=None
    return result


def audit_tolerance_coordination(raw,case,original,replanned,interface,params,arm_index=None):
    n=int(np.asarray(raw['active_tick'],bool).sum());starts=np.arange(0,n,4);mode=case['mode'];mode=mode.replace('_speedquery','_taskaware');mode=('user0_fixed' if case.get('method')=='fixed' else 'user0_guard' if case.get('method')=='guard' else 'speedcap' if case.get('method')=='speedcap' else mode);arm=arm_index if arm_index is not None else 0 if mode=='q_bce' else 1
    if mode in TEMPORAL_TOLERANCE_ARMS+USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS+GUARD_FALLBACK_ARMS and arm_index is None:raise ValueError('temporal predictor requires explicit bundle parameter index')
    prob,offsets=tolerance_numpy_probabilities(params,np.asarray(raw['query_observation'])[starts],np.asarray(raw['coord_latents'])[starts],np.asarray(raw['pre_upper_yaw'])[starts],interface,arm)
    np.testing.assert_allclose(raw['coord_probabilities'][starts],prob,atol=3e-5,rtol=3e-5)
    qmin=prob.min(axis=-1);np.testing.assert_allclose(raw['coord_qmin'][starts],qmin,atol=3e-5,rtol=3e-5)
    user=case.get('effective_user_alpha',1.);ui=int(round(user*2));others=np.array([a for a in range(3) if a!=ui]);user_mode=mode in USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS
    prev=user;latched=False;path=original;maxprob=float(np.max(abs(prob-raw['coord_probabilities'][starts])))
    for j,tick in enumerate(starts):
        if replanned is not None and case.get('replan_task_tick')==tick:path=replanned
        progress=0. if tick==0 or case.get('replan_task_tick')==tick else float(raw['path_progress'][tick-1])
        np.testing.assert_allclose(raw['coord_pre_progress'][tick],progress,atol=1e-4,rtol=1e-6)
        np.testing.assert_allclose(raw['coord_prev_alpha'][tick],prev,atol=1e-7,rtol=0);assert bool(raw['coord_latched_before'][tick])==latched
        nominal=np.asarray(raw['coord_nominal'][tick]);unbounded=nominal[None,:]+offsets[j]
        lo,hi=interface['speed_reference_bounds_m_s'];steer=interface['steer_reference_bound_rad'];targets=np.clip(unbounded,np.array([lo,-steer],np.float32),np.array([hi,steer],np.float32))
        np.testing.assert_allclose(raw['coord_targets'][tick],targets,atol=3e-6,rtol=2e-6)
        np.testing.assert_allclose(raw['coord_unbounded_targets'][tick],unbounded,atol=3e-6,rtol=2e-6)
        np.testing.assert_array_equal(raw['coord_candidate_clipped'][tick],targets!=unbounded)
        def preview(length):
            lower=np.clip(progress,path['s'][0],path['s'][-1]);upper=min(lower+length,path['s'][-1]);mask=(path['s']>=lower)&(path['s']<=upper)
            return max(abs(np.interp(lower,path['s'],path['curvature'])),abs(np.interp(upper,path['s'],path['curvature'])),float(np.max(abs(path['curvature'][mask]))) if mask.any() else 0.)
        curvature=preview(6.);np.testing.assert_allclose(raw['coord_max_abs_curvature'][tick],curvature,atol=2e-5,rtol=2e-5)
        taskmask=targets[:,0]**2*curvature<=3.;np.testing.assert_array_equal(raw['coord_task_mask'][tick],taskmask)
        cost=(.1+.9*user)*(targets[:,0]-case['speed'])**2+(1.-.9*user)*((targets[:,1]-nominal[1])/.2)**2+.05*(np.array([0.,.5,1.])-user)**2+.02*(np.array([0.,.5,1.])-prev)**2
        np.testing.assert_allclose(raw['coord_proxy_cost'][tick],cost,atol=3e-5,rtol=3e-5)
        useq=mode!='task_guard_only' and not mode.endswith('_guard');restore=bool(taskmask[ui] and (not useq or qmin[j,ui]>=(.85 if latched else .8)))
        eligible=taskmask[others] & ((qmin[j,others]>=.8) if useq else True);degraded=not restore and not eligible.any()
        proposed=user if restore else np.array([0.,.5,1.])[others[np.argmin(np.where(eligible,cost[others],np.inf))]] if eligible.any() else .5
        permission=bool(raw['coord_permission'][tick])
        if mode in TASK_AWARE_ARMS:
            np.testing.assert_array_equal(raw['coord_task_aware_fallback'][tick],True)
            selected=degraded and bool(taskmask.any())
            if selected:
                candidates=np.flatnonzero(taskmask);best=candidates[np.lexsort((candidates,cost[candidates],-qmin[j,candidates]))[0]];proposed=float(np.array([0.,.5,1.])[best])
            np.testing.assert_array_equal(raw['coord_task_aware_selected'][tick],selected and permission)
        if mode in GUARD_FALLBACK_ARMS:
            np.testing.assert_array_equal(raw['coord_guard_fallback'][tick],True)
            selected=degraded and bool(taskmask.any())
            if selected:proposed=user if taskmask[ui] else float(np.array([0.,.5,1.])[np.argmin(np.where(taskmask,cost,np.inf))])
            np.testing.assert_array_equal(raw['coord_guard_fallback_selected'][tick],selected and permission)
            np.testing.assert_array_equal(raw['coord_task_aware_selected'][tick],False)
        np.testing.assert_array_equal(raw['coord_degraded'][tick],permission and degraded and mode in TOLERANCE_ARMS[:3]+TEMPORAL_TOLERANCE_ARMS+tuple(a for a in USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS+GUARD_FALLBACK_ARMS if not a.endswith('_fixed')))
        np.testing.assert_array_equal(raw['coord_publication_withheld'][tick],not permission)
        if mode=='existing_geometry':
            proxy=case['speed']**2*preview(1.5);nextlatch=proxy>=2.7 if latched else proxy>3.;proposed=.5 if nextlatch else 1.
        elif mode=='user0_fixed':proposed=user;nextlatch=False
        elif mode in ('fixed_half','fixed1','speedcap'):proposed=.5 if mode=='fixed_half' else 1.;nextlatch=False
        else:nextlatch=not restore
        expected=proposed if permission else prev;nextlatch=nextlatch if permission else latched
        np.testing.assert_allclose(raw['alpha_exec'][tick],expected,atol=1e-7,rtol=0);assert bool(raw['rule_latched'][tick])==nextlatch
        prev=float(expected);latched=bool(nextlatch)
    result=tolerance_alpha_statistics(raw,user,controller_uses_risk_gate=case.get('method') not in ('guard','fixed','speedcap'));result.update(maximum_independent_probability_error=maxprob,independent_numpy_decisions_match=True,inputs='Current observation/three H75 candidates/nominal/execution path+progress/original speed request and declared user alpha/permission and previous latch/alpha; fixed four hypothetical forces/preparations only. No actual force/onset/route/time in decision.',degraded_is_not_safety=True)
    return result


def tolerance_common_windows(rows,raws,paths,goals):
    """Common actual H10 and common pre-first-arrival windows; no new cost weights."""
    arms=tolerance_comparison_arms(rows)
    for label in response_plot_groups(rows):
        group=response_plot_selection(rows,label)
        if len(group)!=len(arms) or {r['mode'] for r in group}!=set(arms):raise ValueError('complete matching original-condition arms required')
        common=min(r['active_lower_ticks'] for r in group);prefix=min(r['prefix']['ticks'] for r in group)
        for row in group:
            raw=raws[row['case_id']];path=paths[analysis_path_key(row)];tree=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[analysis_path_key(row)]]))
            metrics={}
            for name,n in (('same_actual_window',common),('same_pre_first_arrival_window',prefix)):
                ev=raw['actual_forward_speed'][:n]-row['speed'];distance=tree.query(raw['actual_xy'][:n])[0]
                metrics[name]=dict(actual_ticks=n,observed_s=n*.005,speed_rmse_m_s=float(np.sqrt(np.mean(ev**2))) if n else None,original_route_sample_distance_rmse_m=float(np.sqrt(np.mean(distance**2))) if n else None,original_route_sample_distance_max_m=float(distance.max()) if n else None)
            row['matched_original_task_metrics']=metrics
            row['matched_window_methods']=list(arms);row['matched_window_method_count']=len(arms)
            n=row['active_lower_ticks'];time=np.asarray(raw['time'][:n]);phases={}
            for name,mask in (('before',time<row['onset_s']),('during',(time>=row['onset_s'])&(time<row['onset_s']+row['duration_s'])),('after',time>=row['onset_s']+row['duration_s'])):
                phases[name]=dict(actual_ticks=int(mask.sum()),speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed'][:n][mask]-row['speed'])**2))) if mask.any() else None,execution_cross_track_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track'][:n][mask]**2))) if mask.any() else None)
            row['observed_phase_metrics']=phases


def original_path_projection_trace(raw,path,config):
    """NumPy equivalent of frozen project(), retaining original progress state."""
    s=np.asarray(path['s'],np.float32);xy=np.asarray(path['xy'],np.float32);heading=np.asarray(path['heading'],np.float32)
    n=int(np.asarray(raw['active_tick'],bool).sum());actual=np.asarray(raw['actual_xy'][:n],np.float32);c=config['path']
    progress=np.float32(0.);previous=xy[0];values=np.empty((n,3),np.float32)
    # Search bounds select the same valid segments as the native full-array mask.
    for tick,point in enumerate(actual):
        distance=np.linalg.norm(point-previous);lo=np.maximum(np.float32(0.),progress-np.float32(c['projection_backward_m']))
        hi=np.minimum(s[-1],progress+np.maximum(np.float32(c['projection_forward_floor_m']),np.float32(c['projection_forward_motion_multiplier'])*distance+np.float32(c['projection_forward_padding_m'])))
        first=max(0,int(np.searchsorted(s,lo,side='left'))-1);last=min(len(s)-1,int(np.searchsorted(s,hi,side='right')))
        a=xy[first:last];v=xy[first+1:last+1]-a;span=s[first+1:last+1]-s[first:last]
        u=np.sum((point-a)*v,axis=-1)/np.sum(v*v,axis=-1)
        u=np.clip(u,np.clip((lo-s[first:last])/span,0.,1.),np.clip((hi-s[first:last])/span,0.,1.))
        ss=s[first:last]+u*span;points=a+u[:,None]*v;d2=np.sum((point-points)**2,axis=-1)
        idx=int(np.argmin(np.where(d2<=d2.min()+np.float32(1e-10),abs(ss-progress),np.float32(np.inf))))
        h=np.float32(np.interp(ss[idx],s,heading));delta=point-points[idx]
        cross=-np.sin(h)*delta[0]+np.cos(h)*delta[1]
        yaw=np.float32(raw['reference_yaw_unwrapped'][tick])-np.float32(raw['e_psi_unwrapped'][tick]);angle=h-yaw
        error=np.arctan2(np.sin(angle),np.cos(angle));progress=np.maximum(progress,ss[idx]);values[tick]=[progress,cross,error];previous=point
    return values



def adjacent_projection_roundoff(path,point,reconstructed_progress,logged_progress):
    """Certify a local float32 adjacent-segment near-tie; never a path/time repair."""
    s=np.asarray(path['s'],np.float32);xy=np.asarray(path['xy'],np.float32);point=np.asarray(point,np.float32)
    segments=[int(np.clip(np.searchsorted(s,value,side='right')-1,0,len(s)-2)) for value in (reconstructed_progress,logged_progress)]
    if abs(segments[0]-segments[1])!=1:return None
    records=[]
    for i,value in zip(segments,(reconstructed_progress,logged_progress)):
        a=xy[i];v=xy[i+1]-a;u=np.clip(np.sum((point-a)*v)/np.sum(v*v),np.float32(0.),np.float32(1.));ss=s[i]+u*(s[i+1]-s[i]);delta=point-(a+u*v);d2=np.sum(delta*delta)
        if abs(float(ss)-float(value))>2e-5:return None
        records.append(dict(segment=i,projected_progress_m=float(ss),squared_distance_m2=float(d2)))
    d2=np.float32(min(r['squared_distance_m2'] for r in records));ulp=float(abs(np.spacing(d2)));gap=abs(records[0]['squared_distance_m2']-records[1]['squared_distance_m2']);bound=min(1e-6,max(1e-10,8.*ulp))
    if gap>bound:return None
    return dict(segments=records,squared_distance_gap_m2=gap,float32_ulp_m2=ulp,gap_ulps=gap/ulp if ulp else 0.,maximum_gap_m2=bound)


def validate_original_projection(projection,raw,path,before,*,allow_adjacent_roundoff=False):
    columns=('path_progress','path_cross_track','path_heading_error');logged=np.column_stack([np.asarray(raw[k])[:before] for k in columns]);actual=projection[:before]
    close=np.isclose(actual,logged,atol=4e-4,rtol=2e-5);eligible=np.zeros(before,bool);details=[]
    for tick in np.flatnonzero(~close.all(axis=1)):
        error=abs(actual[tick]-logged[tick])
        proof=adjacent_projection_roundoff(path,raw['actual_xy'][tick],actual[tick,0],logged[tick,0]) if allow_adjacent_roundoff and error[0]<=.01 and error[2]<=.005 and close[tick,1] else None
        if proof is None:raise AssertionError(f'original projection before replan differs outside bounded adjacent float32 tie at tick{tick}: {error.tolist()}')
        eligible[tick]=True;details.append(dict(tick=int(tick),time_s=int(tick)*.005,absolute_errors=error.tolist(),**proof))
    if len(details)>4:raise AssertionError('more than4 bounded projection tie frames in one case')
    validation={name+'_max_absolute_error':float(np.max(abs(actual[:,i]-logged[:,i]))) if before else None for i,name in enumerate(('progress','cross_track','heading_error'))}
    validation.update(adjacent_float32_roundoff_enabled=bool(allow_adjacent_roundoff),adjacent_float32_roundoff_frames=len(details),adjacent_float32_roundoff_details=details,adjacent_roundoff_progress_bound_m=.01,adjacent_roundoff_heading_bound_rad=.005,other_frames_original_projection_atol=4e-4,other_frames_original_projection_rtol=2e-5)
    return validation,eligible

def rescore_original_task(raw,path,config,controller,keys,replan_tick=None,*,broaden_cost_numerics=False,task_alpha=1.,raw_alpha=1.):
    """Only original alpha1 V3 task channels; never a reconstructed total reward."""
    from analyze_native_frozen_validation import score
    n=int(np.asarray(raw['active_tick'],bool).sum());projection=original_path_projection_trace(raw,path,config)
    modified={k:np.asarray(v)[:n] for k,v in raw.items()}
    modified.update(path_progress=projection[:,0],path_cross_track=projection[:,1],path_heading_error=projection[:,2],physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool))
    _,data=score(modified,path,config,controller,keys,task_alpha)
    indices=[keys.index(k) for k in ('speed','path','primary')];effective=data['effective'][:,indices]
    before=n if replan_tick is None else min(n,int(replan_tick));validation={}
    validation,tie_frames=validate_original_projection(projection,raw,path,before,allow_adjacent_roundoff=broaden_cost_numerics)
    logged=np.asarray(raw['effective_costs'])[:before,indices]
    if task_alpha!=raw_alpha:
        _,rescaled=score(dict(modified,path_progress=np.asarray(raw['path_progress'])[:n],path_cross_track=np.asarray(raw['path_cross_track'])[:n],path_heading_error=np.asarray(raw['path_heading_error'])[:n]),path,config,controller,keys,task_alpha)
        logged=rescaled['effective'][:before,indices]
    if not np.isfinite(effective).all() or not np.isfinite(logged).all():raise ValueError('nonfinite original task costs')
    error=effective[:before]-logged
    validation['task_cost_max_absolute_error']=float(np.max(abs(error))) if before else None
    validation['task_cost_outside_old_tolerance_elements']=int(np.sum(~np.isclose(effective[:before],logged,atol=4e-4,rtol=2e-5)))
    validation['task_cost_max_relative_error']=float(np.max(abs(error)/np.maximum(abs(logged),1e-8))) if before else None
    validation['broaden_float32_cost_tolerance']=bool(broaden_cost_numerics)
    validation['task_cost_absolute_tolerance']=.003 if broaden_cost_numerics else 4e-4
    validation['task_cost_relative_tolerance']=2e-4 if broaden_cost_numerics else 2e-5
    validation['pre_replan_integrated_three_cost_difference']=(error.sum(axis=0)*config['path_reward']['scale']*config['timing']['lower_dt_s']).tolist()
    # Broaden only: float32 bounded projection errors amplify in squared costs.
    # Keep a finite absolute/relative bound and report the full accumulated effect.
    np.testing.assert_allclose(effective[:before][~tie_frames],logged[~tie_frames],atol=validation['task_cost_absolute_tolerance'],rtol=validation['task_cost_relative_tolerance'],err_msg='original V3 task costs before replan differ')
    if tie_frames.any():
        # Only geometrically certified frames may use these finite rate bounds;
        # retain independently reconstructed values and report integrated impact.
        bounds=np.array([.3,.001,5.],float)
        if np.any(abs(error[tie_frames])>bounds):raise AssertionError('adjacent-tie task cost rate exceeds finite component bounds')
        impact=error[tie_frames].sum(axis=0)*config['path_reward']['scale']*config['timing']['lower_dt_s']
        if np.any(abs(impact)>.005):raise AssertionError('adjacent-tie integrated task cost difference exceeds.005 per component')
        validation.update(adjacent_roundoff_cost_rate_bounds=bounds.tolist(),adjacent_roundoff_cost_max_absolute_error=np.max(abs(error[tie_frames]),axis=0).tolist(),adjacent_roundoff_integrated_cost_difference=impact.tolist(),adjacent_roundoff_integrated_component_bound=.005)

    step=effective*config['path_reward']['scale']*config['timing']['lower_dt_s']
    return dict(original_progress_m=projection[:,0],original_cross_track_m=projection[:,1],original_heading_error_rad=projection[:,2],task_effective_cost_rates=effective,task_step_costs=step,cumulative_task_costs=np.cumsum(step,axis=0)),dict(validated_pre_replan_ticks=before,projection_errors=validation,task_component_names=['speed','path','primary'],terminal_replacement='Excluded from three running task channels; actual global failure replacement remains in original full reward diagnostics. Not allocated to a task channel.',full_total_reward_reconstructed=False)


def rescore_original_task_analysis(root):
    p=json.loads((root/'plan.json').read_text())
    if p['mode'] not in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):raise ValueError('original task rescoring requires a complete tolerance online analysis')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('plan drift')
    rows=json.loads((root/'analysis/results.json').read_text());index=json.loads((root/'evaluation/case_index.json').read_text())
    if p['mode']=='tolerance_online40':
        cache_source=verify_online40_comparison_source(root,p)
        index+=json.loads((cache_source/'evaluation/case_index.json').read_text())
    if p['mode'] in ('user_prior_online72','task_aware_fallback72'):
        source=Path(p['comparison_source']['source']);parents=json.loads((source/'analysis/results.json').read_text())
        index += [{k:r[k] for k in ('case_id','raw_file','raw_sha256','effective_user_alpha','method')} for r in parents]
    if p['mode'] in ('speed_query_online60','guard_fallback_evaluation'):
        source=Path(p['comparison_source']['source']);parents=json.loads((source/'evaluation/case_index.json').read_text());index+=parents
    if p['mode']=='user_preference120':
        source=Path(p['comparison_source']['source']);parents=json.loads((source/'evaluation/case_index.json').read_text());lookup={r['case_id']:r for r in parents}
        index+=[dict(lookup[r['cached_parent_case_id']],case_id=r['case_id']) for r in rows if r.get('source_kind') in ('cached_broaden','cached_preference','cached_prior','cached_frozen200','cached_guard_source')]
    if len(rows)!=len(index) or [r['case_id'] for r in rows]!=[r['case_id'] for r in index]:raise ValueError('complete authenticated main comparison analysis required')
    manifest=json.loads((root/'evaluation/manifest.json').read_text())
    for key,expected in (('plan_sha256',digest(root/'plan.json')),('config',p['config']),('models',p['models'])):
        if manifest[key]!=expected:raise ValueError('original rescore manifest drift '+key)
    if json.loads((root/'analysis/receipt.json').read_text())['plan_sha256']!=digest(root/'plan.json'):raise ValueError('main analysis belongs to another plan')
    raw={};paths={}
    for row,declared in zip(rows,index):
        if any(row.get(k)!=v for k,v in declared.items()):raise ValueError('main analysis case differs from authenticated raw index')
        cached_verified=p['mode']=='tolerance_online40' and row.get('source_kind')=='cached120' and p['comparison_source']['identity'].get(row['raw_file'])==row['raw_sha256']
        if not cached_verified and digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('raw trace drift')
        with np.load(row['raw_file']) as z:raw[row['case_id']]=dict(z)
        key=analysis_path_key(row)
        if key not in paths:
            with np.load(root/'evaluation'/(key+'_path.npz')) as z:paths[key]=dict(z)
    manifest=json.loads((root/'evaluation/manifest.json').read_text())
    return write_original_task_rescoring(root,rows,raw,paths,p['config'],p['initial_source']['controller'],manifest['reward_components'])


def write_original_task_rescoring(root,rows,raws,paths,config,controller,keys):
    """Add separate derived three-channel evidence; no physics or endpoint change."""
    import time
    started=time.monotonic()
    out=root/'analysis'/'original_task_rescoring';out.mkdir(exist_ok=False);results=[];arrays={}
    descriptive=json.loads((root/'plan.json').read_text()).get('mode') in ('tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation')
    for lane,row in enumerate(rows):
        values,receipt=rescore_original_task(raws[row['case_id']],paths[analysis_path_key(row)],config,controller,keys,row.get('replan_task_tick'),broaden_cost_numerics=descriptive,task_alpha=row.get('effective_user_alpha',1.),raw_alpha=row.get('raw_user_alpha',row.get('alpha',1.)))
        arrays[row['case_id']]=values;np.savez_compressed(out/f'case_{lane:03d}.npz',**values)
        results.append(dict(case_id=row['case_id'],bank_index=row['bank_index'],route=row['route'],event=row['event'],mode=row['mode'],effective_user_alpha=row.get('effective_user_alpha',1.),raw_user_alpha=row.get('raw_user_alpha',row.get('alpha',1.)),actual_ticks=row['active_lower_ticks'],actual_physical_outcome=row['horizons']['10']['physical_outcome'],data_file=f'case_{lane:03d}.npz',**receipt))
    for label in response_plot_groups(rows):
        group=response_plot_selection(rows,label);common=min(r['active_lower_ticks'] for r in group);prefix=min(r['prefix']['ticks'] for r in group)
        for row in group:
            record=next(r for r in results if r['case_id']==row['case_id']);step=arrays[row['case_id']]['task_step_costs'];record['windows']={}
            for name,n in (('actual_observed',len(step)),('own_pre_first_arrival',row['prefix']['ticks']),('common_actual',common),('common_pre_first_arrival',prefix)):
                record['windows'][name]=dict(actual_ticks=n,observed_s=n*.005,three_task_costs=step[:n].sum(axis=0).tolist(),three_task_cost_sum=float(step[:n].sum()))
    for candidate,prefix in (('speed_query','full_mask'),('guard_degraded','old_new_candidate')):
        if any(r['method']==candidate for r in rows) and any(r['method']=='task_aware' for r in rows):write_paired_candidate_costs(out,rows,results,arrays,candidate=candidate,prefix=prefix)
    atomic_json(out/'results.json',results);atomic_json(out/'receipt.json',dict(analysis_source_sha256=digest(__file__),plan_sha256=digest(root/'plan.json'),main_analysis_results_sha256=digest(root/'analysis/results.json'),original_path_sha256={key:digest(root/'evaluation'/(key+'_path.npz')) for key in paths},elapsed_s=time.monotonic()-started,cases=len(rows),new_physics_steps=0,optimizer_updates=0,task_alpha='per_row_effective_user_alpha' if any('effective_user_alpha' in r for r in rows) else 1.,projection='Frozen original world path from task0; no reset at executed-path Dubins replan; native bounded local projection/tie/monotone progress semantics.',scoring='Only frozen V3 speed/path/primary running cost channels. Other reward terms are not substituted or relabeled. Actual global terminal replacement excluded from these three channels; full observed reward retains it.',observation='Stop at actual failure/end. Common actual and common pre-first-arrival windows supplied; goal/contact unchanged. Offline original-task rescoring of actual executed trajectories, not hypothetical original-path-controlled trajectories.'))
    (out/'README.md').write_text('# Original effective-user task rescoring\n\nOnly frozen V3 speed/path/primary running costs on the immutable original path. Goal/contact outcomes and actual observation lengths take precedence over costs: early failure can shorten a common window and cannot be declared better tracking from a smaller truncated cost. Original projection state is replayed from task0 and never reset at Dubins replan. All pre-replan windows are checked against recorded native projection and task costs.\n\nActual trajectories stop at their real endpoints. Common observation/pre-first-arrival windows are provided. These three running channels exclude the global terminal reward replacement and are not a reconstructed total reward. Goal/contact outcomes are unchanged.\n\n[Per-case windows and validation](results.json) · [Scope receipt](receipt.json)\n')
    return results


def tolerance_online_summary(rows):
    arms=tolerance_comparison_arms(rows);expected=set(arms);methods={r['mode'] for r in rows}
    if methods!=expected:raise ValueError('all declared tolerance comparison arms required')
    summary={};paired=[]
    for arm in arms:
        summary[arm]={}
        for bank in sorted({r['bank_index'] for r in rows}):
            summary[arm][str(bank)]={}
            for subset in ('all','nominal','disturbed'):
                group=[r for r in rows if r['mode']==arm and r['bank_index']==bank and (subset=='all' or bool(r['force_N'])==(subset=='disturbed'))]
                summary[arm][str(bank)][subset]=dict(n=len(group),arrived=sum(r['horizons']['10']['goal_arrived'] is True for r in group),contact=sum(r['horizons']['10']['physical_outcome']=='contact' for r in group),unknown=sum(r['horizons']['10']['physical_outcome']=='unknown' for r in group),diagnostic_execution_path_user1_cost_sum=sum(r['diagnostic_execution_path_user1_cost'] for r in group))
    for candidate,baseline in itertools.combinations(arms,2):
        a={(r['bank_index'],r['route'],r['event']):r for r in rows if r['mode']==candidate};b={(r['bank_index'],r['route'],r['event']):r for r in rows if r['mode']==baseline}
        if set(a)!=set(b):raise ValueError('incomplete tolerance pairing')
        details=[]
        for key,r in a.items():
            other=b[key];x=r['horizons']['10'];y=other['horizons']['10'];known=x['goal_arrived'] is not None and y['goal_arrived'] is not None and 'unknown' not in (x['physical_outcome'],y['physical_outcome'])
            details.append(dict(bank_index=key[0],route=key[1],event=key[2],force_N=r['force_N'],candidate=r['case_id'],baseline=other['case_id'],goal1m_gain=known and x['goal_arrived'] and not y['goal_arrived'],goal1m_loss=known and y['goal_arrived'] and not x['goal_arrived'],contact_rescue=x['physical_outcome']=='survived' and y['physical_outcome']=='contact',new_contact=x['physical_outcome']=='contact' and y['physical_outcome']!='contact',diagnostic_execution_path_user1_cost_difference=r['diagnostic_execution_path_user1_cost']-other['diagnostic_execution_path_user1_cost'],full_speed_rmse_difference_m_s=r['full_observed_speed_rmse_m_s']-other['full_observed_speed_rmse_m_s'],full_original_route_distance_rmse_difference_m=r['original_route_sample_distance_rmse_m']-other['original_route_sample_distance_rmse_m'],candidate_prefix=r['prefix'],baseline_prefix=other['prefix'],candidate_matched_original_task=r.get('matched_original_task_metrics'),baseline_matched_original_task=other.get('matched_original_task_metrics')))
        paired.append(dict(candidate=candidate,baseline=baseline,details=details,nominal_no_new_contact_or_goal_loss=all(not(d['new_contact'] or d['goal1m_loss']) for d in details if not d['force_N'])))
    return dict(method_bank_subset=summary,new_cases=sum(not r.get('source_kind','').startswith('cached') for r in rows),cached_cases=sum(r.get('source_kind','').startswith('cached') for r in rows),alpha_statistics=[dict(case_id=r['case_id'],**r['alpha_rule']) for r in rows],original_path_v3_aggregate='not_yet_verified; original route distance/speed/prefix metrics provided; execution-path user1 reward remains diagnostic',adoption='not_automatically_adopted',data_role='Seen prepared ancestors/familiar routes engineering DEV; full-time q queries extrapolate task1 training states.'),paired


def verify_online40_comparison_source(root,p,verified_files=None):
    comparison=p['comparison_source'];source=Path(comparison['source']).resolve();identity=comparison['identity']
    if comparison.get('expected_cases')!=120 or comparison.get('total_records')!=160 or comparison.get('cached_methods')!=list(TOLERANCE_ARMS) or comparison.get('new_methods')!=list(TEMPORAL_TOLERANCE_ARMS) or not comparison.get('reuse_physics_only'):raise ValueError('declared40new+120cached comparison required')
    for file,value in identity.items():
        if (verified_files or {}).get(file)!=value and digest(file)!=value:raise ValueError('cached comparison identity drift '+file)
    for name in ('analysis/results.json','analysis/receipt.json','evaluation/case_index.json','plan.json','seal.json'):
        if str(source/name) not in identity:raise ValueError('cached comparison evidence not sealed '+name)
    old=json.loads((source/'plan.json').read_text())
    if old['mode']!='tolerance_online120' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed online120 cache required')
    for key in ('config','initial_source','models','contexts','horizon_s'):
        if old[key]!=p[key]:raise ValueError('cached physical/task identity mismatch '+key)
    for key in ('period_s','radius_m','departure_m','maximum_per_case'):
        if old['replanning'][key]!=p['replanning'][key]:raise ValueError('cached planner contract mismatch '+key)
    for key in ('bank_indices','bank_state_fingerprints','routes','forces_N','onsets_s','zero_onset_s','duration_s','rule'):
        if old['protocol'][key]!=p['protocol'][key]:raise ValueError('cached condition mismatch '+key)
    status=json.loads((source/'evaluation/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=120:raise ValueError('completed120 cache required')
    receipt=json.loads((source/'analysis/receipt.json').read_text())
    if receipt['plan_sha256']!=digest(source/'plan.json') or receipt['all_cases']!=120:raise ValueError('cached complete analysis receipt mismatch')
    return source


def load_online40_comparison_cache(root,p,paths,keys,verified_files=None):
    """Reuse authenticated physical traces; only derived diagnostics recomputed."""
    import copy
    source=verify_online40_comparison_source(root,p,verified_files=verified_files);identity=p['comparison_source']['identity']
    old=json.loads((source/'plan.json').read_text());old_map=old['frozen_predictor_export']['arm_to_export_index']
    if old_map!={'q_rank':1,'q_bce':0}:raise ValueError('cached q_rank/BCE parameter identity changed')
    old_q_receipt=source/'q/receipt.json'
    if str(old_q_receipt) not in identity or json.loads(old_q_receipt.read_text())['arm_order']!=['bce','bce_rank']:raise ValueError('cached original predictor bundle identity missing')
    index=json.loads((source/'evaluation/case_index.json').read_text());rows=copy.deepcopy(json.loads((source/'analysis/results.json').read_text()))
    if len(rows)!=120 or len(index)!=120 or [r['case_id'] for r in rows]!=[r['case_id'] for r in index]:raise ValueError('all120 cached rows required')
    if {r['mode'] for r in rows}!=set(TOLERANCE_ARMS):raise ValueError('all six cached baseline methods required')
    for key,path in paths.items():
        file=source/'evaluation'/(key+'_path.npz')
        if str(file) not in identity:raise ValueError('cached original path not sealed')
        with np.load(file) as z:
            for field in ('s','xy','heading','curvature','turn_end','goal'):np.testing.assert_array_equal(z[field],path[field])
    raws={};executed={};rebuilt={}
    for lane,(row,declared) in enumerate(zip(rows,index)):
        if any(row.get(k)!=v for k,v in declared.items()):raise ValueError('cached analysis/index identity mismatch')
        if not row.get('alpha_rule',{}).get('independent_numpy_decisions_match'):raise ValueError('cached original q decisions were not independently verified')
        file=row['raw_file']
        if identity.get(file)!=row['raw_sha256']:raise ValueError('cached raw not sealed or changed')
        with np.load(file) as z:raw=dict(z)
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum())
        if n!=row['active_lower_ticks'] or not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('cached actual endpoint mismatch')
        changed=None;tick=row['replan_task_tick'];pathfile=source/'evaluation'/f'replanned_path_{lane:02d}.npz'
        if tick is not None:
            if str(pathfile) not in identity:raise ValueError('cached executed path not sealed')
            with np.load(pathfile) as z:changed=dict(z)
            executed[row['case_id']]=changed
        data=reconstruct_execution_path_reward(raw,row,paths[analysis_path_key(row)],changed,n if tick is None else tick,p['config'],p['initial_source']['controller'],keys)
        raws[row['case_id']]=raw;rebuilt[row['case_id']]=data
        row.update(source_kind='cached120',cache_source=str(source),newly_simulated_lower_ticks=0,diagnostic_file=str(source/'analysis'/f'case_{lane:02d}_diagnostics.npz'))
    return dict(rows=rows,raws=raws,executed=executed,rebuilt=rebuilt)


def analyze_response(root,p):
    run=root/'evaluation';out=root/'analysis';status=json.loads((run/'status.json').read_text())
    speed_comparison=p.get('mode')=='speed_query_online60'
    guard_comparison=p.get('mode')=='guard_fallback_evaluation'
    guard_layout=guard_fallback_layout(p['protocol']) if guard_comparison else None
    frozen=p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation')
    frozen_layout=frozen_confirmation_layout(p['protocol']) if frozen else None
    fallback=p.get('mode')=='task_aware_fallback72'
    prior=p.get('mode') in ('user_prior_online72','task_aware_fallback72')
    preference=p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation')
    broaden=p.get('mode')=='tolerance_broaden120'
    temporal=p.get('mode')=='tolerance_online40'
    tolerance=p.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation')
    confirmation=p.get('mode') in ('h75_online_confirmation100','tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation')
    online=p.get('mode') in ('h75_online_scalar','h75_online_confirmation100','tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation')
    continuation=p.get('mode')=='h75_switching_continuation'
    switching=p.get('mode') in ('h75_channel_switching','h75_switching_continuation');count=guard_layout['new_cases'] if guard_comparison else 60 if speed_comparison else frozen_layout['cases'] if frozen else 72 if prior else 40 if temporal else 120 if tolerance else 100 if confirmation else 18 if online else 36 if switching else 54
    inventory=json.loads((root/'inventory.json').read_text());index=json.loads((run/'case_index.json').read_text())
    if p['horizon_s']!=10. or p['cases']!=count or len(inventory)!=count or len(index)!=count or status['phase']!='completed' or status['completed_cases']!=count:
        raise ValueError(f'complete declared{count} H10 cases required')
    if len({c['case_id'] for c in index})!=count:raise ValueError('duplicate case identity')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('plan drift')
    for file,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(file)!=sha:raise ValueError('executed source/input drift '+file)
    manifest=json.loads((run/'manifest.json').read_text())
    for key,expected in (('plan_sha256',digest(root/'plan.json')),('config',p['config']),('models',p['models']),('cases',inventory),('source',p['initial_source'])):
        if manifest[key]!=expected:raise ValueError('manifest identity mismatch '+key)
    # Exactly three methods per route/event; nominal lanes are new physical trajectories.
    if broaden:
        if p.get('comparison_source'):raise ValueError('broaden120 cannot reuse old physics')
        if tuple(p['protocol']['arms'])!=BROADEN_TOLERANCE_ARMS or p['protocol']['bank_indices']!=[4,6]:raise ValueError('broaden120 arm/bank identity mismatch')
    if guard_comparison:validate_guard_fallback_inventory(inventory,p['protocol'])
    elif speed_comparison:validate_speed_query_inventory(inventory,p['protocol'])
    elif frozen:validate_frozen_confirmation_inventory(inventory,p['protocol'])
    elif prior:validate_prior_inventory(inventory,p['protocol'],task_aware=fallback)
    elif preference:validate_preference_inventory(inventory,p['protocol'])
    elif confirmation:validate_confirmation_inventory(inventory,p['protocol'])
    elif online:validate_online_inventory(inventory,p['protocol'])
    elif switching:validate_switching_inventory(inventory,p['protocol'])
    else:validate_response_inventory(inventory,p['protocol'])
    expected={(analysis_path_key(r),r['event'],response_method(r),r.get('effective_user_alpha')) for r in inventory}
    if len(expected)!=count or len({r['route'] for r in inventory})!=(guard_layout['routes'] if guard_comparison else frozen_layout['routes'] if frozen else 2 if speed_comparison else 4 if broaden or preference else 2) or sum(not r['force_N'] for r in inventory)!=(guard_layout['nominal_new_cases'] if guard_comparison else 12 if speed_comparison else frozen_layout['nominal_cases'] if frozen else 24 if prior else 40 if broaden or preference else 8 if temporal else 24 if tolerance else 20 if confirmation else 2 if online else 12 if switching else 6):
        raise ValueError('declared response inventory invalid')
    out.mkdir(exist_ok=False);helper=Path(__file__).with_name('audit_native_goal_initial.py')
    subprocess.run([sys.executable,'-B',str(helper),str(root),str(out/'initial_carry_review.json')],check=True)
    initial=json.loads((out/'initial_carry_review.json').read_text())
    paths={};goals={};trees={};raws={};rows=[];ticks=0;rebuilt={};executed={}
    cache_rows,cache_manifest,cache_initial=(load_switching_cache(root,p,initial) if (switching or online) and not confirmation else ([],None,None))
    index=index+cache_rows
    decisions=json.loads((run/'replanning_decisions.json').read_text());by_case={}
    if (switching or online) and not confirmation:decisions+=json.loads((Path(p['stay_source'])/'evaluation/replanning_decisions.json').read_text())
    selected_ids={c['case_id'] for c in index}
    decisions=[d for d in decisions if d['case_id'] in selected_ids]
    for decision in decisions:
        case=decision['case_id']
        if case in by_case:raise ValueError('more than one replan')
        tick=decision['task_tick']
        if tick not in range(200,1801,200):raise ValueError('replan outside declared1..9s boundaries')
        by_case[case]=decision
    for route in {analysis_path_key(r) for r in inventory}:
        with np.load(run/(route+'_path.npz')) as z:paths[route]=dict(z)
        path=paths[route];goals[route]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,a]) for a in (0,1)])
        if not confirmation:
            with np.load(Path(p['cached_source'])/'evaluation'/(route+'_path.npz')) as z:
                for field in ('s','xy','heading','curvature'):
                    np.testing.assert_array_equal(path[field][:len(z[field])],z[field],err_msg='original path prefix changed')
                for field in ('goal','turn_end'):np.testing.assert_array_equal(path[field],z[field])
        trees[route]=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[route]]))
    for i,c in enumerate(index):
        cached=i>=count
        lane=c.get('_cache_index',i)
        case_run=Path(p['stay_source'])/'evaluation' if cached else run
        case_manifest=cache_manifest if cached else manifest
        case_initial=cache_initial if cached else initial
        if not cached and any(c.get(k)!=v for k,v in inventory[i].items()):raise ValueError('case identity drift')
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('raw SHA drift')
        goal=goals[analysis_path_key(c)]
        np.testing.assert_allclose(c['original_goal_xy'],goal,atol=3e-6,rtol=0)
        np.testing.assert_allclose(case_manifest['original_goals_xy'][lane],goal,atol=3e-6,rtol=0)
        with np.load(c['raw_file']) as z:raw=dict(z)
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum());ticks+=n if not cached else 0
        if not 0<n<=2000 or not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('invalid actual clock prefix')
        clock=np.rint(raw['time']/.005).astype(np.int64)
        np.testing.assert_array_equal(clock[active],np.arange(n));np.testing.assert_allclose(raw['time'][active],np.arange(n)*.005,atol=3e-6,rtol=0)
        np.testing.assert_allclose(raw['v_user'][active],c['speed'],atol=2e-7,rtol=0)
        data=dict(raw,physical_tick=clock,policy_fault=raw['policy_fault']|raw['continuation_geometry_fault'])
        pulse=audit_pulse(data,dict(force_N=c['force_N'],world_direction=[0.,1.,0.],pulse_onset_task_tick=round(c['onset_s']/.005),pulse_duration_s=c['duration_s']))
        channels=audit_tolerance_publication(raw,c) if tolerance else audit_response_channels(raw,c)
        chain=verify_correction_chain(dict(data,issued_latent=raw['latent_z'],nominal_reference=raw['nominal'],reference=raw['governed'],correction_offset=raw['offsets']),p['config'],initial_filtered=case_initial['initial_filtered'][lane],initial_offsets=case_initial['initial_offsets'][lane])
        cut={k:v[:2000] for k,v in data.items()}
        endpoint=goal_arrival_metrics(cut,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=10.)
        endpoint['physical_outcome']=physical_outcome(cut['active_tick'],cut['contact_failure'],cut['policy_fault']|cut['nonfinite']|cut['lower_fault'],2000)
        fault=data['policy_fault']|raw['nonfinite']|raw['lower_fault'];fatal=np.flatnonzero(active&(raw['contact_failure']|fault))
        stop=int(fatal[0]) if len(fatal) else n;arrival=endpoint['first_arrival_s'];task_stop=min(stop,round(arrival/.005)) if arrival is not None else stop
        path=paths[analysis_path_key(c)];distance=trees[analysis_path_key(c)].query(raw['actual_xy'][:n])[0]
        replan=case_run/f'replanned_path_{lane:02d}.npz';decision=by_case.get(c['case_id']);changed=None;replan_tick=n
        if replan.exists()!=bool(c['replanned']) or bool(decision)!=bool(c['replanned']):raise ValueError('replan evidence mismatch')
        if decision:
            if decision['case_index']!=lane:raise ValueError('replan case index mismatch')
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
        rows.append(dict(c,method=(response_method(c) if online else ('stay_'+c['origin']) if cached and c.get('origin')!='background' else 'background_fixed0.5' if cached else response_method(c)),diagnostic_file=f'case_{i:02d}_diagnostics.npz',source_kind='cached36' if cached and online else 'cached18' if cached else f'new{count}' if guard_comparison else 'new60' if speed_comparison else f'new{count}' if frozen else 'new72' if prior else 'new40' if temporal else 'new120' if tolerance else 'new100' if confirmation else 'new18' if online else 'new36' if switching else 'new54',horizons={'10':endpoint},active_lower_ticks=n,pulse=pulse,onset_frame=frame,correction_chain=chain,channels=channels,
            floor_support=audit_floor_support(raw,c,p['config'],p['protocol']['existing_speed_floor_m_s'],initial['initial_nominal_reference'][lane]) if confirmation and not tolerance else None,
            prefix=dict(ticks=task_stop,observed_s=task_stop*.005,
                original_route_sample_distance_rmse_m=float(np.sqrt(np.mean(distance[:task_stop]**2))) if task_stop else None,
                speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed'][:task_stop]-c['speed'])**2))) if task_stop else None,
                path_length_m=float(np.linalg.norm(np.diff(np.vstack([path['xy'][0],raw['actual_xy'][:task_stop]]),axis=0),axis=1).sum()),
                diagnostic_execution_path_user1_cost=float(-rebuilt[c['case_id']]['user1_reward'][:task_stop].sum())),
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
        if preference:numeric.update(effective_user_reward=rebuilt[c['case_id']]['user1_reward'],effective_user_running_components=rebuilt[c['case_id']]['user1_running_components'],effective_user_cumulative_reward=np.cumsum(rebuilt[c['case_id']]['user1_reward']))
        np.savez_compressed(out/f'case_{i:02d}_diagnostics.npz',**numeric)
    if ticks-(7200 if continuation else 0)!=status['active_lower_ticks']:raise ValueError('raw tick charge mismatch')
    reused_prefix_proof=audit_continuation_prefix(root,p,rows,raws) if continuation else None
    prefix_proof=audit_switch_prefixes(rows,raws,p['protocol']) if switching else None
    if switching:
        prefix_receipt=json.loads((run/'own_switch_prefix_receipt.json').read_text())
        if not prefix_receipt.get('own_switch') or not prefix_receipt.get('discrete_exact') or (not prefix_receipt.get('all_fields_within_tolerance') and not (p['protocol']['prefix_comparison'].get('floating_descriptive_only') and prefix_receipt.get('floating_descriptive_only'))):
            raise ValueError('runner own-switch prefix receipt incomplete')
    if tolerance:
        params=load_tolerance_analysis_export(root,p)
        for row in rows:
            row['alpha_rule']=audit_tolerance_coordination(raws[row['case_id']],row,paths[analysis_path_key(row)],executed.get(row['case_id']),p['config']['residual_interface'],params,arm_index=p['frozen_predictor_export']['arm_to_export_index'].get(row['mode']))
        if temporal:
            cached=load_online40_comparison_cache(root,p,paths,manifest['reward_components'],verified_files={**p['source_sha256'],**p['input_sha256']})
            rows.extend(cached['rows']);raws.update(cached['raws']);executed.update(cached['executed']);rebuilt.update(cached['rebuilt'])
        if preference:
            if speed_comparison or guard_comparison:
                cached=load_speed_query_cache(root,p,paths,manifest['reward_components']);rows.extend(cached['rows']);raws.update(cached['raws']);executed.update(cached['executed']);rebuilt.update(cached['rebuilt'])
            elif not frozen:
                cached=(load_prior_online_cache if prior else load_user_preference_cache)(root,p,paths,manifest['reward_components']);rows.extend(cached['rows']);raws.update(cached['raws']);executed.update(cached['executed']);rebuilt.update(cached['rebuilt'])
            preference_common_windows(rows,raws,paths,goals)
            if speed_comparison:full_mask_common_windows(rows,raws,paths,goals)
            if guard_comparison:paired_candidate_common_windows(rows,raws,paths,goals,candidate='guard_degraded',metric_key='old_new_candidate_matched_original_task_metrics')
        else:tolerance_common_windows(rows,raws,paths,goals)
        if broaden or frozen or speed_comparison or guard_comparison:
            for row in rows:
                raw=raws[row['case_id']];n=row['prefix']['ticks'];a=raw['alpha_exec'][:n]
                if broaden:row['alpha_rule']['prearrival_alpha']=dict(actual_ticks=n,nonuser_ticks=int(np.sum(a!=1.)),mean=float(np.mean(a)) if n else None,final=float(a[-1]) if n else None)
                if 'speedcap_active' in raw:row['alpha_rule']['speedcap']=audit_speedcap(raw,row,p['config']['residual_interface'])
    if online:
        for row in rows:
            if row['mode']=='online_scalar':
                row['alpha_rule']=audit_online_rule(raws[row['case_id']],row,paths[analysis_path_key(row)],executed.get(row['case_id']),p['protocol']['rule'])
    summary,paired=preference_summary(rows,frozen_confirmation=frozen,speed_comparison=speed_comparison,frozen_layout=frozen_layout,guard_layout=guard_layout) if preference else tolerance_online_summary(rows) if tolerance else confirmation_summary(rows) if confirmation else online_summary(rows) if online else switching_summary(rows) if switching else response_summary(rows)
    if broaden:summary['original_path_v3_aggregate']='independently reconstructed three running task channels; common windows; terminal/global other channels excluded'
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(all_cases=len(rows),new_cases=count,cached_cases=(guard_layout['cached_cases'] if guard_comparison else 200 if speed_comparison else 0 if frozen else 288 if fallback else 216 if prior else 96 if preference else 120 if temporal else len(cache_rows)),comparison_source=p.get('comparison_source'),prefix_proof=prefix_proof,reused_prefix_proof=reused_prefix_proof,numerical_amendment=p['protocol'].get('numerical_amendment'),campaign_budget=p['protocol'].get('total_campaign_lower_ticks'),horizons_s=[10],actual_lower_ticks=ticks,cached_observed_lower_ticks=sum(r['active_lower_ticks'] for r in rows if r.get('source_kind')=='cached120' or (preference and r.get('source_kind') in ('cached_broaden','cached_preference','cached_prior','cached_frozen200','cached_guard_source'))),total_observed_lower_ticks=ticks+sum(r['active_lower_ticks'] for r in rows if r.get('source_kind')=='cached120' or (preference and r.get('source_kind') in ('cached_broaden','cached_preference','cached_prior','cached_frozen200','cached_guard_source'))),newly_simulated_lower_ticks=status['active_lower_ticks'],reused_source_lower_ticks=7200 if continuation else 0,new_physics_steps=0,optimizer_updates=0,
        plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),initial_carry_sha256=initial['initial_carry_sha256'],
        cached_q_audit=('Inherited authenticated original q/core receipts; cached rows are not re-audited with modified coordinator' if guard_comparison else 'Inherited authenticated original fullprior analysis/q receipts for200 cached rows; independently audited new60 against masked bundle only' if speed_comparison else None),
        checked='Raw SHA, full initial audit, config/model/goal/inventory, actual pulse/clock, masked channel composition, correction chain and reconstructed execution-path V3 channels/rewards.',
        limitation='Non-task reward components reused from authenticated raw logs, not independently rederived physics. Original-path distances are descriptive sampled nearest distances; execution-path V3 is not original-user utility after replan.'))
    render_response(root,rows,raws,paths,goals,executed,rebuilt,manifest['reward_components'])
    if preference:render_preference_crossuser(root,rows,raws,paths,goals,executed)
    lines=['# H75通道受扰响应：54条H10','','原始XY与执行路径优先；两tight S×三通道映射×九事件。六条无扰轨迹为本合同重新执行，不复用旧原生停止合同。',
        '','主终点：原世界目标1m首达；整10s真实触地单列。早期domain仅遥测，不算触地；重规划后不声称严格完成原S。',
        '','相关熟悉DEV、同祖先；无新训练。各法在起扰前已有不同状态，不能把转向通道替换解释为独立于速度的因果效应。']
    if confirmation:
        lines=['# 两初态五方法100回合确认','','全部100为本轮实际H10轨迹，不复用旧物理轨迹。两prepared bank不是独立训练种子；每bank/route/event五法完整原生初态配对。全部state/user/reward alpha=1，固定Actor查询alpha0/.5/1另列。',
               '','原世界goal1m与真实触地主终点；Zero+既有1.5m/s下限若未触发须标为本组等同Zero，不据此宣称排除了速度反馈支持。执行路径user1奖励仅诊断，重规划后不称原S严格完成。']
    if guard_comparison:
        lines=[f"# Guard等价退化阶段{p['protocol']['stage']}：{count}新物理＋{guard_layout['cached_cases']}原轨迹缓存",'',f"{guard_layout['conditions']}条件×3user；共{guard_layout['logical_cases']}记录。唯一改变是无jointeligible时的guard等价退化；q/阈值/风险latch/物理不变。新旧候选配对窗口与4/5方法共同窗口分开，原缓存q/core审计身份继承。反例驱动DEV修订，非独立确认；不把degraded执行用户请求称风险合格或安全。先报告goal/contact和真实失败，再解释费用；earlyfailure短窗不构成低成本胜出。"]
    elif speed_comparison:
        lines=['# 匹配输入在线机制消融：60新物理＋200冻结缓存','','候选是固定同训练配方单独训练的speed+query输入消融q，不是对full模型相同权重即时遮罩；原fullprior taskaware与guard/fixed/cap缓存保留原q审计身份、raw用户及奖励；不同q门通过仅各自模型诊断，不是共同真实安全标签。260逻辑记录，相关已见条件DEV机制消融，非新的前瞻确认。完整/遮罩两方法共同实际与首达前原任务费用单列，不以加入候选后的全方法窗口替换旧200数字。0优化更新。先看目标到达/触地/故障与真实终点，再解释费用；早失败缩短共同窗口不能据低截断费用称优。']
    elif frozen:
        lines=frozen_confirmation_intro(p['protocol'],frozen_layout)
    elif fallback:
        lines=['# 任务可行退化选择：72新物理＋288身份匹配缓存','','每用户五方法共同观察窗口，共360逻辑记录。只有无jointeligible时的退化选择改变；风险门未满足仍记degraded，不是安全保证。工程DEV，非独立测试；0优化更新。原raw与有效用户身份保持。']
    elif prior:
        lines=['# 用户条件先验：72新物理＋216身份匹配缓存','','用户0/.5/1 × 24条件 × prior_last，合并原last/guard/fixed共288逻辑记录；每用户四方法共同窗口。工程DEV，非独立测试；0优化更新。原raw、奖励与有效用户身份保持。']
    elif preference:
        lines=['# 离散用户偏好120新物理＋96缓存','','216逻辑记录：用户0/.5/1 × last/guard/fixed × 24条件。仅120新增物理；原raw用户/奖励不改，fixedhalf缓存有效用户.5独立重评分。执行α与用户α分开；不同用户加权奖励不直接排名物理表现。']
    elif temporal:
        lines=['# 时间状态混合容扰在线40新回合＋120缓存对照','','新物理40：temporal_rank_selected/temporal_rank_last，各20条件；旧六方法120条完整物理轨迹身份匹配后复用，共160记录/八方法/20条件。没有重跑旧物理；不同方法的真实失败、首达与样本分母均保留。','Selected与last为同混合rank训练的不同固定checkpoint选择，不能标成BCE两臂。原worldgoal、H75、物理、时序、执行权限、几何guard和容扰阈值不变。共同观察/首达前窗口由全部八法共同决定。']
    elif broaden:
        lines=['# 冻结last跨条件120新回合','','banks4/6 × 四路线 × 三事件 × 五臂 temporal_rank_last/task_guard_only/fixed_half/fixed1/speedcap；24条件全覆盖，无旧物理缓存。last是冻结混合rank训练slot1，非BCE。H10，原worldgoal与真实触地保留；银行不是独立训练seed。speedcap通过原非对称残差权限与原滤波发布，实际介入另报，非安全保证。']
    elif tolerance:
        lines=['# 冻结容扰在线协调120回合','','全部120为本轮新H10物理：banks2/5×左右S×五事件×六方法；两个prepared祖先不是独立训练seed。已见来源/熟悉路线的在线工程DEV，非独立确认。', '','q_rank与q_bce为冻结三seed概率等权集成；task_guard_only用于分离几何guard收益。全程查询超出训练task1源分布，分数未校准为安全概率；fallback不是安全保证。原worldgoal1m、触地、无扰损失与原路径距离分别报告；重规划后执行路径user1奖励仅诊断。']
    elif online and not confirmation:
        lines=['# 因果几何预瞻scalar-alpha：18新回合＋36固定缓存','','控制器不使用起扰时刻或力标签，仅以当下执行路径/进度及原速度请求决定alpha；几何阈值是已知DEV上的工程启发式，不是稳定性界。无训练/阈值搜索。',
               '','两tight S、九事件，配对fixed.5/fixed1来自同原始54合同。共同原goal1m与真实触地为主，保留所有无扰任务/触地损失；不以百分比奖励优势作门槛。Alpha即时请求，原生参考滤波/限幅保留。']
    if switching:
        lines=['# H75同完整前缀准备时刻诊断：36新回合＋12原点stay＋6固定.5背景','','所有新回合H10；已知时刻切换不是未知起扰在线方案。任务1s完整状态核验后按原点配对比较，不声称起扰时状态相同或独立速度/转向因果效应。1.6s切换发生于首个脉冲步之前，不是反应式恢复。',
               '','原点stay为同来源同合同物理缓存；固定.5背景从回合初态不同策略演化，不能作为同状态切换对照。主要比较触地挽救及无扰不新增触地/故障/原goal1m损失；不设百分比代价门槛。',
               '','四个route/origin源组的全部事件和时刻保留。任务1s完整carry和主机状态逐叶核验；各分支至自身真实switch tick的原点前缀按协议核查并报告浮点差异，dtype/离散字段精确匹配。']
    if continuation:
        lines+=['','用户授权数值修订：原36条各自task1完整carry原样续接，未复制代表状态、未reset；同原点状态仅近似匹配。保留原失败封存。此为工程失败后协议修订，不是科学结果调参。与缓存stay的所有有限浮点前缀差异仅描述、不作拒绝门槛；报告max及旧参考容差超出数（phi_dot 1e-4，其余1e-5/rtol1e-6）。非有限量、dtype、离散量、实际动作/时刻/力及故障仍严格核查。完整组差异见initial_carry_review.json。']
    for event in response_plot_groups(rows):
        lines+=['',f'## {event}','',f'![Original and executed XY](full_xy_{event}.png)','',f'![First-arrival prefixes](task_xy_{event}.png)']
        if tolerance:lines+=['',f'![Per-step rewards and coordination](steps_and_coordination_{event}.png)']
        if broaden or ((frozen or speed_comparison or guard_comparison) and any(r.get('method')=='speedcap' for r in response_plot_selection(rows,event))):lines+=['',f'![Actual speedcap activity](speedcap_activity_{event}.png)']
    lines+=['','执行路径V3图以相同user1重评分；运行本身alpha评分另行核验并保留。逐步/累计奖励和分项仅为诊断，不是重规划后原用户路径效用。原路线距离与执行路径误差分别保存；全窗距离包含首达后运动，首达/真实失败前缀另列；未完成短窗不能当更低完成代价。',
        '世界Y脉冲投影使用日志起扰前upper yaw作一次onset-frame近似，不是逐物理子步积分。完整剂量/截断暴露、首达早于起扰、名义重规划逐例列出。4s若重规划，先规划后执行含脉冲的步。',
        '[逐例结果](results.json) · [无扰/受扰汇总](summary.json) · [全部配对得失](paired.json) · [核验](receipt.json)',
        '全部声明条件逐步原始帧、实际末点、重建奖励/累计分项、真速度/轮速代理与误差保留于case诊断NPZ；图见diagnostics。无20s数据或结论。']
    if broaden or preference:lines+=['','原路径三项V3费用已独立重建，见[共同实际/首达前窗口](original_task_rescoring/results.json)；不包含其余奖励或全局终止替换，不称完整原任务总回报。']
    elif tolerance:lines+=['','原路径V3加权聚合尚未核验，不能用执行路径奖励代替。已提供固定原路线距离/请求速度误差、同一实际窗口及同一首达前窗口配对；nominal损失逐条件保留。所有candidate latent/target、3×4概率/taskmask/proxy/原生权限逐步数据位于case诊断NPZ；独立NumPy重建概率与因果选择全部核查。']
    if preference:
        lines=[line.replace('相同user1','各自effective user').replace('user1重评分','effective-user重评分') for line in lines]
        lines+=['','## 同条件跨用户XY（不同用户加权奖励不得直接排序）']
        for b,r,e in dict.fromkeys((r['bank_index'],r['route'],r['event']) for r in rows):lines+=['',f'![Cross-user XY](crossuser_xy_{e}__bank{b}__route{r}.png)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    if broaden or preference:write_original_task_rescoring(root,rows,raws,paths,p['config'],p['initial_source']['controller'],manifest['reward_components'])
    return summary


def render_tolerance_steps_and_decisions(out,label,selected,raws,rebuilt,checkpoint):
    import matplotlib.pyplot as plt
    palette=('#0072B2','#D55E00','#008060','#CC79A7','#555555','#E69F00','#56B4E9','#7B3294')
    qmethods=[r['mode'] for r in selected if r['mode'] in TEMPORAL_TOLERANCE_ARMS+('q_rank','q_bce')+tuple(a for a in USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS if a.endswith('_last') or a.endswith('_prior') or a.endswith('_taskaware'))+SPEED_QUERY_ARMS+GUARD_FALLBACK_ARMS]
    fig,axes=plt.subplots(5+len(qmethods),1,figsize=(12,14+2*len(qmethods)),sharex=True,layout='constrained')
    for color,row in zip(palette,selected):
        raw=raws[row['case_id']];n=row['active_lower_ticks'];steps=np.arange(1,n+1);reward=rebuilt[row['case_id']]['user1_reward'];method=row['method']
        for ax,y in zip(axes[:3],(reward,np.cumsum(reward),raw['alpha_exec'][:n])):
            ax.plot(steps,y,c=color,label=method,lw=.8);ax.scatter(steps[-1],y[-1],c=color,marker='x' if row['horizons']['10']['physical_outcome']=='contact' else 's',s=16)
        for ax in axes:
            if row['force_N']:ax.axvspan(row['onset_s']/.005,(row['onset_s']+row['duration_s'])/.005,color='gray',alpha=.04)
            if row['replan_task_tick'] is not None:ax.axvline(row['replan_task_tick'],c=color,ls=':',lw=.6)
        if row['mode'] in qmethods:
            ax=axes[3+qmethods.index(row['mode'])]
            for a,style in enumerate(('-','--',':')):ax.plot(steps,raw['coord_qmin'][:n,a],ls=style,label=f"{method} candidate alpha{[0,.5,1][a]}",lw=.8)
            ax.axhline(.8,c='gray',ls=':');ax.axhline(.85,c='gray',ls='--')
        if 'effective_user_alpha' in row:axes[2].axhline(row['effective_user_alpha'],c='black',ls=':',lw=.8,label='Requested user alpha' if row is selected[0] else None)
        axes[-2].plot(steps,raw['coord_degraded'][:n],c=color,label=method,lw=.8)
        axes[-1].plot(steps,raw['coord_publication_withheld'][:n],c=color,label=method,lw=.8)
    for ax,title in zip(axes,['Diagnostic execution-path effective-user reward / actual step','Cumulative diagnostic execution-path effective-user reward','Actual executed alpha']+[name+': min over four hypotheses (uncalibrated)' for name in qmethods]+['Actual degraded fallback flag (not safety)','Native publication withheld flag']):
        ax.set_ylabel(title);ax.grid(alpha=.15);ax.legend(fontsize=6)
    axes[-1].set_xlabel('Actual lower-control transition step (dt=.005s)')
    fig.suptitle(label+' / '+checkpoint+' / effective user='+str(selected[0].get('effective_user_alpha',1.))+'; gray=intended disturbance, dotted=replan; actual failure endpoints retained')
    fig.savefig(out/f'steps_and_coordination_{label}.png',dpi=125);plt.close(fig)
    capped=[r for r in selected if is_speedcap_case(r)]
    if capped:
        row=capped[0];raw=raws[row['case_id']];n=row['active_lower_ticks'];steps=np.arange(1,n+1)
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
        for field in ('cap_m_s','requested_target_m_s','reachable_target_m_s','h75_target_m_s'):
            values=np.asarray(raw['speedcap_'+field][:n]);axes[0].plot(steps,np.where(np.isfinite(values),values,np.nan),label=field)
        axes[1].plot(steps,raw['governed'][:n,0],label='Native filtered reference');axes[1].plot(steps,raw['actual_forward_speed'][:n],label='Actual body speed')
        axes[2].plot(steps,raw['speedcap_active'][:n],label='Actual permitted speedcap activity');axes[2].plot(steps,raw['reference_target_permitted'][:n],ls=':',label='Native permission')
        for ax in axes:
            if row['force_N']:ax.axvspan(row['onset_s']/.005,(row['onset_s']+row['duration_s'])/.005,color='gray',alpha=.1)
            ax.legend(fontsize=7);ax.grid(alpha=.15)
        axes[0].set_ylabel('Target speed (m/s)');axes[1].set_ylabel('Speed (m/s)');axes[2].set_ylabel('Boolean flag');axes[-1].set_xlabel('Actual lower transition step (dt=.005s)')
        fig.suptitle(label+' / '+checkpoint+' / speedcap; infinite straight cap omitted; actual endpoint retained')
        fig.savefig(out/f'speedcap_activity_{label}.png',dpi=125);plt.close(fig)



def render_response(root,rows,raws,paths,goals,executed,rebuilt,keys,output_directory=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=output_directory or root/'analysis';diag=out/'diagnostics';diag.mkdir();routes=sorted(paths)
    tolerance=any(r.get('mode') in TOLERANCE_ARMS[:4]+TEMPORAL_TOLERANCE_ARMS+USER_PREFERENCE_ARMS+PRIOR_ARMS+TASK_AWARE_ARMS for r in rows)
    checkpoint='H75 frozen';late_display=any(r.get('trace_start_s')==6. for r in rows)
    if tolerance or late_display:
        plan=json.loads((root/'plan.json').read_text());checkpoint='H75 actor SHA '+plan['models']['H75']['actor_sha256'][:12]
    palette=('#0072B2','#D55E00','#008060','#CC79A7','#555555','#E69F00','#56B4E9','#7B3294')
    for event in response_plot_groups(rows):
        group=response_plot_selection(rows,event)
        routes=sorted({analysis_path_key(r) for r in group})
        methods=list(dict.fromkeys(r['method'] for r in group));colors=palette[:len(methods)]
        for task in (False,True):
            fig,axes=plt.subplots(1,len(routes),figsize=(13,6),squeeze=False,layout='constrained')
            for ax,route in zip(axes.flat,routes):
                path=paths[route];goal=goals[route];ax.plot(*path['xy'][path['s']<=path['goal']].T,'k-',lw=1,label='Original S to goal')
                ax.add_patch(Circle(goal,1,fill=False,ec='black'));ax.scatter(*goal,c='black',marker='*')
                for method,color in zip(methods,colors):
                    r=next(r for r in group if (analysis_path_key(r),r['method'])==(route,method));raw=raws[r['case_id']]
                    h=r['horizons']['10'];n=r['prefix']['ticks'] if task and h['goal_arrived'] else r['active_lower_ticks'];xy=raw['actual_xy'][:n]
                    if r.get('inherited_execution_path_file'):
                        with np.load(r['inherited_execution_path_file']) as z:inherited=dict(z)
                        ax.plot(*inherited['xy'][inherited['s']<=inherited['goal']].T,c=color,ls=':',lw=.8,alpha=.7,label=method+' inherited source execution path')
                    if r['case_id'] in executed:
                        path2=executed[r['case_id']];ax.plot(*path2['xy'][path2['s']<=path2['goal']].T,c=color,ls='--',lw=.8,alpha=.65,label=method+' executed replan')
                    if n:
                        ax.plot(*xy.T,c=color,label=method);marker='^' if task and h['goal_arrived'] else 'x' if h['physical_outcome']=='contact' else 's' if h['physical_outcome']=='survived' else 'D'
                        ax.scatter(*xy[-1],c=color,marker=marker,s=30)
                    if h['first_arrival_s'] is not None:
                        k=round((h['first_arrival_s']-r.get('trace_start_s',0.))/.005)-1
                        if k<n:ax.scatter(*raw['actual_xy'][k],c=color,marker='^')
                    if 'switch_s' in r:
                        k=round(r['switch_s']/.005)-1
                        if k<n:ax.scatter(*raw['actual_xy'][k],c=color,marker='o',facecolors='none',s=50,label=method+' switch')
                    if r['force_N']:
                        dose=np.flatnonzero(np.linalg.norm(raw['applied_force_impulse'][:n],axis=1)>1e-10)
                        if len(dose):ax.scatter(*raw['actual_xy'][dose].T,c=color,s=8,alpha=.5)
                ax.set(title=route,xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6);ax.grid(alpha=.15)
            fig.suptitle((checkpoint+' / ' if tolerance or late_display else '')+event+' / '+('first-arrival/contact prefix' if task else 'actual task6..10 suffix endpoint' if late_display else 'full actual H10 endpoint')+'; dashed=executed replan; dots=actual worldY exposure')
            fig.savefig(out/f'{"task" if task else "full"}_xy_{event}.png',dpi=140);plt.close(fig)
        for route in routes:
            selected=[r for r in group if analysis_path_key(r)==route]
            if tolerance:render_tolerance_steps_and_decisions(out,event,selected,raws,rebuilt,checkpoint)
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
                if 'switch_s' in r:
                    for ax in [*axes,*cax.flat]:ax.axvline(r['switch_s'],c=color,ls='--',lw=.8)
                if r['replan_task_tick'] is not None:
                    for ax in [*axes,*cax.flat]:ax.axvline(r['replan_task_tick']*.005,c=color,ls=':',lw=.6)
            for ax,label in zip(axes,('Speed: true/dotted wheel/dashed user (m/s)','True - wheel proxy (m/s)','True - original request (m/s)','Execution-path cross-track (m)','Execution-path heading error (rad)','Original goal distance (m)','Roll (rad)','Roll rate (rad/s)','Steer (rad)','Actual worldY mean force/tick (N)','Original route sample distance (m)','Effective-user diagnostic reward / actual step','Effective-user diagnostic cumulative reward')):
                ax.set_ylabel(label);ax.grid(alpha=.15);ax.legend(fontsize=6)
            for k,name in enumerate(list(keys)+['terminal replacement']):cax[k,0].set_ylabel(name);cax[k,1].set_ylabel('Cumulative '+name)
            cax[0,0].legend(fontsize=6)
            for ax in [axes[-1],*cax[-1]]:ax.set_xlabel('Actual post-tick time (s)')
            title=route+' / '+event+' / actual observed endpoints; vertical lines=replan, gray=intended pulse; no new training'
            fig.suptitle(title+'\nExecution-path errors/rewards are diagnostic; original goal unchanged')
            comp.suptitle(title+'\nSigned per-step and cumulative effective-user execution-path reward components; no original-user utility ranking')
            fig.savefig(diag/f'{route}_{event}.png',dpi=125);comp.savefig(diag/f'{route}_{event}_components.png',dpi=125);plt.close(fig);plt.close(comp)
            for r in selected:
                if r.get('mode')!='online_scalar':continue
                raw=raws[r['case_id']];n=r['active_lower_ticks'];t=np.arange(n)*.005
                rulefig,rax=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
                rax[0].plot(t,raw['alpha_exec'][:n],label='Actual queried scalar alpha');rax[0].axhline(1.,c='black',ls=':',label='Original user1')
                rax[1].plot(t,raw['rule_accel_estimate'][:n],label='Original v request squared x preview max curvature')
                rax[1].axhline(3.,c='black',ls='--',label='Engage >3');rax[1].axhline(2.7,c='gray',ls=':',label='Release <2.7')
                rax[2].plot(t,raw['actual_forward_speed'][:n]-raw['v_user'][:n],label='True - user speed (m/s)')
                for ax in rax:
                    if r['force_N']:ax.axvspan(r['onset_s'],r['onset_s']+r['duration_s'],color='gray',alpha=.15)
                    if r['replan_task_tick'] is not None:ax.axvline(r['replan_task_tick']*.005,c='purple',ls=':',label='Execution path replan')
                    ax.grid(alpha=.15);ax.legend(fontsize=7)
                rax[0].set_ylabel('Alpha (unitless)');rax[1].set_ylabel('Proxy (m/s²)');rax[2].set_ylabel('Speed error (m/s)');rax[2].set_xlabel('Pre-upper task time (s)')
                rulefig.suptitle(route+' / '+event+' / causal scalar rule; actual observed endpoint; no event label input')
                rulefig.savefig(diag/f'{route}_{event}_alpha_rule.png',dpi=125);plt.close(rulefig)


def validate_preference_inventory(rows,protocol):
    if tuple(protocol['arms'])!=USER_PREFERENCE_ARMS or len(rows)!=120:raise ValueError('declared user preference five arms required')
    actual=[]
    for row in rows:
        user=0. if row['mode'].startswith('user0_') else .5
        method=row['mode'].rsplit('_',1)[1]
        if row['mode'] not in USER_PREFERENCE_ARMS or row['alpha']!=user or row['user_alpha']!=user or row['effective_user_alpha']!=user or row['query_alpha']!=user or row['method']!=method:raise ValueError('user/exec/query/method identity mismatch')
        if row['initial_fingerprint']!=protocol['bank_state_fingerprints'][str(row['bank_index'])] or row['path_key']!=f"bank{row['bank_index']}_"+row['route']:raise ValueError('user preference bank/path identity mismatch')
        actual.append((row['bank_index'],row['route'],row['mode'],row['force_N'],row['onset_s'],row['duration_s']))
    events=[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))
    expected=[(b,r['route'],a,f,t,protocol['duration_s']) for b,r,a,(f,t) in itertools.product(protocol['bank_indices'],protocol['routes'],USER_PREFERENCE_ARMS,events)]
    if sorted(actual)!=sorted(expected):raise ValueError('complete user preference condition inventory required')


def load_user_preference_cache(root,p,paths,keys):
    import copy,importlib.util
    helper=Path(__file__).with_name('audit_native_goal_initial.py');spec=importlib.util.spec_from_file_location('_user_cache_audit',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    declared=p['comparison_source'];source=Path(declared['source'])
    checked=module.validate_user_preference_cache(source,config=p['config'],models=p['models'],protocol=p['protocol'],initial_source=p['initial_source'],replanning=p['replanning'])
    if checked!=declared:raise ValueError('sealed user preference cache metadata differs')
    original=json.loads((source/'analysis/results.json').read_text());rows=[];raws={};executed={};rebuilt={}
    for key,path in paths.items():
        with np.load(source/'evaluation'/(key+'_path.npz')) as z:
            for field in ('s','xy','heading','curvature','goal','turn_end'):np.testing.assert_array_equal(z[field],path[field])
    for lane,parent in enumerate(original):
        if parent['mode'] not in ('temporal_rank_last','task_guard_only','fixed1','fixed_half'):continue
        row=copy.deepcopy(parent);user=.5 if parent['mode']=='fixed_half' else 1.;method={'temporal_rank_last':'last','task_guard_only':'guard','fixed1':'fixed','fixed_half':'fixed'}[parent['mode']]
        row.update(case_id=f'effectiveuser{user:g}/'+parent['case_id'],effective_user_alpha=user,user_alpha=user,raw_user_alpha=1.,method=method,source_kind='cached_broaden',cache_source=str(source),cached_parent_case_id=parent['case_id'],newly_simulated_lower_ticks=0)
        with np.load(row['raw_file']) as z:raw=dict(z)
        changed=None;tick=row['replan_task_tick']
        if tick is not None:
            with np.load(source/'evaluation'/f'replanned_path_{lane:02d}.npz') as z:changed=dict(z)
            executed[row['case_id']]=changed
        data=reconstruct_execution_path_reward(raw,row,paths[analysis_path_key(row)],changed,row['active_lower_ticks'] if tick is None else tick,p['config'],p['initial_source']['controller'],keys)
        row['alpha_rule']=tolerance_alpha_statistics(raw,user,controller_uses_risk_gate=row.get('method') not in ('guard','fixed','speedcap'))
        row['diagnostic_execution_path_effective_user_cost']=float(-data['user1_reward'].sum())
        target=root/'analysis'/f'cached_case_{lane:03d}_diagnostics.npz';np.savez_compressed(target,**{k:np.asarray(v)[:row['active_lower_ticks']] for k,v in raw.items()},effective_user_reward=data['user1_reward'],effective_user_cumulative_reward=np.cumsum(data['user1_reward']),effective_user_running_components=data['user1_running_components'],effective_user_terminal_adjustment=data['user1_terminal_adjustment']);row['effective_user_diagnostic_file']=str(target)
        row['prefix']['diagnostic_execution_path_effective_user_cost']=float(-data['user1_reward'][:row['prefix']['ticks']].sum())
        rows.append(row);raws[row['case_id']]=raw;rebuilt[row['case_id']]=data
    return dict(rows=rows,raws=raws,executed=executed,rebuilt=rebuilt)


def preference_common_windows(rows,raws,paths,goals):
    for label in response_plot_groups(rows):
        group=response_plot_selection(rows,label)
        methods=preference_methods([r for r in rows if r['effective_user_alpha']==group[0]['effective_user_alpha']])
        if len(group)!=len(methods) or {r['method'] for r in group}!=set(methods):raise ValueError('complete matched methods required for each user/condition')
        common=min(r['active_lower_ticks'] for r in group);prefix=min(r['prefix']['ticks'] for r in group)
        for row in group:
            raw=raws[row['case_id']];user=row['effective_user_alpha'];a=raw['alpha_exec'][:row['active_lower_ticks']];n=row['prefix']['ticks']
            row['raw_user_alpha']=row.get('raw_user_alpha',row['alpha'])
            row['diagnostic_execution_path_effective_user_cost']=row.get('diagnostic_execution_path_effective_user_cost',row['diagnostic_execution_path_user1_cost'])
            row['alpha_rule']['effective_user_alpha']=user
            row['alpha_rule']['full_horizon_survived']=row['active_lower_ticks']==2000 and row['horizons']['10']['physical_outcome']=='survived'
            row['alpha_rule']['full_horizon_at_user']=bool(row['alpha_rule']['full_horizon_survived'] and row['alpha_rule']['final_at_user'])
            row['alpha_rule']['prearrival_alpha']=dict(actual_ticks=n,user_duration_s=float(np.sum(a[:n]==user)*.005),deviation_duration_s=float(np.sum(a[:n]!=user)*.005),final_at_user=bool(a[n-1]==user) if n else None)
            row['alpha_rule']['mean_absolute_alpha_deviation']=float(np.mean(abs(a-user)))
            path=paths[analysis_path_key(row)];tree=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[analysis_path_key(row)]]));metrics={}
            for name,n in [('same_actual_window',common),('same_pre_first_arrival_window',prefix)]:
                speed=raw['actual_forward_speed'][:n]-row['speed'];d=tree.query(raw['actual_xy'][:n])[0]
                metrics[name]=dict(actual_ticks=n,observed_s=n*.005,speed_rmse_m_s=float(np.sqrt(np.mean(speed**2))) if n else None,original_route_sample_distance_rmse_m=float(np.sqrt(np.mean(d**2))) if n else None)
            row['matched_original_task_metrics']=metrics
            row['matched_window_methods']=list(methods);row['matched_window_method_count']=len(methods)


def preference_methods(rows):
    methods=tuple(x for x in ('last','guard','fixed','prior_last','task_aware','speedcap','speed_query','guard_degraded') if any(r['method']==x for r in rows))
    if methods not in (('last','guard','fixed'),('last','guard','fixed','prior_last'),('last','guard','fixed','prior_last','task_aware'),('guard','fixed','task_aware'),('guard','fixed','task_aware','speedcap'),('guard','fixed','task_aware','speed_query'),('guard','fixed','task_aware','speedcap','speed_query'),('guard','fixed','task_aware','guard_degraded'),('guard','fixed','task_aware','speedcap','guard_degraded'),('guard','fixed','speedcap','guard_degraded')):raise ValueError('unsupported preference method set')
    return methods


def preference_summary(rows,frozen_confirmation=False,speed_comparison=False,frozen_layout=None,guard_layout=None):
    frozen_layout=frozen_layout or dict(cases=200,conditions=20,user_groups=60,variant=None)
    conditions=guard_layout['conditions'] if guard_layout else frozen_layout['conditions'] if frozen_confirmation else 20 if speed_comparison else 24
    if guard_layout:
        if len(rows)!=guard_layout['logical_cases'] or len(response_plot_groups(rows))!=guard_layout['user_groups']:raise ValueError('complete guard-fallback comparison required')
    elif speed_comparison:
        if len(rows)!=260 or len(response_plot_groups(rows))!=60:raise ValueError('260 records/60 user-condition groups required')
    elif frozen_confirmation:
        if len(rows)!=frozen_layout['cases'] or len(response_plot_groups(rows))!=frozen_layout['user_groups']:raise ValueError('complete frozen records/user-condition groups required')
    else:
        methods=preference_methods(rows)
        if len(rows)!=72*len(methods) or len(response_plot_groups(rows))!=72:raise ValueError('complete user-condition groups required')
    summary={};paired=[]
    for user in frozen_layout.get('users',(0.,.5,1.)) if frozen_confirmation else (0.,.5,1.):
        summary[str(user)]={}
        methods=preference_methods([r for r in rows if r['effective_user_alpha']==user])
        if guard_layout and set(methods)!=({'task_aware','guard','fixed','speedcap','guard_degraded'} if user==1. else {'task_aware','guard','fixed','guard_degraded'}):raise ValueError('declared per-user guard-fallback methods required')
        if speed_comparison and set(methods)!=({'task_aware','guard','fixed','speedcap','speed_query'} if user==1. else {'task_aware','guard','fixed','speed_query'}):raise ValueError('declared per-user input ablation methods required')
        if frozen_confirmation and set(methods)!=(set(frozen_layout['methods']) if 'methods' in frozen_layout else {'task_aware','guard','fixed','speedcap'} if user==1. else {'task_aware','guard','fixed'}):raise ValueError('declared per-user methods required')
        for method in methods:
            group=[r for r in rows if r['effective_user_alpha']==user and r['method']==method]
            if len(group)!=conditions:raise ValueError('complete conditions per user/method required')
            summary[str(user)][method]=dict(cases=conditions,goal_arrived=sum(r['horizons']['10']['goal_arrived'] is True for r in group),contact=sum(r['horizons']['10']['physical_outcome']=='contact' for r in group),unknown=sum(r['horizons']['10']['physical_outcome']=='unknown' for r in group),final_at_user=sum(r['alpha_rule']['final_at_user'] for r in group),mean_deviation_s=float(np.mean([r['alpha_rule']['nonuser_duration_s'] for r in group])),mean_prearrival_user_s=float(np.mean([r['alpha_rule']['prearrival_alpha']['user_duration_s'] for r in group])))
            summary[str(user)][method].update(full_horizon_at_user=sum(r['alpha_rule'].get('full_horizon_at_user',False) for r in group),guard_fallback_selected_user_without_restore_s=sum(r['alpha_rule'].get('guard_fallback_selected_user_without_restore_s',0.) for r in group))
            gates=[r['alpha_rule'].get('final_qualified_user_restoration') for r in group]
            summary[str(user)][method].update(final_qualified_user_restoration=sum(x is True for x in gates),restoration_gate_unknown=sum(x is None for x in gates),fallback_selection_metric_unknown=sum(r['alpha_rule'].get('fallback_selected_user_without_restore_s') is None for r in group),user_execution_under_degraded_s=sum(r['alpha_rule'].get('user_execution_under_degraded_s',0.) for r in group),fallback_selected_user_without_restore_s=sum(r['alpha_rule'].get('fallback_selected_user_without_restore_s') or 0. for r in group))
            not_used=sum(r['alpha_rule'].get('restoration_gate_used_by_controller') is False for r in group)
            summary[str(user)][method]['restoration_gate_not_applicable']=not_used
            if not_used==len(group):
                summary[str(user)][method]['final_qualified_user_restoration']=None
                summary[str(user)][method]['restoration_gate_unknown']=0
        for left,right in itertools.combinations(methods,2):
            a={(r['bank_index'],r['route'],r['event']):r for r in rows if r['effective_user_alpha']==user and r['method']==left};b={(r['bank_index'],r['route'],r['event']):r for r in rows if r['effective_user_alpha']==user and r['method']==right}
            paired.append(dict(effective_user_alpha=user,candidate=left,baseline=right,conditions=[dict(bank_index=k[0],route=k[1],event=k[2],candidate_case=a[k]['case_id'],baseline_case=b[k]['case_id'],candidate_goal=a[k]['horizons']['10']['goal_arrived'],baseline_goal=b[k]['horizons']['10']['goal_arrived'],candidate_contact=a[k]['horizons']['10']['physical_outcome'],baseline_contact=b[k]['horizons']['10']['physical_outcome'],candidate_matched_metrics=a[k]['matched_original_task_metrics'],baseline_matched_metrics=b[k]['matched_original_task_metrics'],candidate_full_mask_metrics=a[k].get('full_mask_matched_original_task_metrics'),baseline_full_mask_metrics=b[k].get('full_mask_matched_original_task_metrics'),candidate_old_new_metrics=a[k].get('old_new_candidate_matched_original_task_metrics'),baseline_old_new_metrics=b[k].get('old_new_candidate_matched_original_task_metrics')) for k in a]))
    report=dict(new_cases=guard_layout['new_cases'] if guard_layout else 60 if speed_comparison else frozen_layout['cases'] if frozen_confirmation else 120 if len(methods)==3 else 72,cached_cases=guard_layout['cached_cases'] if guard_layout else 200 if speed_comparison else 0 if frozen_confirmation else 72*len(methods)-(120 if len(methods)==3 else 72),logical_records=len(rows),per_user_method=summary,cross_user_reward_ranking='invalid: different user weights',adoption='not_automatic',data_role='Counterexample-driven guard-equivalent fallback DEV revision with authenticated original q/core cache; no independent confirmation claim' if guard_layout else 'Matched input online ablation on already seen related conditions; cached original q audit inherited; not independent confirmation' if speed_comparison else 'Frozen prospective combination confirmation; prepared banks not training seeds; no checkpoint selection on these results' if frozen_confirmation else 'Frozen candidate discrete user preference bounded engineering validation; prepared banks are not training seeds')
    if guard_layout:
        report['new_vs_old']=new_vs_old_outcomes(rows,conditions)
        report['stage']=guard_layout['stage']
    if frozen_confirmation:
        report['protocol_variant']=frozen_layout['variant']
        report['conditions_per_user_method']=conditions
        if frozen_layout['variant']=='geometry_duration_extension_v1':report['data_role']='Frozen capability-domain geometry/actual-duration extension on reused banks1/7; actual0.15s beyond frozen0.1s hypothetical queries; not new independent initial states'
    if speed_comparison:
        report['full_mask_task_outcomes']={}
        for user in (0.,.5,1.):
            full={(r['bank_index'],r['route'],r['event']):r for r in rows if r['effective_user_alpha']==user and r['method']=='task_aware'}
            masked={(r['bank_index'],r['route'],r['event']):r for r in rows if r['effective_user_alpha']==user and r['method']=='speed_query'}
            report['full_mask_task_outcomes'][str(user)]=dict(conditions=20,masked_goal_gains=sum(masked[k]['horizons']['10']['goal_arrived'] is True and full[k]['horizons']['10']['goal_arrived'] is False for k in full),masked_goal_losses=sum(masked[k]['horizons']['10']['goal_arrived'] is False and full[k]['horizons']['10']['goal_arrived'] is True for k in full),both_goal_arrived=sum(masked[k]['horizons']['10']['goal_arrived'] is True and full[k]['horizons']['10']['goal_arrived'] is True for k in full),full_contacts=sum(r['horizons']['10']['physical_outcome']=='contact' for r in full.values()),masked_contacts=sum(r['horizons']['10']['physical_outcome']=='contact' for r in masked.values()))
    return report,paired



def render_preference_crossuser(root,rows,raws,paths,goals,executed):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=root/'analysis';colors={0.:'#0072B2',.5:'#D55E00',1.:'#008060'};styles={'last':'-','guard':'--','fixed':':','prior_last':'-.','task_aware':(0,(3,1,1,1,1,1)),'speedcap':(0,(5,2)),'speed_query':(0,(2,1)),'guard_degraded':(0,(4,1,1,1))}
    checkpoint=json.loads((root/'plan.json').read_text())['models']['H75']['actor_sha256'][:12]
    groups=list(dict.fromkeys((r['bank_index'],r['route'],r['event']) for r in rows))
    for bank,route,event in groups:
        group=[r for r in rows if (r['bank_index'],r['route'],r['event'])==(bank,route,event)];path=paths[analysis_path_key(group[0])];goal=goals[analysis_path_key(group[0])]
        fig,ax=plt.subplots(figsize=(12,7),layout='constrained');ax.plot(*path['xy'][path['s']<=path['goal']].T,'k-',label='Immutable original task reference');ax.add_patch(Circle(goal,1.,fill=False,ec='black'));ax.scatter(*goal,c='black',marker='*')
        for row in group:
            raw=raws[row['case_id']];n=row['active_lower_ticks'];color=colors[row['effective_user_alpha']];label=f"user{row['effective_user_alpha']:g}/{row['method']}";xy=raw['actual_xy'][:n]
            ax.plot(*xy.T,color=color,ls=styles[row['method']],label=label,lw=1.);ax.scatter(*xy[-1],c=color,marker='x' if row['horizons']['10']['physical_outcome']=='contact' else 's',s=20)
            if row['case_id'] in executed:
                p2=executed[row['case_id']];ax.plot(*p2['xy'][p2['s']<=p2['goal']].T,c=color,ls='-.',lw=.5,alpha=.4)
            dose=np.flatnonzero(np.linalg.norm(raw['applied_force_impulse'][:n],axis=1)>1e-10)
            if len(dose):ax.scatter(*xy[dose].T,c=color,s=7)
        ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f'H75 {checkpoint} / bank{bank}/{route}/{event}: users{sorted(set(r["effective_user_alpha"] for r in group))}; actual H10 endpoints; dashdot=execution replan');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=7);ax.grid(alpha=.15)
        fig.savefig(out/f'crossuser_xy_{event}__bank{bank}__route{route}.png',dpi=140);plt.close(fig)





def frozen_transfer_layout(protocol):
    declarations=protocol['method_declarations'];arms=protocol['arms']
    if [d['arm'] for d in declarations]!=arms or len(set(arms))!=len(arms):raise ValueError('unique transfer declarations required')
    if len(declarations)!=4 or {d['method'] for d in declarations}!={'guard_degraded','guard','fixed','speedcap'} or any(d['user_alpha']!=1. for d in declarations):raise ValueError('declared user1 four-method transfer required')
    events=[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))
    banks=protocol['bank_indices'];routes=protocol['routes']
    if len(set(banks))!=len(banks) or len({r['route'] for r in routes})!=len(routes) or len(set(events))!=len(events):raise ValueError('unique transfer cartesian conditions required')
    conditions=len(banks)*len(routes)*len(events);cases=conditions*len(arms)
    if protocol['expected_cases']!=cases or protocol['maximum_lower_ticks']!=cases*2000 or protocol['horizon_s']!=10. or protocol['optimizer_updates']!=0:raise ValueError('transfer declared count/budget differs')
    return dict(variant='frozen_transfer_evaluation',cases=cases,conditions=conditions,routes=len(routes),user_groups=conditions,nominal_cases=len(banks)*len(routes)*len(arms),users=[1.],methods=[d['method'] for d in declarations])


def validate_frozen_transfer_inventory(rows,protocol):
    layout=frozen_transfer_layout(protocol)
    if len(rows)!=layout['cases']:raise ValueError('complete transfer inventory required')
    declarations={d['arm']:d for d in protocol['method_declarations']};actual=[]
    for row in rows:
        d=declarations.get(row['mode'])
        if d is None or row['method']!=d['method'] or any(row[k]!=d['user_alpha'] for k in ('alpha','user_alpha','effective_user_alpha','query_alpha')):raise ValueError('transfer user/method mismatch')
        if row['initial_fingerprint']!=protocol['bank_state_fingerprints'][str(row['bank_index'])] or row['path_key']!=f"bank{row['bank_index']}_"+row['route']:raise ValueError('transfer bank/path mismatch')
        actual.append((row['bank_index'],row['route'],row['speed'],row['mode'],row['force_N'],row['onset_s'],row['duration_s']))
    expected=[(b,r['route'],r['speed'],a,f,t,protocol['duration_s']) for b,r,a,(f,t) in itertools.product(protocol['bank_indices'],protocol['routes'],protocol['arms'],[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s'])))]
    if sorted(actual)!=sorted(expected):raise ValueError('complete transfer condition coverage required')


def frozen_confirmation_layout(protocol):
    """Only explicitly registered variants may alter the historical200 layout."""
    if protocol.get('mode')=='frozen_transfer_evaluation':return frozen_transfer_layout(protocol)
    variant=protocol.get('variant')
    if variant not in (None,'geometry_duration_extension_v1'):raise ValueError('unknown frozen confirmation variant')
    if tuple(protocol['arms'])!=FROZEN_CONFIRMATION_ARMS:raise ValueError('frozen ten-arm contract required')
    events=[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))
    banks=protocol['bank_indices'];routes=protocol['routes']
    if len(set(banks))!=len(banks) or len({r['route'] for r in routes})!=len(routes) or len(set(events))!=len(events):raise ValueError('unique frozen conditions required')
    conditions=len(banks)*len(routes)*len(events);cases=conditions*len(FROZEN_CONFIRMATION_ARMS)
    layout=dict(variant=variant,conditions=conditions,cases=cases,routes=len(routes),user_groups=3*conditions,nominal_cases=len(banks)*len(routes)*len(FROZEN_CONFIRMATION_ARMS))
    if variant is None:
        if cases!=200 or conditions!=20 or len(routes)!=2 or layout['nominal_cases']!=40:raise ValueError('legacy frozen200 layout must remain unchanged')
    else:
        expected_routes={'S_left_60_R2p2','S_right_60_R2p2','S_left_120_R2p2','S_right_120_R2p2'}
        if banks!=[1,7] or {r['route'] for r in routes}!=expected_routes or any(r['speed']!=2.6 for r in routes) or protocol['forces_N']!=[-64.,64.] or protocol['onsets_s']!=[2.6] or protocol['zero_onset_s']!=2.6 or protocol['duration_s']!=.15 or cases!=240:raise ValueError('geometry-duration extension differs from registered protocol')
    if protocol['expected_cases']!=cases:raise ValueError('declared frozen case count differs from protocol')
    return layout


def frozen_confirmation_intro(protocol,layout):
    if protocol.get('mode')=='frozen_transfer_evaluation':
        return [f"# 冻结控制组合确认{layout['cases']}",'',f"全部{layout['cases']}为新H10物理，{layout['conditions']}条件×四user1方法；banks{protocol['bank_indices']}，原始goal1m与真实contact分别报告；q+guardfallback对无qguard为主比较，fixed1/speedcap为简单对照。0优化，全部条件/失败保留。",'',"仅改变声明初态/几何/实际脉冲组合，不称全研究未见TEST或独立训练seed；0.15s实际脉冲相对冻结0.1s假设外推。控制、q三seed/H75/lower、门槛和one-shot接回全冻结；guard/fixed/cap的qgate只作诊断，非其控制恢复门。用户0/.5未新增确认。"]
    routes=', '.join(f"{r['route']} ({r['speed']:g}m/s)" for r in protocol['routes'])
    text=f"全部{layout['cases']}为新物理：banks{protocol['bank_indices']}；{routes}；实际力{protocol['forces_N']}N、起扰{protocol['onsets_s']}s、持续{protocol['duration_s']:g}s，含无扰。每用户{layout['conditions']}条件、同用户3/4方法共同窗口。user0/.5各taskaware/guard/fixed，user1另有speedcap。冻结候选后声明，不按本结果选checkpoint；guard/fixed/speedcap不使用q决策，其模型gate仅为事后诊断，合格恢复记N/A。0优化更新。"
    if layout['variant']=='geometry_duration_extension_v1':
        scope='能力域几何/实际持续时间扩展：四路均R2.2m、v2.6m/s，60/120度只改变转角与行程，不代表不同曲率/速度；复用已用banks1/7，不称新独立初态或独立训练seed；实际0.15s脉冲相对冻结q的±64N/0.1s假想查询属于持续时间外推。q查询、网络、控制器和原任务成功门均不因本扩展更改。'
    else:scope='原冻结200组合确认范围；不声称全项目历史从未看过这些banks。'
    return [f"# 冻结候选组合确认{layout['cases']}",'',text,'',scope]

def validate_frozen_confirmation_inventory(rows,protocol):
    if protocol.get('mode')=='frozen_transfer_evaluation':return validate_frozen_transfer_inventory(rows,protocol)
    layout=frozen_confirmation_layout(protocol)
    if len(rows)!=layout['cases']:raise ValueError('complete declared frozen confirmation cases required')
    expected=set()
    for b,r,a,(force,onset) in itertools.product(protocol['bank_indices'],protocol['routes'],FROZEN_CONFIRMATION_ARMS,[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))):
        expected.add((b,r['route'],r['speed'],a,force,onset,protocol['duration_s']))
    actual=set()
    for row in rows:
        user=0. if row['mode'].startswith('user0_') else .5 if row['mode'].startswith('userhalf_') else 1.
        method={'taskaware':'task_aware','guard':'guard','fixed':'fixed','speedcap':'speedcap'}[row['mode'].rsplit('_',1)[1]]
        if any(row[k]!=user for k in ('alpha','user_alpha','effective_user_alpha','query_alpha')) or row['method']!=method:raise ValueError('frozen confirmation user/method identity mismatch')
        if row['initial_fingerprint']!=protocol['bank_state_fingerprints'][str(row['bank_index'])] or row['path_key']!=f"bank{row['bank_index']}_"+row['route']:raise ValueError('frozen confirmation bank/path mismatch')
        actual.add((row['bank_index'],row['route'],row['speed'],row['mode'],row['force_N'],row['onset_s'],row['duration_s']))
    if actual!=expected:raise ValueError('complete frozen confirmation conditions required')

def validate_prior_inventory(rows,protocol,task_aware=False):
    arms=TASK_AWARE_ARMS if task_aware else PRIOR_ARMS
    if tuple(protocol['arms'])!=arms or len(rows)!=72:raise ValueError('72 prior cases required')
    expected=set()
    for b,r,a,(force,onset) in itertools.product(protocol['bank_indices'],protocol['routes'],arms,[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))):
        expected.add((b,r['route'],a,force,onset,protocol['duration_s']))
    actual=set()
    for row in rows:
        user=dict(zip(arms,(0.,.5,1.)))[row['mode']]
        if any(row[k]!=user for k in ('alpha','user_alpha','effective_user_alpha','query_alpha')) or row['method']!=('task_aware' if task_aware else 'prior_last'):raise ValueError('prior user identity mismatch')
        if row['initial_fingerprint']!=protocol['bank_state_fingerprints'][str(row['bank_index'])] or row['path_key']!=f"bank{row['bank_index']}_"+row['route']:raise ValueError('prior bank/path mismatch')
        actual.add((row['bank_index'],row['route'],row['mode'],row['force_N'],row['onset_s'],row['duration_s']))
    if actual!=expected:raise ValueError('complete prior conditions required')


def load_prior_online_cache(root,p,paths,keys):
    import copy,importlib.util
    spec=importlib.util.spec_from_file_location('_prior_cache_audit',Path(__file__).with_name('audit_native_goal_initial.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=Path(p['comparison_source']['source'])
    checked=(module.validate_task_aware_cache if p['mode']=='task_aware_fallback72' else module.validate_prior_online_cache)(source,config=p['config'],models=p['models'],protocol=p['protocol'],initial_source=p['initial_source'],replanning=p['replanning'])
    if checked!=p['comparison_source']:raise ValueError('sealed prior cache metadata differs')
    physical_indices={}
    for key,path in paths.items():
        with np.load(source/'evaluation'/(key+'_path.npz')) as z:
            for field in ('s','xy','heading','curvature','goal','turn_end'):np.testing.assert_allclose(z[field],path[field],atol=3e-6,rtol=1e-6)
    rows=copy.deepcopy(json.loads((source/'analysis/results.json').read_text()));raws={};executed={};rebuilt={}
    for lane,row in enumerate(rows):
        with np.load(row['raw_file']) as z:raw=dict(z)
        changed=None;tick=row['replan_task_tick']
        if tick is not None:
            evaluation=Path(row['raw_file']).parent.parent
            if evaluation not in physical_indices:
                physical_indices[evaluation]={str(Path(r['raw_file']).resolve()):i for i,r in enumerate(json.loads((evaluation/'case_index.json').read_text()))}
            index=physical_indices[evaluation][str(Path(row['raw_file']).resolve())]
            with np.load(evaluation/f'replanned_path_{index:02d}.npz') as z:changed=dict(z)
            executed[row['case_id']]=changed
        data=reconstruct_execution_path_reward(raw,row,paths[analysis_path_key(row)],changed,row['active_lower_ticks'] if tick is None else tick,p['config'],p['initial_source']['controller'],keys)
        row.setdefault('cache_ancestry',[]).append(dict(source=str(source),source_kind=row.get('source_kind'),cache_source=row.get('cache_source')))
        row.update(source_kind='cached_prior' if p['mode']=='task_aware_fallback72' else 'cached_preference',cache_source=str(source),newly_simulated_lower_ticks=0)
        row['alpha_rule']=tolerance_alpha_statistics(raw,row['effective_user_alpha'],controller_uses_risk_gate=row.get('method') not in ('guard','fixed','speedcap'))
        raws[row['case_id']]=raw;rebuilt[row['case_id']]=data
    return dict(rows=rows,raws=raws,executed=executed,rebuilt=rebuilt)


def validate_speed_query_inventory(rows,protocol):
    if tuple(protocol['arms'])!=SPEED_QUERY_ARMS or len(rows)!=60:raise ValueError('60 speed-query cases required')
    expected={(b,r['route'],a,f,t,protocol['duration_s']) for b,r,a,(f,t) in itertools.product(protocol['bank_indices'],protocol['routes'],SPEED_QUERY_ARMS,[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s'])))}
    actual=set()
    for row in rows:
        user=dict(zip(SPEED_QUERY_ARMS,(0.,.5,1.)))[row['mode']]
        if any(row[k]!=user for k in ('alpha','user_alpha','effective_user_alpha','query_alpha')) or row['method']!='speed_query':raise ValueError('speed-query user/method identity mismatch')
        if row['initial_fingerprint']!=protocol['bank_state_fingerprints'][str(row['bank_index'])] or row['path_key']!=f"bank{row['bank_index']}_"+row['route']:raise ValueError('speed-query bank/path mismatch')
        actual.add((row['bank_index'],row['route'],row['mode'],row['force_N'],row['onset_s'],row['duration_s']))
    if actual!=expected:raise ValueError('complete speed-query conditions required')


def load_speed_query_cache(root,p,paths,keys):
    """Reuse authenticated original q/core audits, not the modified candidate audit."""
    import copy,importlib.util
    spec=importlib.util.spec_from_file_location('_speed_cache_audit',Path(__file__).with_name('audit_native_goal_initial.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=Path(p['comparison_source']['source'])
    guard_comparison=p['mode']=='guard_fallback_evaluation'
    checked=(module.validate_guard_fallback_cache if guard_comparison else module.validate_speed_query_cache)(source,config=p['config'],models=p['models'],protocol=p['protocol'],initial_source=p['initial_source'],replanning=p['replanning'])
    if checked!=p['comparison_source']:raise ValueError('sealed speed-query cache metadata differs')
    original_plan=json.loads((source/'plan.json').read_text())
    for key,path in paths.items():
        with np.load(source/'evaluation'/(key+'_path.npz')) as z:
            for field in ('s','xy','heading','curvature','goal','turn_end'):np.testing.assert_allclose(z[field],path[field],atol=3e-6,rtol=1e-6)
    rows=copy.deepcopy(json.loads((source/'analysis/results.json').read_text()));raws={};executed={};rebuilt={}
    for lane,row in enumerate(rows):
        if not row['alpha_rule'].get('independent_numpy_decisions_match'):raise ValueError('cached full coordinator audit incomplete')
        row['q_audit_inheritance']=dict(source=str(source),analysis_receipt_sha256=digest(source/'analysis/receipt.json'),original_q_receipt_sha256=original_plan['frozen_predictor_export']['receipt_sha256'],original_q_directory=original_plan['frozen_predictor_export']['directory'],original_core_sources={k:v for k,v in original_plan['source_sha256'].items() if k.endswith('tolerance_coordination.py')},reused_complete_original_audit=True,modified_core_not_applied=guard_comparison,masked_bundle_not_applied=True)
        row['cached_original_matched_task_metrics']=copy.deepcopy(row['matched_original_task_metrics'])
        row['cached_original_window_methods']=copy.deepcopy(row.get('matched_window_methods'))
        row['cached_original_task_cost_source']=str(source/'analysis/original_task_rescoring/results.json')
        if not guard_comparison:
            row['frozen200_original_matched_task_metrics']=copy.deepcopy(row['matched_original_task_metrics'])
            row['frozen200_original_window_methods']=copy.deepcopy(row.get('matched_window_methods'))
            row['frozen200_original_task_cost_source']=str(source/'analysis/original_task_rescoring/results.json')
        with np.load(row['raw_file']) as z:raw=dict(z)
        changed=None;tick=row['replan_task_tick']
        if tick is not None:
            with np.load(source/'evaluation'/f'replanned_path_{lane:02d}.npz') as z:changed=dict(z)
            executed[row['case_id']]=changed
        rebuilt[row['case_id']]=reconstruct_execution_path_reward(raw,row,paths[analysis_path_key(row)],changed,row['active_lower_ticks'] if tick is None else tick,p['config'],p['initial_source']['controller'],keys)
        row.update(source_kind='cached_guard_source' if guard_comparison else 'cached_frozen200',cache_source=str(source),newly_simulated_lower_ticks=0,diagnostic_file=str(source/'analysis'/row['diagnostic_file']))
        raws[row['case_id']]=raw
    return dict(rows=rows,raws=raws,executed=executed,rebuilt=rebuilt)


def full_mask_common_windows(rows,raws,paths,goals):
    return paired_candidate_common_windows(rows,raws,paths,goals,candidate='speed_query',metric_key='full_mask_matched_original_task_metrics')


def paired_candidate_common_windows(rows,raws,paths,goals,*,candidate,metric_key):
    """Pair-only windows stay separate from all-method observation windows."""
    for label in response_plot_groups(rows):
        pair=[r for r in response_plot_selection(rows,label) if r['method'] in ('task_aware',candidate)]
        if len(pair)!=2 or {r['method'] for r in pair}!={'task_aware',candidate}:raise ValueError('two full/mask methods required')
        for row in pair:
            raw=raws[row['case_id']];path=paths[analysis_path_key(row)];tree=cKDTree(np.vstack([path['xy'][path['s']<=path['goal']],goals[analysis_path_key(row)]]));metrics={}
            for name,n in [('same_actual_window',min(r['active_lower_ticks'] for r in pair)),('same_pre_first_arrival_window',min(r['prefix']['ticks'] for r in pair))]:
                speed=raw['actual_forward_speed'][:n]-row['speed'];distance=tree.query(raw['actual_xy'][:n])[0]
                metrics[name]=dict(actual_ticks=n,observed_s=n*.005,methods=['task_aware',candidate],speed_rmse_m_s=float(np.sqrt(np.mean(speed**2))) if n else None,original_route_sample_distance_rmse_m=float(np.sqrt(np.mean(distance**2))) if n else None)
            for name,metric in metrics.items():metric['user_execution_s']=float(np.sum(np.asarray(raw['alpha_exec'][:metric['actual_ticks']])==row['effective_user_alpha'])*.005)
            row[metric_key]=metrics

def guard_fallback_layout(protocol):
    stage=protocol.get('stage')
    if stage not in ('extension','regression'):raise ValueError('unknown guard-fallback stage')
    if tuple(protocol['arms'])!=GUARD_FALLBACK_ARMS:raise ValueError('three declared guard-fallback arms required')
    source_protocol=dict(protocol,arms=list(FROZEN_CONFIRMATION_ARMS),expected_cases=240 if stage=='extension' else 200)
    layout=frozen_confirmation_layout(source_protocol)
    if (stage=='extension')!=(layout['variant']=='geometry_duration_extension_v1'):raise ValueError('guard-fallback stage/source variant mismatch')
    new_cases=layout['conditions']*3
    if protocol['expected_cases']!=new_cases:raise ValueError('guard-fallback declared new count mismatch')
    return dict(stage=stage,conditions=layout['conditions'],new_cases=new_cases,cached_cases=layout['cases'],logical_cases=new_cases+layout['cases'],user_groups=layout['user_groups'],routes=layout['routes'],nominal_new_cases=len(protocol['bank_indices'])*len(protocol['routes'])*3)


def validate_guard_fallback_inventory(rows,protocol):
    layout=guard_fallback_layout(protocol)
    if len(rows)!=layout['new_cases']:raise ValueError('complete guard-fallback new inventory required')
    events=[(0.,protocol['zero_onset_s'])]+list(itertools.product(protocol['forces_N'],protocol['onsets_s']))
    expected={(b,r['route'],r['speed'],a,f,t,protocol['duration_s']) for b,r,a,(f,t) in itertools.product(protocol['bank_indices'],protocol['routes'],GUARD_FALLBACK_ARMS,events)}
    actual=set()
    for row in rows:
        user=dict(zip(GUARD_FALLBACK_ARMS,(0.,.5,1.)))[row['mode']]
        if any(row[k]!=user for k in ('alpha','user_alpha','effective_user_alpha','query_alpha')) or row['method']!='guard_degraded':raise ValueError('guard-fallback user/method identity mismatch')
        if row['initial_fingerprint']!=protocol['bank_state_fingerprints'][str(row['bank_index'])] or row['path_key']!=f"bank{row['bank_index']}_"+row['route']:raise ValueError('guard-fallback bank/path mismatch')
        actual.add((row['bank_index'],row['route'],row['speed'],row['mode'],row['force_N'],row['onset_s'],row['duration_s']))
    if actual!=expected:raise ValueError('complete guard-fallback conditions required')


def new_vs_old_outcomes(rows,conditions):
    result=dict(comparison='guard_degraded versus frozen old task_aware',total_goal_gains=0,total_goal_losses=0,total_contact_rescues=0,total_new_contacts=0,per_user={},details=[])
    for user in (0.,.5,1.):
        old={(r['bank_index'],r['route'],r['event']):r for r in rows if r['effective_user_alpha']==user and r['method']=='task_aware'}
        new={(r['bank_index'],r['route'],r['event']):r for r in rows if r['effective_user_alpha']==user and r['method']=='guard_degraded'}
        if old.keys()!=new.keys() or len(old)!=conditions:raise ValueError('complete new/old matched conditions required')
        stats=dict(conditions=conditions,goal_gains=0,goal_losses=0,contact_rescues=0,new_contacts=0)
        for key in old:
            a=old[key];b=new[key];ah=a['horizons']['10'];bh=b['horizons']['10']
            # A fault/unknown replacement is not counted as a contact rescue.
            gains=bh['goal_arrived'] is True and ah['goal_arrived'] is False;losses=bh['goal_arrived'] is False and ah['goal_arrived'] is True
            rescues=ah['physical_outcome']=='contact' and bh['physical_outcome']=='survived';contacts=ah['physical_outcome']=='survived' and bh['physical_outcome']=='contact'
            for name,value in (('goal_gains',gains),('goal_losses',losses),('contact_rescues',rescues),('new_contacts',contacts)):stats[name]+=int(value)
            result['details'].append(dict(user_alpha=user,bank_index=key[0],route=key[1],event=key[2],old_case=a['case_id'],new_case=b['case_id'],old_goal=ah['goal_arrived'],new_goal=bh['goal_arrived'],old_physical_outcome=ah['physical_outcome'],new_physical_outcome=bh['physical_outcome'],old_actual_ticks=a['active_lower_ticks'],new_actual_ticks=b['active_lower_ticks'],goal_gain=gains,goal_loss=losses,contact_rescue=rescues,new_contact=contacts,old_final_user=a['alpha_rule']['final_at_user'],new_final_user=b['alpha_rule']['final_at_user'],nominal=not bool(a.get('force_N',0.))))
        result['per_user'][str(user)]=stats
        for name in ('goal_gains','goal_losses','contact_rescues','new_contacts'):result['total_'+name]+=stats[name]
    result['any_rescue']=bool(result['total_goal_gains'] or result['total_contact_rescues'])
    result['rescue_contract']='Goal gain=True versus oldFalse; contact rescue requires oldcontact and newsurvived H10, not fault/unknown. Counts retain all losses; no adoption or safety claim.'
    return result


def write_paired_candidate_costs(out,rows,results,arrays,*,candidate,prefix):
    old_label,new_label=('full','masked') if candidate=='speed_query' else ('old','new')
    comparisons=[]
    for label in response_plot_groups(rows):
        pair={r['method']:r for r in response_plot_selection(rows,label) if r['method'] in ('task_aware',candidate)}
        if set(pair)!={'task_aware',candidate}:raise ValueError('complete candidate/old pair required')
        old=next(r for r in results if r['case_id']==pair['task_aware']['case_id']);new=next(r for r in results if r['case_id']==pair[candidate]['case_id'])
        names=(prefix+'_common_actual',prefix+'_common_pre_first_arrival')
        for row,record in ((pair['task_aware'],old),(pair[candidate],new)):
            step=arrays[row['case_id']]['task_step_costs']
            for name,n in zip(names,(min(r['active_lower_ticks'] for r in pair.values()),min(r['prefix']['ticks'] for r in pair.values()))):
                record['windows'][name]=dict(actual_ticks=n,observed_s=n*.005,methods=['task_aware',candidate],three_task_costs=step[:n].sum(axis=0).tolist(),three_task_cost_sum=float(step[:n].sum()))
        item=dict(effective_user_alpha=pair['task_aware']['effective_user_alpha'],bank_index=old['bank_index'],route=old['route'],event=old['event'])
        for name,record,row in ((old_label,old,pair['task_aware']),(new_label,new,pair[candidate])):
            item.update({name+'_case':record['case_id'],name+'_goal_arrived':row['horizons']['10']['goal_arrived'],name+'_physical_outcome':record['actual_physical_outcome'],name+'_actual_ticks':record['actual_ticks'],name+'_first_arrival_s':row['horizons']['10'].get('first_arrival_s')})
        item['windows']={name:{old_label:old['windows'][name],new_label:new['windows'][name],new_label+'_minus_'+old_label+'_three_task_cost_sum':new['windows'][name]['three_task_cost_sum']-old['windows'][name]['three_task_cost_sum']} for name in names}
        comparisons.append(item)
    atomic_json(out/(prefix+'_paired_costs.json'),comparisons)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--include-heldout-labels',action='store_true',help='Explicit final evaluation only after root freezes the fitting recipe; unseals bank1 new labels')
    parser.add_argument('--original-task-rescore-only',action='store_true',help='Add independent original-path three V3 task channels from completed main analysis, no physics')
    parser.add_argument('--include-confirmation-labels',action='store_true',help='Explicit sealed-predictor new-source banks2/5 confirmation only; never unseals historical banks')
    args=parser.parse_args()
    if args.original_task_rescore_only:rescore_original_task_analysis(args.root.resolve())
    else:analyze(args.root,include_heldout_labels=args.include_heldout_labels,include_confirmation_labels=args.include_confirmation_labels)
