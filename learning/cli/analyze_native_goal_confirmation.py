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


def analyze(root,include_heldout_labels=False):
    root=root.resolve();run=root/'evaluation';out=root/'analysis'
    p=json.loads((root/'plan.json').read_text())
    if p.get('mode')=='h75_samestate_bank144':return analyze_bank144(root,p,include_heldout_labels=include_heldout_labels)
    if p.get('mode')=='h75_samestate_tolerance72':return analyze_samestate(root,p)
    if p.get('mode') in ('h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100'):return analyze_response(root,p)
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
        _,one=score(segment,path,config,controller,keys,1.)
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
    np.testing.assert_array_equal(raw['alpha_user'][active],np.ones(n))
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
    if any(r.get('mode')=='samestate_scalar' for r in rows):
        return list(dict.fromkeys(r['event']+'__source'+str(r['source_index']) for r in rows))
    if any('path_key' in r for r in rows):
        return list(dict.fromkeys(r['event']+'__bank'+str(r['bank_index']) for r in rows))
    if any(r.get('mode','').startswith('switch_') for r in rows):
        return list(dict.fromkeys(r['event']+'__'+r['origin'] for r in rows if r['mode'].startswith('switch_')))
    return list(dict.fromkeys(r['event'] for r in rows))


def response_plot_selection(rows,label):
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
        if c['mode']!='samestate_scalar' or c['alpha']!=1. or c['query_alpha']!=c['candidate_alpha']:raise ValueError('same-state user/query identity drift')
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
        fault_in_window=fault,contract='Source task1 through pulse END+3s; preparation fall is negative; faults/nonfinite invalid, missed goal is never a fall.')


def audit_samestate_query(raw,case):
    active=np.asarray(raw['active_tick'],bool);n=int(active.sum());alpha=case['candidate_alpha']
    np.testing.assert_array_equal(raw['alpha_user'][active],np.ones(n));np.testing.assert_array_equal(raw['alpha_exec'][active],np.full(n,alpha))
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
    return dict(candidate_alpha=alpha,state_user_alpha=1.,candidate_requested_at_source_s=1.,
        first_actual_permitted_query_s=float(raw['time'][np.flatnonzero(permitted)[0]]) if permitted.any() else None,
        permitted_ticks=int(permitted.sum()),masked_ticks=int((active&~permitted).sum()),checked='Scalar query held from common source; actual publication and native filters remain auditable')


def samestate_summary(rows):
    groups={};paired=[]
    for source in sorted({r['source_index'] for r in rows}):
        group=[r for r in rows if r['source_index']==source]
        groups[str(source)]=dict(source_origin=group[0]['source_origin'],route=group[0]['route'],n_conditional_suffixes=len(group),
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
    if len(groups) not in (4,8) or any(g['n_conditional_suffixes']!=18 for g in groups.values()):raise ValueError('declared source groups18 conditions required')
    return dict(source_groups=groups,conditional_suffixes=len(rows),independent_source_states=len(groups),
        adoption='no_learner_trained',interpretation='Familiar shared-ancestor state-conditional preparation/hypothetical-onset response only; preparation also changes absolute pulse onset and encountered geometry, so it does not isolate pure timing at the same force phase.Conditional suffixes are not independent source states. No automatic force/state search or classifier training.'),paired


def validate_bank144_inventory(rows,p):
    groups={g['source_index']:g for g in p['source_groups']};actual=[]
    for c in rows:
        group=groups[c['source_index']]
        if any(c.get(k)!=v for k,v in group.items()):raise ValueError('bank source metadata drift')
        if c['split']!=('development' if c['bank_index']==0 else 'label_heldout') or c['alpha']!=1. or c['query_alpha']!=c['candidate_alpha']:raise ValueError('bank split/user/query identity mismatch')
        if not np.isclose(c['onset_s'],1.+c['preparation_s']) or not np.isclose(c['primary_window_end_s'],c['onset_s']+.1+3.):raise ValueError('bank conditional timing mismatch')
        actual.append((c['source_index'],c['candidate_alpha'],c['preparation_s'],c['force_N']))
    expected=list(itertools.product(groups,p['candidate_alphas'],p['preparation_s'],p['forces_N']))
    if sorted(actual)!=sorted(expected) or len(set(actual))!=144:raise ValueError('complete144 conditional inventory required')


def select_bank_label_lanes(index,include_heldout_labels=False):
    """Select BEFORE any suffix raw open, scoring, plotting or label export."""
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


def analyze_bank144(root,p,*,include_heldout_labels=False):
    run=root/'evaluation';out=root/('analysis_final' if include_heldout_labels else 'analysis_development')
    status=json.loads((run/'status.json').read_text());inventory=json.loads((root/'inventory.json').read_text());index=json.loads((run/'case_index.json').read_text())
    validate_bank144_inventory(inventory,p['protocol'])
    if status['phase']!='completed' or status['completed_cases']!=144 or len(index)!=144:raise ValueError('completed144 collection required')
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
    selected=select_bank_label_lanes(index,include_heldout_labels)
    # Nothing below opens unselected bank1 suffix traces or emits their labels.
    out.mkdir(exist_ok=False);helper=Path(__file__).with_name('audit_native_goal_initial.py')
    subprocess.run([sys.executable,'-B',str(helper),str(root),str(out/'initial_carry_review.json')],check=True)
    initial=json.loads((out/'initial_carry_review.json').read_text());sources=json.loads((root/'sources/records.json').read_text());source_by_id={r['source_index']:r for r in sources}
    source=Path(p['samestate_source']['source']);paths={};goals={};traces={};executed={};rebuilt={};rows=[];ticks=0
    # Only selected bank paths/decisions are used for label-bearing diagnostics.
    for key in sorted({analysis_path_key(c) for _,c in selected}):
        with np.load(run/(key+'_path.npz')) as z:paths[key]=dict(z)
        with np.load(source/'evaluation'/(key+'_path.npz')) as z:
            for field in z.files:np.testing.assert_array_equal(paths[key][field],z[field])
        path=paths[key];goals[key]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,a]) for a in (0,1)])
    selected_ids={c['case_id'] for _,c in selected}
    decision_rows=[d for d in json.loads((run/'replanning_decisions.json').read_text()) if d['case_id'] in selected_ids];decisions={d['case_id']:d for d in decision_rows}
    if len(decisions)!=len(decision_rows):raise ValueError('more than one bank replan per case')
    with np.load(root/'sources/host_state.npz') as host:
        if np.any(host['ever_arrived']|host['replanned']):raise ValueError('source antecedent must be original-path/unarrived')
    for lane,c in selected:
        if any(c.get(k)!=v for k,v in inventory[lane].items()):raise ValueError('selected bank inventory drift')
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('selected bank raw SHA drift')
        with np.load(c['raw_file']) as z:raw=dict(z)
        active=np.asarray(raw['active_tick'],bool);n=int(active.sum());ticks+=n
        if not 0<n<=1800 or not np.array_equal(np.flatnonzero(active),np.arange(n)):raise ValueError('bank suffix observation prefix broken')
        np.testing.assert_allclose(raw['time'][active],1.+np.arange(n)*.005,atol=3e-6,rtol=0);np.testing.assert_allclose(raw['v_user'][active],c['speed'],atol=2e-7,rtol=0)
        key=analysis_path_key(c);goal=goals[key]
        np.testing.assert_allclose(c['original_goal_xy'],goal,atol=3e-6,rtol=0);np.testing.assert_allclose(manifest['original_goals_xy'][lane],goal,atol=3e-6,rtol=0)
        query=audit_samestate_query(raw,c);clock=np.rint(raw['time']/.005).astype(int)
        pulse=audit_pulse(dict(raw,physical_tick=clock),dict(force_N=c['force_N'],world_direction=[0.,1.,0.],pulse_onset_task_tick=round(c['onset_s']/.005),pulse_duration_s=.1))
        chain=verify_correction_chain(dict(raw,issued_latent=raw['latent_z'],nominal_reference=raw['nominal'],reference=raw['governed'],correction_offset=raw['offsets']),p['config'],initial_filtered=initial['initial_filtered'][lane],initial_offsets=initial['initial_offsets'][lane])
        physical=dict(raw,physical_tick=clock-200,policy_fault=raw['policy_fault']|raw['continuation_geometry_fault'])
        outcome=goal_arrival_metrics(physical,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=9.)
        relative_arrival=outcome['first_arrival_s']
        if relative_arrival is not None:outcome['first_arrival_s']=relative_arrival+1.
        outcome.update(physical_outcome=physical_outcome(raw['active_tick'],raw['contact_failure'],physical['policy_fault']|raw['lower_fault']|raw['nonfinite'],1800),
            observation_contract='Source task1 was valid/unarrived; observed suffix task1..10, no original prefix rerun or display. First arrival uses original task clock.')
        decision=decisions.get(c['case_id']);replan=run/f'replanned_path_{lane:02d}.npz';changed=None;replan_tick=200+n
        if bool(decision)!=bool(c['replanned']) or replan.exists()!=bool(c['replanned']):raise ValueError('bank replan evidence mismatch')
        if decision:
            replan_tick=decision['task_tick']
            if decision['case_index']!=lane or replan_tick not in range(200,1801,200):raise ValueError('bank replan clock mismatch')
            from sttw_control.goal_replanning import replan_requested
            if not replan_requested(xy=np.asarray(decision['measured_pose'])[:2],goal_xy=goal,cross_track_m=decision['cross_track_m'],follower_invalid=decision['follower_invalid'],already_replanned=False,contact=decision['contact'],fault=decision['fault'],ever_arrived=decision['ever_arrived']):raise ValueError('bank replan lacks measured trigger')
            with np.load(replan) as z:changed=dict(z)
            np.testing.assert_allclose([np.interp(float(changed['goal']),changed['s'],changed['xy'][:,a]) for a in (0,1)],goal,atol=3e-6,rtol=0);executed[c['case_id']]=changed
        data=reconstruct_execution_path_reward(raw,c,paths[key],changed,replan_tick-200,p['config'],p['initial_source']['controller'],manifest['reward_components'],start_tick=200)
        rebuilt[c['case_id']]=data;traces[c['case_id']]=raw
        fault=physical['policy_fault']|raw['lower_fault']|raw['nonfinite'];fatal=np.flatnonzero(active&(raw['contact_failure']|fault));stop=int(fatal[0]) if len(fatal) else n
        task_stop=min(stop,round(relative_arrival/.005)) if relative_arrival is not None else stop
        event=f"prep{c['preparation_s']:g}_F{c['force_N']:+g}"
        rows.append(dict(c,event=event,method=f"alpha{c['candidate_alpha']:g}",source_fingerprint=source_by_id[c['source_index']]['source_fingerprint'],
            source_feature_file='../evaluation/source_features.npz',source_feature_row=lane,diagnostic_file=f'case_{lane:03d}_diagnostics.npz',
            primary=samestate_primary_label(raw,c),horizons={'10':outcome},active_lower_ticks=n,new_suffix_lower_ticks=n,trace_start_s=1.,prefix=dict(ticks=task_stop),
            pulse=pulse,query=query,correction_chain=chain,replan_task_tick=None if decision is None else replan_tick,
            diagnostic_execution_path_user1_cost=float(-data['user1_reward'].sum()),actual_final_xy=raw['actual_xy'][n-1].tolist()))
        numeric={k:v[:n] for k,v in raw.items()};numeric.update(user1_suffix_reward=data['user1_reward'],cumulative_user1_suffix_reward=np.cumsum(data['user1_reward']),
            user1_suffix_reward_components=data['user1_running_components'],cumulative_user1_suffix_components=np.cumsum(data['user1_running_components'],axis=0),terminal_adjustment=data['user1_terminal_adjustment'])
        np.savez_compressed(out/f'case_{lane:03d}_diagnostics.npz',**numeric)
    summary,paired=samestate_summary(rows)
    summary.update(label_selection='final_all_banks_explicitly_unsealed' if include_heldout_labels else 'bank0_development_only',heldout_labels_accessed=include_heldout_labels,
        data_role='Old bank3 training; bank0 development; bank1 new labels held until frozen fitting recipe. Prior bank1 control seen; not independent TEST.')
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),analyzed_suffix_cases=len(rows),
        analyzed_suffix_active_lower_ticks=ticks,active_new_hybrid_prefix_lower_ticks=initial['active_new_prefix_lower_ticks'],maximum_suffix_budget=259200,maximum_total_budget=260000,
        heldout_suffix_labels_accessed=include_heldout_labels,unselected_suffix_files_opened=0,new_physics_steps=0,optimizer_updates=0,
        limitation='No unselected suffix labels/curves/counts exported. Prefix800 is a separate source-collection charge. Numeric validity on selected trajectories does not fit or validate a predictor.'))
    # Existing renderer expects root/analysis; pass an explicit output directory.
    render_response(root,rows,traces,paths,goals,executed,rebuilt,manifest['reward_components'],output_directory=out)
    lines=['# 分源容扰数据：'+('显式解封两bank最终评价' if include_heldout_labels else '仅bank0开发标签'),'',
        '默认不打开、不计算、不绘制、不导出bank1新后缀标签；bank1此前控制工况已见，不能称独立TEST。仅显式最终评价选项才解封。',
        '八个source身份可核查，但默认结果仅含bank0四source的72条件；不从全局活动步数反推或报告bank1失败。原bank3旧72用于训练，本轮bank0开发；未来按source祖先分组，禁止随机按行拆分。',
        '原source前缀不重跑也不拼接。XY仅展示task1..10实测后缀与原始/执行路径；主标签source至pulseEND+3s，准备期摔倒为负、fault为invalid。准备改变绝对起扰时刻与遭遇几何，不隔离纯时序。']
    for label in response_plot_groups(rows):lines+=['',f'## {label}','',f'![Selected suffix XY](full_xy_{label}.png)']
    lines+=['','[所选逐例标签](results.json) · [按source分组](summary.json) · [配对](paired.json) · [身份与单独费用](receipt.json)',
        '本轮不拟合预测器。351源观测含真实速度/历史部署依赖，尚非最终小特征集。执行路径user1奖励仅诊断；四固定1来源复用，一秒hybrid源仅新800lower，后缀物理单独计费。']
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


def analyze_response(root,p):
    run=root/'evaluation';out=root/'analysis';status=json.loads((run/'status.json').read_text())
    confirmation=p.get('mode')=='h75_online_confirmation100'
    online=p.get('mode') in ('h75_online_scalar','h75_online_confirmation100')
    continuation=p.get('mode')=='h75_switching_continuation'
    switching=p.get('mode') in ('h75_channel_switching','h75_switching_continuation');count=100 if confirmation else 18 if online else 36 if switching else 54
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
    if confirmation:validate_confirmation_inventory(inventory,p['protocol'])
    elif online:validate_online_inventory(inventory,p['protocol'])
    elif switching:validate_switching_inventory(inventory,p['protocol'])
    else:validate_response_inventory(inventory,p['protocol'])
    expected={(analysis_path_key(r),r['event'],response_method(r)) for r in inventory}
    if len(expected)!=count or len({r['route'] for r in inventory})!=2 or sum(not r['force_N'] for r in inventory)!=(20 if confirmation else 2 if online else 12 if switching else 6):
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
        channels=audit_response_channels(raw,c)
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
        rows.append(dict(c,method=(response_method(c) if online else ('stay_'+c['origin']) if cached and c.get('origin')!='background' else 'background_fixed0.5' if cached else response_method(c)),source_kind='cached36' if cached and online else 'cached18' if cached else 'new100' if confirmation else 'new18' if online else 'new36' if switching else 'new54',horizons={'10':endpoint},active_lower_ticks=n,pulse=pulse,onset_frame=frame,correction_chain=chain,channels=channels,
            floor_support=audit_floor_support(raw,c,p['config'],p['protocol']['existing_speed_floor_m_s'],initial['initial_nominal_reference'][lane]) if confirmation else None,
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
        np.savez_compressed(out/f'case_{i:02d}_diagnostics.npz',**numeric)
    if ticks-(7200 if continuation else 0)!=status['active_lower_ticks']:raise ValueError('raw tick charge mismatch')
    reused_prefix_proof=audit_continuation_prefix(root,p,rows,raws) if continuation else None
    prefix_proof=audit_switch_prefixes(rows,raws,p['protocol']) if switching else None
    if switching:
        prefix_receipt=json.loads((run/'own_switch_prefix_receipt.json').read_text())
        if not prefix_receipt.get('own_switch') or not prefix_receipt.get('discrete_exact') or (not prefix_receipt.get('all_fields_within_tolerance') and not (p['protocol']['prefix_comparison'].get('floating_descriptive_only') and prefix_receipt.get('floating_descriptive_only'))):
            raise ValueError('runner own-switch prefix receipt incomplete')
    if online:
        for row in rows:
            if row['mode']=='online_scalar':
                row['alpha_rule']=audit_online_rule(raws[row['case_id']],row,paths[analysis_path_key(row)],executed.get(row['case_id']),p['protocol']['rule'])
    summary,paired=confirmation_summary(rows) if confirmation else online_summary(rows) if online else switching_summary(rows) if switching else response_summary(rows)
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    atomic_json(out/'receipt.json',dict(all_cases=len(rows),new_cases=count,cached_cases=len(cache_rows),prefix_proof=prefix_proof,reused_prefix_proof=reused_prefix_proof,numerical_amendment=p['protocol'].get('numerical_amendment'),campaign_budget=p['protocol'].get('total_campaign_lower_ticks'),horizons_s=[10],actual_lower_ticks=ticks,newly_simulated_lower_ticks=status['active_lower_ticks'],reused_source_lower_ticks=7200 if continuation else 0,new_physics_steps=0,optimizer_updates=0,
        plan_sha256=digest(root/'plan.json'),analysis_source_sha256=digest(__file__),initial_carry_sha256=initial['initial_carry_sha256'],
        checked='Raw SHA, full initial audit, config/model/goal/inventory, actual pulse/clock, masked channel composition, correction chain and reconstructed execution-path V3 channels/rewards.',
        limitation='Non-task reward components reused from authenticated raw logs, not independently rederived physics. Original-path distances are descriptive sampled nearest distances; execution-path V3 is not original-user utility after replan.'))
    render_response(root,rows,raws,paths,goals,executed,rebuilt,manifest['reward_components'])
    lines=['# H75通道受扰响应：54条H10','','原始XY与执行路径优先；两tight S×三通道映射×九事件。六条无扰轨迹为本合同重新执行，不复用旧原生停止合同。',
        '','主终点：原世界目标1m首达；整10s真实触地单列。早期domain仅遥测，不算触地；重规划后不声称严格完成原S。',
        '','相关熟悉DEV、同祖先；无新训练。各法在起扰前已有不同状态，不能把转向通道替换解释为独立于速度的因果效应。']
    if confirmation:
        lines=['# 两初态五方法100回合确认','','全部100为本轮实际H10轨迹，不复用旧物理轨迹。两prepared bank不是独立训练种子；每bank/route/event五法完整原生初态配对。全部state/user/reward alpha=1，固定Actor查询alpha0/.5/1另列。',
               '','原世界goal1m与真实触地主终点；Zero+既有1.5m/s下限若未触发须标为本组等同Zero，不据此宣称排除了速度反馈支持。执行路径user1奖励仅诊断，重规划后不称原S严格完成。']
    elif online:
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
    lines+=['','执行路径V3图以相同user1重评分；运行本身alpha评分另行核验并保留。逐步/累计奖励和分项仅为诊断，不是重规划后原用户路径效用。原路线距离与执行路径误差分别保存；全窗距离包含首达后运动，首达/真实失败前缀另列；未完成短窗不能当更低完成代价。',
        '世界Y脉冲投影使用日志起扰前upper yaw作一次onset-frame近似，不是逐物理子步积分。完整剂量/截断暴露、首达早于起扰、名义重规划逐例列出。4s若重规划，先规划后执行含脉冲的步。',
        '[逐例结果](results.json) · [无扰/受扰汇总](summary.json) · [全部配对得失](paired.json) · [核验](receipt.json)',
        '全部声明条件逐步原始帧、实际末点、重建奖励/累计分项、真速度/轮速代理与误差保留于case诊断NPZ；图见diagnostics。无20s数据或结论。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def render_response(root,rows,raws,paths,goals,executed,rebuilt,keys,output_directory=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=output_directory or root/'analysis';diag=out/'diagnostics';diag.mkdir();routes=sorted(paths)
    palette=('#0072B2','#D55E00','#008060','#CC79A7','#555555')
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
            fig.suptitle(event+' / '+('first-arrival/contact prefix' if task else 'full actual H10 endpoint')+'; dashed=executed replan; dots=actual worldY exposure')
            fig.savefig(out/f'{"task" if task else "full"}_xy_{event}.png',dpi=140);plt.close(fig)
        for route in routes:
            selected=[r for r in group if analysis_path_key(r)==route]
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
            for ax,label in zip(axes,('Speed: true/dotted wheel/dashed user (m/s)','True - wheel proxy (m/s)','True - original request (m/s)','Execution-path cross-track (m)','Execution-path heading error (rad)','Original goal distance (m)','Roll (rad)','Roll rate (rad/s)','Steer (rad)','Actual worldY mean force/tick (N)','Original route sample distance (m)','Diagnostic user1 reward / actual step','Diagnostic user1 cumulative reward')):
                ax.set_ylabel(label);ax.grid(alpha=.15);ax.legend(fontsize=6)
            for k,name in enumerate(list(keys)+['terminal replacement']):cax[k,0].set_ylabel(name);cax[k,1].set_ylabel('Cumulative '+name)
            cax[0,0].legend(fontsize=6)
            for ax in [axes[-1],*cax[-1]]:ax.set_xlabel('Actual post-tick time (s)')
            title=route+' / '+event+' / actual observed endpoints; vertical lines=replan, gray=intended pulse; no new training'
            fig.suptitle(title+'\nExecution-path errors/rewards are diagnostic; original goal unchanged')
            comp.suptitle(title+'\nSigned per-step and cumulative user1 execution-path reward components; no original-user utility ranking')
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



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--include-heldout-labels',action='store_true',help='Explicit final evaluation only after root freezes the fitting recipe; unseals bank1 new labels')
    args=parser.parse_args();analyze(args.root,include_heldout_labels=args.include_heldout_labels)
