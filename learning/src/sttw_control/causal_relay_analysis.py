"""Read factual/cached causal relay trajectories and matched reference rules."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .causal_recovery_relay import MODES,relay_choice,relay_contrasts


def analyze(root):
    from prepare_short_recovery_campaign import register
    from .anticipatory_analysis import cost_accounting,save_csv
    from .anticipatory_run import physical_outcome
    from .rejoin_comparison import geometric_task_trace
    from .survival_endpoint_task_metrics import task_metrics
    from .preference_trace_rescore import rescore_trace_alpha
    root=Path(root).resolve();state=json.loads((root/'status.json').read_text())
    if state['phase']!='completed' or state['completed_new_cases']!=30:raise ValueError('complete30 factual new cases required')
    out=root/'analysis';out.mkdir(exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),completed_cases=0,physics_steps=0,optimizer_updates=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
    update();register(out/'ACTIVE_RUN.json','STTW_CONTROL 因果接管50逻辑轨迹只读分析',out/'status.json')
    try:
        p=json.loads((root/'plan.json').read_text());inv=json.loads((root/'inventory.json').read_text())
        assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for f,sha in p['sealed_files'].items():assert digest(f)==sha
        cfg=json.loads((root/'common_environment_config.json').read_text())
        assert digest(p['path_file'])==p['path_sha256']
        with np.load(p['path_file'],allow_pickle=False) as z:path={k:z[k] for k in z.files}
        cache={c['case_index']:c for c in inv['cached']}
        new={c['case_id']:c for c in json.loads((root/'physical/results.json').read_text())}
        index={c['case_id']:c for c in json.loads((root/'physical/case_index.json').read_text())}
        rows=[];traces={};audits=[];new_ticks=old_ticks=0
        for i,c in enumerate(inv['cases']):
            reused=i in cache
            if reused:
                cp=cache[i];file=Path(cp['source_file']);sha=cp['source_file_sha256'];initial=cp['initial_xy'];expected=cp['source_case']['physical_outcome']
            else:
                item=index[c['case_id']];file=root/item['file'];sha=item['sha256'];initial=item['initial_xy'];expected=new[c['case_id']]['physical_outcome']
                for k,v in c.items():assert new[c['case_id']][k]==v
            assert digest(file)==sha
            with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            a=raw['active_tick'].astype(bool);ticks=int(a.sum())
            assert np.array_equal(raw['physical_tick'][a],np.arange(ticks))
            outcome=physical_outcome(a,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            assert outcome==expected and outcome!='unknown'
            if reused:old_ticks+=ticks
            else:new_ticks+=ticks
            geometry=geometric_task_trace(raw,path,cfg,initial_xy=initial)
            metrics=task_metrics(dict(raw,**geometry,domain_exit=np.zeros_like(a),task_reward_valid=a),
                speed=c['requested_speed_m_s'],goal_s=float(path['goal']),config=cfg)
            category='contact' if outcome=='contact' else 'safe_task' if metrics['task_recovered'] else 'safe_task_failed'
            if metrics['task_recovered'] is None:category='unknown'
            q=rescore_trace_alpha(raw,alpha=c['alpha'],config=cfg)
            np.testing.assert_allclose(q['scored_reward'],raw['scored_tick_reward'],rtol=2e-5,atol=2e-6)
            costs=cost_accounting(raw,scale=cfg['path_reward']['scale'],dt=.005)
            if reused:
                first=None;gate_seconds=None;selected_seconds=None;episodes=None
            else:
                _verify_relay(raw,c)
                hits=np.flatnonzero(a&raw['relay_trigger'].astype(bool))
                first=None if not len(hits) else int(hits[0])
                gate_seconds=float(np.sum(a&raw['relay_active'].astype(bool))*.005)
                selected_seconds=float(np.sum(a&raw['selected_B'].astype(bool))*.005)
                flags=raw['relay_active'].astype(bool)&a
                episodes=int(np.sum(flags&~np.r_[False,flags[:-1]]))
            impulse=raw['applied_force_impulse'][a].sum(0,dtype=np.float64)
            expected_impulse=c['force_N']*c['pulse_duration_s']*np.asarray(c['world_direction'])
            row=dict(c,category=category,physical_outcome=outcome,physical_ticks=ticks,**metrics,**costs,
                reused=reused,raw_file=str(file),raw_sha256=sha,first_trigger_tick=first,
                trigger_delay_from_onset_s=None if first is None or not c['force_N'] else (first-c['pulse_onset_task_tick'])*.005,
                gate_active_s=gate_seconds,selected_B_s=selected_seconds,gate_activation_episodes=episodes,
                max_original_path_error_m=float(np.max(abs(geometry['path_cross_track'][a]))),
                full_pulse_exposure=bool(np.allclose(impulse,expected_impulse,atol=.025,rtol=.05)))
            rows.append(row);traces[c['case_id']]=dict(raw,**geometry)
            audits.append(dict(case_id=c['case_id'],raw_sha256=sha,reused=reused,selection_log_verified=not reused,
                selection_annotation='Derived exact constant diagonal Actor label' if reused else 'Actual wrapper record verified'))
        assert len(rows)==50 and new_ticks==state['new_active_lower_ticks']
        contrasts=relay_contrasts(rows)
        atomic_json(out/'results.json',rows);save_csv(out/'results.csv',rows)
        atomic_json(out/'contrasts.json',contrasts);atomic_json(out/'raw_audit.json',audits)
        counts={m:{cat:sum(r['mode']==m and r['category']==cat for r in rows) for cat in ['safe_task','safe_task_failed','contact','unknown']} for m in MODES}
        atomic_json(out/'counts.json',counts)
        figures=_render(out,rows,traces,path,cfg)
        atomic_json(out/'receipt.json',dict(logical_cases=50,new_cases=30,cached_cases=20,new_actual_lower_ticks=new_ticks,cached_historical_lower_ticks=old_ticks,
            figures=figures,physics_steps=0,optimizer_updates=0,visual_review_complete=False,
            scope='Current measured-rate causal relay on selected DEV conditions; no future event input, no independent test or safety guarantee.'))
        lines=['# 现时传感器接管：完整50逻辑轨迹','',
            'A=原task64，B=新plain64；当前传感器gate控制短时借用B或同gate速度支持，无事件时间输入。','',
            '|组合|未摔且任务完成|未摔但任务失败|接触|未知|','|---|---:|---:|---:|---:|']
        for mode,items in counts.items():lines.append('|'+mode+'|'+'|'.join(str(items[k]) for k in ['safe_task','safe_task_failed','contact','unknown'])+'|')
        lines+=['',f'实际新积分{new_ticks}个5ms步，缓存历史{old_ticks}步单独记录，0optimizer。',
            '全部XY/分项已生成待目视裁决；实际sensor输入/滞回/选择已复核，不将确定性DEV分支当独立重复。',
            '全初末物理/ESO/history在原新与缓存run保留，不重置；未逐tick导出的ESO不补造。',
            '收益仍须超过同gate/恒定速度规则并保留原alpha任务，再作独立确认。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');update(phase='completed',stage='numeric_and_figures_complete_review_pending',completed_cases=50)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


def _verify_relay(raw,case):
    import jax
    import jax.numpy as j
    from .reference_reserve_gate import initial_gate,gate_step
    a=raw['active_tick'].astype(bool);ticks=np.flatnonzero(a & (raw['physical_tick']%4==0))
    rate=raw['relay_input_roll_rate'][ticks]
    def step(state,value):
        end,_,receipt=gate_step(state,j.zeros(2),rho=j.float32(0),roll_rate=value,mode=2,dt=.02)
        return end,(receipt['active'],receipt['trigger'],end.remaining_s)
    _,(active,trigger,remaining)=jax.jit(lambda x:jax.lax.scan(step,initial_gate(),x))(j.asarray(rate))
    np.testing.assert_array_equal(np.asarray(active),raw['relay_active'][ticks])
    np.testing.assert_array_equal(np.asarray(trigger),raw['relay_trigger'][ticks])
    np.testing.assert_allclose(np.asarray(remaining),raw['relay_remaining_s'][ticks],atol=1e-6)
    later=ticks[ticks>0]
    np.testing.assert_allclose(raw['relay_input_roll_rate'][later],raw['phi_dot'][later-1],atol=2e-6,rtol=2e-5)
    use_b,profile=relay_choice(case['mode_index'],raw['relay_active'])
    np.testing.assert_array_equal(raw['desired_B'],a&np.asarray(use_b))
    np.testing.assert_array_equal(raw['selected_B'],a&raw['actor_action_valid'].astype(bool)&np.asarray(use_b))
    np.testing.assert_array_equal(raw['relay_profile'][a],np.asarray(profile)[a])


def _render(out,rows,traces,path,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .live_endpoint_analysis import block_reward_series
    from .path_command_reward import configured_cost_keys
    colors=dict(zip(MODES,['#0072B2','#CC79A7','#D55E00','#009E73','#E69F00']));styles=dict(zip(MODES,[':', '-','--','-.',':']))
    groups={};figures=[]
    for r in rows:groups.setdefault((r['alpha'],r['event_id']),[]).append(r)
    overview,oa=plt.subplots(2,5,figsize=(16,8),layout='constrained');reference=path['xy'][path['s']<=float(path['goal'])]
    for number,((alpha,event),group) in enumerate(groups.items()):
        fig,grid=plt.subplots(4,3,figsize=(14,14),layout='constrained');axes=list(grid.flat);ax=oa.flat[number]
        clouds=[reference]
        for x in [axes[0],ax]:x.plot(reference[:,0],reference[:,1],':',color='.55',lw=1);x.scatter(*reference[-1],marker='*',color='k')
        for r in group:
            d=traces[r['case_id']];a=d['active_tick'];t=d['physical_tick'][a]*.005;xy=d['actual_xy'][a];clouds.append(xy)
            style=dict(color=colors[r['mode']],ls=styles[r['mode']],label=r['mode'])
            for x in [axes[0],ax]:x.plot(xy[:,0],xy[:,1],**style);x.scatter(*xy[-1],color=colors[r['mode']],marker='x' if r['category']=='contact' else 'o',s=25)
            for i,key in [(1,'path_cross_track'),(3,'path_heading_error'),(4,'phi'),(5,'actual_delta')]:axes[i].plot(t,d[key][a],**style)
            axes[2].plot(t,d['actual_forward_speed'][a]-1.5,**style)
            series=block_reward_series(d,config=cfg);u=series['time_s'];reward=series['scored_reward']
            axes[6].plot(u,reward,**style);axes[7].plot(u,np.cumsum(reward),**style)
            chosen=d['selected_B'] if 'selected_B' in d else a&d['actor_action_valid']&(r['mode']=='BB')
            axes[8].plot(t,chosen[a].astype(float),**style)
            axes[10].plot(t,d['phi_dot'][a],**style)
            axes[11].plot(t,d['actor_action_valid'][a].astype(float),**style)
            if 'relay_active' in d:
                axes[9].plot(t,d['relay_active'][a].astype(float),color=colors[r['mode']],ls='-')
                support=d['actor_action_valid'][a]&(d['relay_profile'][a]==7)
                axes[9].plot(t,support.astype(float),color=colors[r['mode']],ls=':')
                axes[10].plot(t,d['relay_input_roll_rate'][a],color=colors[r['mode']],ls=':',alpha=.5)

            np.savez_compressed(out/f'series_{number:02d}_{r["mode"]}.npz',**series,cumulative_reward=np.cumsum(reward),
                cumulative_running_components=np.cumsum(series['signed_running_components'],axis=0),
                cost_keys=np.asarray(configured_cost_keys(cfg)),actual_time_s=t,xy=xy,reference=d['reference'][a],
                nominal_reference=d['nominal_reference'][a],actual_speed=d['actual_forward_speed'][a],actual_delta=d['actual_delta'][a])
        extent=np.vstack(clouds);lo=extent.min(0)-.4;hi=extent.max(0)+.4
        for x in [axes[0],ax]:x.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)');x.set_aspect('equal',adjustable='box');x.grid(alpha=.2)
        ax.set_title(f'alpha{alpha:g} {event}');axes[0].legend(fontsize=7)
        titles=['Actual XY / original goal','Signed path error (m)','Speed error (m/s)','Signed heading error (rad)',
            'Roll (rad)','Actual steering (rad)','Reward per20ms block','Cumulative originalalpha reward','Actual legal Actor B selected','Gate (solid) / legal speed support (dotted)',
            'Measured roll rate (rad/s)','Actor permitted (1) / fallback (0)']
        for i,(x,title) in enumerate(zip(axes,titles)):
            x.set_title(title);x.grid(alpha=.2)
            if i:
                x.set(xlabel='Task time (s)',xlim=(0,10));x.axvline(group[0]['pulse_onset_task_tick']*.005,color='k',ls=':',lw=.8)
                if group[0]['force_N']:x.axvspan(group[0]['pulse_onset_task_tick']*.005,group[0]['pulse_onset_task_tick']*.005+.1,color='red',alpha=.08)
        for threshold in [-.8,-.4,.4,.8]:axes[10].axhline(threshold,color='.5',ls='--' if abs(threshold)==.8 else ':',lw=.7)
        fig.suptitle(f'Causal sensor relay / matched rules | alpha{alpha:g} {event} | A original / B plain64')
        name=f'paired_{number:02d}_a{alpha:g}.png';fig.savefig(out/name,dpi=115);plt.close(fig);figures.append(name)
    handles,labels=oa[0,0].get_legend_handles_labels();overview.legend(handles,labels,loc='outside lower center',ncol=4)
    overview.suptitle('All10 conditions /50 logical trajectories | measured-rate causal relay, selected DEV')
    name='all_xy.png';overview.savefig(out/name,dpi=120);plt.close(overview);figures.append(name)
    return figures
