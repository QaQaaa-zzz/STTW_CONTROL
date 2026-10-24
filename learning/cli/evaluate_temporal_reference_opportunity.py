"""Finite privileged DEV opportunity comparison; no training or online selector."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

from sttw_control.preference_best import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    import numpy as np
    from prepare_short_recovery_campaign import register
    from sttw_control.piecewise_reference import mean_matched_library
    worktree=Path(__file__).resolve().parents[2]
    old=worktree/'runs/reference_prefix_return_feasibility_v1/evaluation'
    source=json.loads((old/'plan.json').read_text())
    constants,sequences=mean_matched_library(20261024)
    assert len(constants)==14 and len(sequences)==16
    contexts=[]
    for route,speed,force,rescue,mask in [
        ('left90_R1p1',1.8,32.,[0.,-.04],[False,True]),
        ('right90_R1p1',1.5,-32.,[.15,0.],[True,False])]:
        selected=[s for s in source['states'] if s['route']==route and s['speed']==speed and s['alpha']==0.]
        assert len(selected)==1
        state=selected[0]
        assert digest(state['snapshot'])==state['snapshot_sha256']
        with Path(state['snapshot']).open('rb') as f:snapshot=pickle.load(f)
        remaining=2000-int(snapshot.tick)
        roster=constants+sequences+[
            dict(candidate='parent',family='parent',target_deltas=[[0.,0.],[0.,0.]],override_mask=[False,False]),
            dict(candidate='single_channel',family='single_channel',target_deltas=[rescue,rescue],override_mask=mask)]
        contexts.append(dict(state=state,force_N=force,remaining_lower_ticks=remaining,roster=roster))
    root.mkdir(exist_ok=False)
    sources=list((worktree/'learning/src/sttw_control').rglob('*.py'))+[
        Path(__file__).resolve(),worktree/'learning/cli/probe_reference_prefix_return.py']
    hashes={str(p):digest(p) for p in sources}
    for p in sources:
        dest=root/'source_at_launch'/p.relative_to(worktree);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    cases=[]
    for context in contexts:
        cases += [{**context['state'],**candidate,'force_N':context['force_N']} for candidate in context['roster']]
    maximum=sum(c['remaining_lower_ticks']*len(c['roster']) for c in contexts)
    assert len(cases)==64 and maximum==97536
    directory=root/'execution';directory.mkdir()
    plan={'project':'STTW_CONTROL','experiment':'temporal_mean_matched_reference_opportunity',
          'purpose':'Finite two-stage reference library versus per-sequence mean constant controls.',
          'source_declaration':source['source_declaration'],
          'path_evidence_directory':source['path_evidence_directory'],
          'path_evidence_atol_m':source.get('path_evidence_atol_m',1e-6),
          'states':[c['state'] for c in contexts],
          'cases':cases,'declared_cases':64,'source_sha256':hashes,
          'prefix_start_delay_s':.04,'prefix_duration_s':.4,'piecewise_segment_durations_s':[.2,.2],
          'pulse_duration_s':.1,'original_task_deadline_s':10.,'maximum_new_lower_ticks':maximum,
          'role':'DEV_privileged_opportunity','implementation':'opt-in piecewise selection; original environment/filter/physical carry',
          'information_boundary':'Authentic onset snapshots and known signed force used offline; not deployed detection or selection.'}
    atomic_json(directory/'plan.json',plan)
    metadata={str(p):digest(p) for p in [old/'plan.json',Path(source['source_declaration'])]}
    design=dict(project='STTW_CONTROL',purpose='Compare fixed finite temporal references with their constant mean-target controls.',
                seed=20261024,contexts=contexts,physical_cases=64,logical_slots=68,
                aliases='Two duplicate mean targets per context reuse one physical result; no independent repetition.',
                candidate_anchor='snapshot.offsets for BOTH segments, never reanchor',
                constant_control='Equal pre-filter target mean per sequence; actual effort/dose may differ.',
                plan=str(directory/'plan.json'),plan_sha256=digest(directory/'plan.json'),source_sha256=hashes,
                metadata_sha256=metadata,maximum_new_lower_ticks=maximum,training_steps=0,model_fits=0,
                ranking='Only full no-contact and full intended pulse eligible. Report no-contact first, task separately. Among task-success candidates compare original alpha0 return over full remaining window. Among task-failures show upper-valid duration; no short-return winner.',
                stop='One frozen library/two contexts/.04 delay/.4 duration/32N. No extra stages, force, seed, time or amplitude search. If no useful benefit over mean controls, stop this library. Even positive is privileged finite DEV opportunity, not optimality, deployability, cross-alpha or new search algorithm.')
    atomic_json(root/'design.json',design)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=64,logical_slots=68,completed_cases=0,active_lower_ticks=0,training_steps=0,model_fits=0))
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 时序与均值常值参考机会对照及全分析',root/'status.json')
    register(directory/'ACTIVE_RUN.json','STTW_CONTROL 64条时序与均值常值参考后缀',directory/'status.json')
    print(root/'design.json')


def report(root,design):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sttw_control.balance_probe import classify_contact_balance
    from sttw_control.recovery_margin_scan import outcome_labels
    from sttw_control.temporal_reference_evidence import pair_transitions
    directory=Path(design['plan']).parent
    manifest=json.loads((directory/'manifest.json').read_text());cfg=manifest['config']
    rows=json.loads((directory/'results.json').read_text())['cases'];assert len(rows)==64
    out=root/'analysis';out.mkdir(exist_ok=False)
    cache={};hashes={};total=0
    for row in rows:
        path=directory/(row['name']+'.npz');hashes[str(path)]=digest(path)
        with np.load(path) as archive:data={k:archive[k] for k in archive.files}
        total+=len(data['time']);cache[row['name']]=data
        duration=10.-round(float(data['time'][0])/.005)*.005
        physical=classify_contact_balance(dict(data,time=data['relative_time']),horizon_s=duration)
        for key in ('balance_label','active_ticks','event_contact','full_observation'):
            assert physical[key]==row[key],(row['name'],key)
        with np.load(directory/(row['route']+'_path.npz')) as p:goal=float(p['goal'])
        task=outcome_labels(data,speed=row['speed'],goal_s=goal,working_roll=cfg['stage_B_baseline']['working_roll_rad'],horizon=10.,hold_s=cfg['stage_B_baseline']['hold_s'])
        assert task['task_recovered']==row['task_recovered']
        impulse=data['applied_force_impulse'].sum(axis=0,dtype=np.float64)
        np.testing.assert_allclose(impulse,row['impulse_Ns'],atol=1e-8)
        intended=np.linalg.norm(row['expected_impulse_Ns'])
        exposure=abs(np.linalg.norm(impulse)-intended)<=.01*intended
        row['full_force_exposure']=bool(exposure)
        row['eligible_no_contact']=bool(exposure and row['full_observation'] and row['balance_label']=='NO_FLOOR_CONTACT')
        row['upper_active_seconds']=float(data['upper_active'].sum()*.005)
        row['override_seconds']=float(data['prefix_override'].sum()*.005)
        row['observed_seconds']=len(data['time'])*.005
        row['reference_offset_abs_integral_while_override']=(.005*np.sum(
            np.abs(data['governed']-data['nominal'])*data['prefix_override'][:,None],axis=0,dtype=np.float64)).tolist()
        row['max_reference_rate_while_override']=np.max(np.abs(np.diff(data['governed'],axis=0,prepend=data['governed'][:1]))/.005*data['prefix_override'][:,None],axis=0,initial=0).tolist()
        row['reference_clip_lower_ticks']=int(data['reference_clip_channels'].any(axis=1).sum())
        row['rate_clip_lower_ticks']=int(data['correction_rate_clipped'].sum())
        mask=np.asarray(row['override_mask'],bool)
        ticks=np.rint(data['relative_time']/.005).astype(int)
        expected=data['upper_active'].astype(bool)&mask.any()&(ticks>=8)&(ticks<88)
        np.testing.assert_array_equal(data['prefix_override'],expected)
        untouched=data['upper_active'][:,None]&~(expected[:,None]&mask)
        np.testing.assert_array_equal(data['executed_policy_latent'][untouched],data['original_policy_latent'][untouched])
        index=rows.index(row)
        targets=np.asarray(manifest['realized_targets'][index])
        selected=expected[:,None]&mask
        z=data['executed_policy_latent'];a=np.tanh(z);c=cfg['residual_interface']
        actual=np.stack([a[:,0]*np.where(a[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[:,1]*c['steer_scale_rad']],axis=1)
        target=targets[np.clip((ticks-8)//40,0,1)]
        np.testing.assert_allclose(actual[selected],target[selected],atol=1e-6)
        ix=np.arange(0,len(data['time']),4)
        valid=np.logical_or.reduceat(np.isfinite(data['scored_tick_reward']),ix)
        reward=np.add.reduceat(np.nan_to_num(data['scored_tick_reward']).astype(np.float64),ix)
        costs=-.005*cfg['path_reward']['scale']*np.add.reduceat(np.nan_to_num(data['effective_costs']).astype(np.float64),ix,axis=0)
        reward[~valid]=np.nan;costs[~valid]=np.nan
        np.testing.assert_allclose(np.nansum(reward),row['logged_suffix_return_while_upper_active'],atol=1e-8)
        remainder=reward-costs.sum(axis=1)
        np.savez_compressed(out/(row['name']+'_rewards.npz'),time=data['time'][ix],step_reward=reward,signed_components=costs,failure_remainder=remainder,component_names=manifest['reward_components'])
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for k,name in enumerate(manifest['reward_components']):ax.plot(data['time'][ix],costs[:,k],label=name)
        ax.plot(data['time'][ix],remainder,'k--',label='failure remainder');ax.set_yscale('symlog',linthresh=.02)
        ax.set(title=row['route']+' / '+row['candidate'],xlabel='task seconds',ylabel='signed reward /20ms');ax.legend(fontsize=6,ncol=3)
        fig.savefig(out/(row['name']+'_components.png'));plt.close(fig)
    pairs=[];contexts=[]
    for context in design['contexts']:
        route=context['state']['route'];subset=[r for r in rows if r['route']==route];by_id={r['candidate']:r for r in subset}
        fig,axes=plt.subplots(4,4,figsize=(22,20),layout='constrained')
        with np.load(directory/(route+'_path.npz')) as p:ref=p['xy'];ref_s=p['s']
        def draw(ax,selected):
            progress=[]
            for row,color in zip(selected,['black','tab:orange']):
                data=cache[row['name']];valid=data['path_progress'][np.isfinite(data['path_progress'])]
                if len(valid):progress.append(float(valid.max()))
                exits=np.flatnonzero(data['domain_exit']);cut=int(exits[0]) if len(exits) else len(data['time'])-1
                label=row['candidate']+'; safe='+str(row['eligible_no_contact'])+'; task='+str(row['task_recovered'])
                ax.plot(*data['actual_xy'][:cut+1].T,color=color,label=label)
                if len(exits):ax.plot(*data['actual_xy'][cut:].T,':',color=color);ax.scatter(*data['actual_xy'][cut],marker='s',color=color)
                ax.scatter(*data['actual_xy'][0],marker='^',color=color,s=12)
                ax.scatter(*data['actual_xy'][-1],marker='x' if row['event_contact'] else 'o',color=color,s=16)
            ax.plot(*ref[ref_s<=max(progress,default=0)+1].T,'--',color='gray',label='original path')
            ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='x /m',ylabel='y /m');ax.legend(fontsize=5)
        for i,seq in enumerate(context['roster'][14:30]):
            temporal=by_id[seq['candidate']];constant=by_id[seq['mean_candidate']]
            draw(axes.flat[i],[constant,temporal]);axes.flat[i].set_title(seq['candidate']+' vs '+seq['mean_candidate'],fontsize=9)
            pairs.append(dict(route=route,temporal=temporal['name'],constant=constant['name'],
                              **pair_transitions(constant,temporal)))
            tf,aa=plt.subplots(7,1,figsize=(10,14),layout='constrained')
            for row,color in [(constant,'black'),(temporal,'tab:orange')]:
                data=cache[row['name']];t=data['time']
                for ax,key in zip(aa[:4],['actual_forward_speed','path_cross_track','path_heading_error','phi']):ax.plot(t,data[key],color=color,label=row['candidate'])
                aa[4].plot(t,data['governed'][:,0],color=color);aa[4].plot(t,data['governed'][:,1],'--',color=color)
                with np.load(out/(row['name']+'_rewards.npz')) as z:
                    aa[5].plot(z['time'],z['step_reward'],color=color);aa[6].plot(z['time'],np.where(np.isfinite(z['step_reward']),np.nancumsum(z['step_reward']),np.nan),color=color)
                for ax in aa:ax.axvspan(t[0],t[0]+.1,color='gray',alpha=.1);ax.axvline(t[0]+.04,color='gray',ls=':');ax.axvline(t[0]+.24,color='gray',ls='--');ax.grid(alpha=.2)
            for ax,label in zip(aa,['speed /m/s','path /m','heading /rad','roll /rad','reference speed/steer','reward /20ms','cumulative reward']):ax.set_ylabel(label)
            aa[0].legend();aa[-1].set_xlabel('task seconds');aa[5].set_yscale('symlog',linthresh=.1);aa[6].set_yscale('symlog',linthresh=.1)
            tf.suptitle(route+' '+seq['candidate']);tf.savefig(out/(route+'_'+seq['candidate']+'_timeseries.png'));plt.close(tf)
        fig.savefig(out/(route+'_all_mean_pairs_xy.png'));plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained');draw(ax,[by_id['parent'],by_id['single_channel']]);fig.savefig(out/(route+'_controls_xy.png'));plt.close(fig)
        winners=[r for r in subset if r['eligible_no_contact'] and r['task_recovered']]
        winner=max(winners,key=lambda r:r['logged_suffix_return_while_upper_active'])['candidate'] if winners else None
        contexts.append(dict(route=route,actual_cases=len(subset),safe=sum(r['eligible_no_contact'] for r in subset),task_success=sum(r['task_recovered'] for r in subset),best_joint_full_window_return_candidate=winner))
    assert total<=design['maximum_new_lower_ticks']
    summary=dict(contexts=contexts,pairs=pairs,cases=rows,raw_sha256=hashes,active_lower_ticks=total,physical_cases=64,logical_slots=68,training_steps=0,model_fits=0)
    atomic_json(out/'analysis.json',summary)
    lines=['# 时序参考与配对均值常值：离线机会诊断','']
    for context in contexts:
        route=context['route'];lines.append(f'[{route}全部配对XY]({route}_all_mean_pairs_xy.png) · [原策略/单通道对照XY]({route}_controls_xy.png)')
    lines+=['','64条实际后缀，68逻辑槽；重复均值常值引用同一结果，不是独立重复。已知起扰快照/方向的特权DEV，不是在线方法。两段均锚定捕获offset；.04秒延迟，.4秒总窗，.2秒切段，原截止10秒，域退出不恢复任务资格。','', '|路径|均值常值→时序防摔gain/loss|联合gain/loss|','|---|---|---|']
    for context in contexts:
        subset=[p for p in pairs if p['route']==context['route']]
        lines.append(f"|{context['route']}|{sum(p['safe_gain'] for p in subset)}/{sum(p['safe_loss'] for p in subset)}|{sum(p['joint_gain'] for p in subset)}/{sum(p['joint_loss'] for p in subset)}|")
    lines+=['','配对共享常值，不能把逻辑pair当独立重复。目标均值相同不保证实际参考剂量或能量相同。所有候选/接触/任务/暴露/观察长度在analysis.json；不以提前退出的较短return选优。全64条20ms奖励、带符号分项和失败余项已保存。',f'新增{total}lower ticks；0训练/拟合。有限库未找到不等于物理不可恢复；若无实用增量停止此库，不加段/改时长/换优化器补考。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return total


def run(root):
    design=json.loads((root/'design.json').read_text())
    state=dict(phase='running',pid=os.getpid(),completed_cases=0,declared_cases=64,active_lower_ticks=0,training_steps=0,model_fits=0)
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    update(stage='preflight')
    try:
        for p,sha in {**design['source_sha256'],**design['metadata_sha256'],design['plan']:design['plan_sha256']}.items():
            if digest(p)!=sha:raise ValueError('sealed source changed: '+p)
        directory=Path(design['plan']).parent
        with (directory/'execution.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('probe_reference_prefix_return.py')),'--plan',design['plan']],stdout=log,stderr=subprocess.STDOUT)
            update(stage='physics',child_pid=child.pid);code=child.wait()
        status=json.loads((directory/'status.json').read_text())
        update(active_lower_ticks=status['active_lower_ticks'],completed_cases=status['completed_cases'])
        if code or status['phase']!='completed':raise RuntimeError('physics child exit '+str(code))
        update(stage='analysis');ticks=report(root,design);assert ticks==status['active_lower_ticks']
        update(phase='completed',stage='reported',report=str(root/'analysis/REPORT.md'))
    except BaseException as exc:
        update(phase='error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','run'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'prepare':prepare,'run':run}[args.mode](args.output.resolve())
