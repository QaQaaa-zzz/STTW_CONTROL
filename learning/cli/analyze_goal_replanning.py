"""Read all fixed48 trajectories: original destination, two horizons, full XY."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sttw_control.preference_best import atomic_json,digest
from sttw_control.anticipatory_run import physical_outcome
from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics
from sttw_control.reference_preservation_confirmation import audit_pulse
from sttw_control.support_permission_audit import verify_correction_chain

ROLES=('original_full','replan_full','original_zero','replan_zero')


def analyze(root):
    root=root.resolve();physical=root/'physical';out=root/'analysis'
    status=json.loads((physical/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=48:
        raise ValueError('all48 numerical trajectories must finish; no partial result selection')
    p=json.loads((root/'plan.json').read_text());index=json.loads((physical/'case_index.json').read_text())
    if len(index)!=48 or len({c['case_id'] for c in index})!=48:raise ValueError('complete unique fixed48 required')
    out.mkdir(exist_ok=False)
    paths={};goals={};trees={}
    for context,c in p['contexts'].items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('original reference changed')
        with np.load(c['path']) as z:paths[context]={k:z[k] for k in z.files}
        path=paths[context];goal=float(path['goal'])
        goals[context]=np.array([np.interp(goal,path['s'],path['xy'][:,i]) for i in (0,1)])
        trees[context]=cKDTree(np.vstack([path['xy'][path['s']<=goal],goals[context]]))
    # Actual initial carry readback, excluding only declared task geometry/alpha.
    import jax
    with (physical/'initial_carry.pkl').open('rb') as f:initial=pickle.load(f)
    initial_checked=0;rows=[];raws={};actual=0
    for i,c in enumerate(index):
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('changed raw trace')
        np.testing.assert_allclose(c['original_goal_xy'],goals[c['context']],atol=1e-6,rtol=0)
        with Path(p['contexts'][c['context']]['initial']).open('rb') as f:bank=pickle.load(f).state
        origin=jax.tree.map(lambda x:x[0],bank)
        loaded=jax.tree.map(lambda x:x[i],initial.state)
        for field in origin.__dataclass_fields__:
            if field in ('alpha','projection','follower'):continue
            a,ta=jax.tree.flatten(getattr(origin,field));b,tb=jax.tree.flatten(getattr(loaded,field))
            if ta!=tb or len(a)!=len(b):raise ValueError('initial state structure changed '+field)
            if any(not np.array_equal(np.asarray(x),np.asarray(y),equal_nan=True) for x,y in zip(a,b)):
                raise ValueError('initial dynamic field changed '+field)
        if float(loaded.alpha)!=c['alpha']:raise ValueError('original user alpha changed')
        initial_checked+=1
        with np.load(c['raw_file']) as z:raw={k:z[k] for k in z.files}
        raw_idx=np.flatnonzero(raw['active_tick']);n=len(raw_idx);actual+=n
        np.testing.assert_array_equal(raw['physical_tick'][raw_idx],np.arange(n))
        audit_pulse(raw,c)
        verify_correction_chain(raw,json.loads((root/'common_environment_config.json').read_text()),
                                initial_filtered=np.asarray(loaded.filtered),initial_offsets=np.asarray(loaded.offsets))
        # Respect first contact/fault for arrival and useful-task diagnostics.
        fault=raw['nonfinite']|raw['lower_fault']|raw['policy_fault']
        fatal=np.flatnonzero(raw['active_tick']&(raw['contact_failure']|fault))
        stop=int(fatal[0]) if len(fatal) else n
        xy=raw['actual_xy'][:stop];speed=raw['actual_forward_speed'][:stop]
        deviation=trees[c['context']].query(xy)[0] if len(xy) else np.array([])
        origin_xy=np.asarray(paths[c['context']]['xy'][0])
        travelled=float(np.linalg.norm(np.diff(np.vstack([origin_xy,xy]),axis=0),axis=1).sum())
        report={}
        for horizon in (10,20):
            cut={k:v[:horizon*200] for k,v in raw.items()}
            goal=goal_arrival_metrics(cut,goal_xy=goals[c['context']],radius_m=1.,dt=.005,horizon_s=float(horizon))
            outcome=physical_outcome(cut['active_tick'],cut['contact_failure'],
                                     cut['nonfinite']|cut['lower_fault']|cut['policy_fault'],horizon*200)
            report[str(horizon)]=dict(**goal,physical_outcome=outcome)
        arrival=report['20']['first_arrival_s']
        task_stop=min(stop,int(round(arrival/.005))) if arrival is not None else stop
        task_xy=raw['actual_xy'][:task_stop]
        task_deviation=trees[c['context']].query(task_xy)[0] if len(task_xy) else np.array([])
        rows.append(dict(c,horizons=report,observed_s=n*.005,
            path_length_before_contact_or_fault_m=travelled,
            original_route_sample_distance_max_m=float(deviation.max()) if len(deviation) else None,
            original_route_sample_distance_time_integral_m_s=float(deviation.sum()*.005),
            speed_abs_error_before_contact_time_integral_m=float(abs(speed-c['requested_speed_m_s']).sum()*.005),
            task_window_ends_at='first_arrival' if arrival is not None else 'first_contact_or_fault_or_horizon',
            task_window_observed_s=task_stop*.005,
            task_window_path_length_m=float(np.linalg.norm(np.diff(np.vstack([origin_xy,task_xy]),axis=0),axis=1).sum()),
            task_window_original_route_sample_distance_max_m=float(task_deviation.max()) if len(task_deviation) else None,
            task_window_original_route_sample_distance_time_integral_m_s=float(task_deviation.sum()*.005),
            task_window_speed_abs_error_time_integral_m=float(abs(raw['actual_forward_speed'][:task_stop]-c['requested_speed_m_s']).sum()*.005),
            diagnostic_changed_path_scored_cost=float(-raw['scored_tick_reward'][raw['active_tick']].sum())))
        raws[c['case_id']]=raw
    if actual!=status['active_lower_ticks']:raise ValueError('physical charge differs from all48 raw traces')
    summary={};paired={}
    for horizon in ('10','20'):
        summary[horizon]={}
        for role in ROLES:
            group=[r for r in rows if r['role']==role]
            if len(group)!=12:raise ValueError('all12 balanced events per role required')
            hs=[r['horizons'][horizon] for r in group]
            summary[horizon][role]=dict(n=12,arrived=sum(h['goal_arrived'] is True for h in hs),
                survived=sum(h['physical_outcome']=='survived' for h in hs),
                contact=sum(h['physical_outcome']=='contact' for h in hs),
                unknown=sum(h['physical_outcome']=='unknown' or h['goal_arrived'] is None for h in hs),
                arrived_and_survived=sum(h['goal_arrived'] is True and h['physical_outcome']=='survived' for h in hs))
        for candidate,baseline in [('replan_full','original_full'),('replan_zero','original_zero'),('replan_full','replan_zero')]:
            key=lambda r:(r['context'],r['alpha'],r['event_id'])
            cc={key(r):r for r in rows if r['role']==candidate};bb={key(r):r for r in rows if r['role']==baseline}
            if set(cc)!=set(bb) or len(cc)!=12:raise ValueError('paired event mismatch')
            details=[]
            for k,c in cc.items():
                b=bb[k];ch=c['horizons'][horizon];bh=b['horizons'][horizon]
                known=ch['goal_arrived'] is not None and bh['goal_arrived'] is not None and 'unknown' not in (ch['physical_outcome'],bh['physical_outcome'])
                details.append(dict(key=k,known=known,candidate=c['case_id'],baseline=b['case_id'],
                    arrival_change=int(ch['goal_arrived'])-int(bh['goal_arrived']) if known else None,
                    survival_change=int(ch['physical_outcome']=='survived')-int(bh['physical_outcome']=='survived') if known else None))
            paired[horizon+'/'+candidate+'_vs_'+baseline]=dict(n=12,known=sum(d['known'] for d in details),
                arrival_gain=sum(d['arrival_change']==1 for d in details),arrival_loss=sum(d['arrival_change']==-1 for d in details),
                survival_gain=sum(d['survival_change']==1 for d in details),survival_loss=sum(d['survival_change']==-1 for d in details),details=details)
    atomic_json(out/'results.json',rows);atomic_json(out/'summary.json',summary);atomic_json(out/'paired.json',paired)
    render(out,rows,raws,paths,goals,physical)
    atomic_json(out/'receipt.json',dict(phase='completed',analysis_physics_steps=0,raw_traces_sha_checked=48,
        complete_initial_dynamic_fields_checked=initial_checked,actual_lower_ticks=actual,
        allocated_lower_slots=status['allocated_lower_slots'],pulse_and_reference_chain_checked=48,
        original_route_distance='Nearest original path sample up to goal; ~2cm spatial sampling, not exact segment distance or original V3 cost',
        cost_warning='Replanned-path score is diagnostic; 10s slice belongs to a 20s rollout, not the old 10s terminal penalty contract',
        figure_visual_review='pending'))
    lines=['# GitHub Dubins重规划固定48结果','','原目标1m首次触地前进入，无末端姿态/速度/保持门槛。完整与zero中层都保留相同几何反馈/冻结local300。',
           '原路与重规划均20秒；10秒是同轨迹前缀，不是新旧终止罚奖励比较。','','|窗口|方法|到达/12|不摔/12|触地/12|未知|','|---|---|---:|---:|---:|---:|']
    for h,ss in summary.items():
        for role,r in ss.items():lines.append(f'|{h}s|{role}|{r["arrived"]}|{r["survived"]}|{r["contact"]}|{r["unknown"]}|')
    lines+=['','首先看xy_force_+0.png及±32图：黑点线为原目标路径，彩虚线为新规划路径，实线为实际轨迹；圆点是10秒位置，方点是20秒终点，x是真实触地。',
            '首达后不停车，因此20秒末位置不等于是否曾到达；first_arrival_s逐例保存。全部48/原alpha/原goal与失败保留，不选择最好个例。',
            '这是相关DEV、两个路线、一个进入历史与冻结模型；正结果首先属于放松原路线约束后的规划基线，不证明新抗扰机制/规划器创新/独立泛化。',
            '10/20秒完整窗口存活另列；到达后再摔仍计曾到达但不计整窗存活。规划后的path条件reward只诊断，不能作为同原固定路线效用。',
            '原路偏离统计为原路线到目标段离散采样最近距离，位置采样约2cm；task_window字段在首次到达/触地/故障/时限中最早者截止。',
            '全观察窗before_contact字段亦保留，但正常到达后沿尾段继续行驶也会远离目标前路线段，不能用该全窗数值断言跟踪恶化。未完成轨迹短窗指标也不自动更优。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,paired={k:{a:b for a,b in v.items() if a!='details'} for k,v in paired.items()})))


def render(out,rows,raws,paths,goals,physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors=dict(original_full='#D55E00',replan_full='#0072B2',original_zero='#666666',replan_zero='#009E73')
    contexts=list(paths)
    for force in (0.,-32.,32.):
        fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
        for col,context in enumerate(contexts):
            path=paths[context];g=goals[context]
            for row,alpha in enumerate((0.,1.)):
                ax=axes[row,col];ax.plot(*path['xy'][path['s']<=path['goal']].T,c='black',ls=':',lw=1,label='Original route')
                ax.add_patch(Circle(g,1.,fill=False,ec='black',lw=.8));ax.scatter(*g,marker='*',c='black',s=35)
                group=[r for r in rows if r['context']==context and r['alpha']==alpha and r['force_N']==force]
                for r in group:
                    raw=raws[r['case_id']];a=raw['active_tick'].astype(bool);xy=raw['actual_xy'][a];c=colors[r['role']]
                    ax.plot(*xy.T,c=c,lw=1.2,label=r['role'])
                    if len(xy)>=2000:ax.scatter(*xy[1999],c=c,marker='o',s=18)
                    ax.scatter(*xy[-1],c=c,marker='x' if r['horizons']['20']['physical_outcome']=='contact' else 's',s=24)
                    arrival=r['horizons']['20']['first_arrival_s']
                    if arrival is not None:
                        at=int(round(arrival/.005))-1
                        ax.scatter(*xy[at],c=c,marker='^',s=32)
                    i=rows.index(r);file=physical/f'replanned_path_{i:02d}.npz'
                    if file.exists():
                        with np.load(file) as z:ax.plot(*z['xy'][z['s']<=z['goal']].T,c=c,ls='--',lw=.8,alpha=.65)
                ax.set_title(context.split('__')[1]+f' / original alpha={alpha:g}',fontsize=10)
                ax.set(xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'Original goal unchanged; force {force:+g} N / 0.1s; both methods 20s\nActual solid / plans dashed; circle=10s, square=20s, x=contact, triangle=first arrival')
        fig.savefig(out/f'xy_force_{force:+g}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # Full per-event dynamics, without selecting only successful replans.
    diag=out/'diagnostics';diag.mkdir()
    for context in contexts:
        for alpha in (0.,1.):
            for force in (0.,-32.,32.):
                group=[r for r in rows if r['context']==context and r['alpha']==alpha and r['force_N']==force]
                fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
                for r in group:
                    raw=raws[r['case_id']];a=raw['active_tick'].astype(bool);t=(raw['physical_tick'][a]+1)*.005;c=colors[r['role']]
                    for ax,key in zip(axes,('actual_forward_speed','actual_delta','phi','phi_dot')):
                        ax.plot(t,raw[key][a],c=c,lw=1,label=r['role'])
                    axes[0].plot(t,raw['wheel_speed_proxy'][a],c=c,ls='--',alpha=.4)
                    axes[0].plot(t,raw['reference'][a,0],c=c,ls=':',alpha=.5)
                for ax,label in zip(axes,('Speed m/s; faint dashed wheel / dotted ref','Actual steering rad','Roll rad','Roll rate rad/s')):
                    ax.set_ylabel(label);ax.grid(alpha=.15);ax.axvline(10,c='black',ls=':',lw=.6)
                axes[0].legend(fontsize=7);axes[-1].set_xlabel('Task time (s)')
                fig.suptitle(context+f' / alpha{alpha:g} / F{force:+g}N; no smoothing')
                fig.savefig(diag/f'{context}_a{alpha:g}_F{force:+g}.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    import traceback
    from engineer_reference_bias import notify
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    try:analyze(a.root)
    except BaseException:
        directory=a.root/'analysis'
        if directory.exists():(directory/'analysis_error.txt').write_text(traceback.format_exc())
        notify(a.root,'STTW重规划结果分析报错','原轨迹不变，分析错误已保留；不重跑仿真。')
        raise
    else:notify(a.root,'STTW重规划结果分析完成','48条轨迹的10/20秒到达、触地和全部XY图已生成，等待图形核对。')
