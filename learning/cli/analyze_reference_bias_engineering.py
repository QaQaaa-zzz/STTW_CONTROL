"""Audit/plot all16 fixed wiring trajectories; no new simulation or selection."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.reference_bias_runtime import score_raw
from sttw_control.reference_bias_optimization import bias_targets


def analyze(root, *, recovered=False):
    root=root.resolve();eng=root/'engineering';physical=eng/'physical'
    numeric=eng/'recovered_numeric' if recovered else physical
    if json.loads((numeric/'status.json').read_text())['phase']!='completed':raise ValueError('complete numeric audit required')
    if recovered and not (numeric/'recovery_receipt.json').exists():raise ValueError('explicit recovered receipt required')
    out=eng/('analysis_recovered_numeric' if recovered else 'analysis');out.mkdir(exist_ok=False)
    plan=json.loads((root/'plan.json').read_text());cfg=json.loads((root/'common_environment_config.json').read_text())
    design=json.loads((eng/'design.json').read_text());rows=json.loads((numeric/'results.json').read_text())
    if len(rows)!=16 or len({r['case_id'] for r in rows})!=16 or {r['case_id'] for r in rows}!={r['case_id'] for r in design['cases']}:raise ValueError('complete fixed16 required')
    paths={}
    for name,c in plan['contexts'].items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('path changed')
        with np.load(c['path']) as z:paths[name]={k:z[k] for k in z.files}
    traces={};audits=[]
    for r in rows:
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('raw identity changed')
        with np.load(r['raw_file']) as z:raw={k:z[k] for k in z.files}
        q=score_raw(raw,r,cfg,paths[r['context']],targets=bias_targets(r['vector']),positions=plan['contexts'][r['context']]['bias_positions_m'],feedback_spec=plan['feedback_spec'])
        if abs(q['scored_cost']-r['scored_cost'])>1e-8 or q['physical_outcome']!=r['physical_outcome']:raise ValueError('numeric result mismatch')
        legal=raw['active_tick']&raw['actor_action_valid'];native=raw['original_actor_latent'][legal];coord=raw['coordinated_actor_latent'][legal]
        def mapped(z):
            t=np.tanh(z);c=cfg['residual_interface']
            return t*np.column_stack([np.where(t[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),np.full(len(t),c['steer_scale_rad'])])
        delta=mapped(coord)-mapped(native)
        audits.append(dict(case_id=r['case_id'],original_reward_goal_bias_gate_audits=True,
            bare_vs_coordinated_exactly_equal=bool(np.array_equal(native,coord)),legal_ticks=int(legal.sum()),
            max_actual_bias_target_change=[float(np.max(abs(delta[:,i]))) if len(delta) else 0. for i in range(2)]))
        traces[r['case_id']]=raw
    cost=json.loads((numeric/'status.json').read_text())
    if sum(r['physical_ticks'] for r in rows)!=cost['active_lower_ticks']:raise ValueError('cost mismatch')
    groups={}
    for r in rows:groups.setdefault((r['context'],r['alpha']),{})[r['role']]=r
    if len(groups)!=8 or any(set(v)!={'zero_bias','probe_bias'} for v in groups.values()):raise ValueError('all eight paired groups required')
    pairs=[]
    for k,g in groups.items():
        a,b=g['probe_bias'],g['zero_bias']
        pairs.append(dict(context=k[0],alpha=k[1],zero_cost=b['scored_cost'],probe_cost=a['scored_cost'],
            cost_change=a['scored_cost']-b['scored_cost'],
            arrival_change=int(a['goals']['1.0']['goal_arrived'])-int(b['goals']['1.0']['goal_arrived']),
            survival_change=int(a['physical_outcome']=='survived')-int(b['physical_outcome']=='survived')))
    # Old baseline numeric variation is descriptive; no exact replay gate.
    historical=json.loads(Path(plan['source'],'analysis/results.json').read_text());old={ (r['context'],r['alpha']):r for r in historical if r['role']=='goal_full_feedback' and r['force_N']==0}
    readback=[dict(context=k[0],alpha=k[1],new_zero_cost=g['zero_bias']['scored_cost'],old_cost=old[k]['scored_cost'],cost_change=g['zero_bias']['scored_cost']-old[k]['scored_cost'],scope='descriptive only; not exact old replay acceptance') for k,g in groups.items()]
    summary={role:dict(n=8,arrived_1m=sum(r['goals']['1.0']['goal_arrived'] for r in rows if r['role']==role),survived=sum(r['physical_outcome']=='survived' for r in rows if r['role']==role)) for role in ['zero_bias','probe_bias']}
    atomic_json(out/'results.json',dict(summary=summary,pairs=pairs,raw_audits=audits,zero_historical_readback=readback,
        costs=cost,scope='Fixed nominal wiring probe, not an optimizer, robustness experiment or method gain. All16 retained; historical numeric variations not an exact replay gate.',new_physics_steps=0,optimizer_updates=0,visual_review_complete=False))
    render(out,rows,groups,traces,paths,cfg)
    render_tracking(out,rows,paths)
    (out/'README.md').write_text('# 完整中层参考偏置：16条真实接线检查\n\n先看[全部实际XY与原参考](all_xy.png)。覆盖4context、alpha0/1、zero/probe全部16条；原world目标1m圈、0.5m圈及真实终点。\n\n固定无扰、完整原10秒、冻结local300和原完整中层/alpha，不清history/ESO/filter。每组非零probe预先指定，未运行优化。全部原始轨迹重新核SHA、原goal、偏置/固定反馈/权限/gate及原reward/三个user目标成本。\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n全部原奖励/分项每20ms图见reward_pair_00..07.png；未以累计reward代替到达或存活。实际与allocated成本、零偏置对历史差异及每组得失见results.json。探针好坏只是接线结果，不做新方法结论，未启动24搜索或120受扰评价。图审状态另记录，不能把生成图片当目视审阅。\n')
    print(json.dumps(summary,ensure_ascii=False))


def render(out,rows,groups,traces,paths,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from sttw_control.path_command_reward import configured_cost_keys
    fig,axes=plt.subplots(4,2,figsize=(13,15),layout='constrained')
    colors={'zero_bias':'#333333','probe_bias':'#0072B2'}
    keys=configured_cost_keys(cfg)
    for i,((context,alpha),g) in enumerate(groups.items()):
        ax=axes.flat[i];path=paths[context];s=path['s'];goal=float(path['goal']);point=[np.interp(goal,s,path['xy'][:,j]) for j in range(2)]
        ax.plot(path['xy'][s<=goal,0],path['xy'][s<=goal,1],ls=':',color='gray',lw=1)
        for radius,style in [(1.,'-'),(.5,'--')]:ax.add_patch(Circle(point,radius,fill=False,ec='gray',ls=style,lw=.7))
        ax.scatter(*point,marker='*',color='black',s=28)
        for role,r in g.items():
            raw=traces[r['case_id']];a=raw['active_tick'].astype(bool);xy=raw['actual_xy'][a]
            ax.plot(xy[:,0],xy[:,1],color=colors[role],ls='-' if role=='zero_bias' else '--',lw=1.2)
            ax.scatter(*xy[-1],marker='x' if r['physical_outcome']=='contact' else 'o',color=colors[role],s=25)
        ax.set_title(context+'\nalpha'+str(alpha),fontsize=8);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.2)
        f,aa=plt.subplots(4,3,figsize=(14,10),layout='constrained')
        for role,r in g.items():
            raw=traces[r['case_id']];active=raw['active_tick'].astype(bool);tick=raw['physical_tick'][active];block=tick//4;n=int(block.max())+1
            acc=lambda v:np.bincount(block,weights=np.asarray(v)[active],minlength=n)
            total=acc(raw['scored_tick_reward']);running=acc(raw['tick_reward'])
            aa.flat[0].plot(np.arange(n),total,color=colors[role],label=role)
            for j,key in enumerate(keys):
                component=acc(-cfg['path_reward']['scale']*.005*raw['effective_costs'][:,j]);aa.flat[j+1].plot(np.arange(n),component,color=colors[role]);aa.flat[j+1].set_title(key)
            aa.flat[-1].plot(np.arange(n),total-running,color=colors[role]);aa.flat[-1].set_title('Terminal reward replacement')
        aa.flat[0].set_title('Actual scored reward /20ms');aa.flat[0].legend(fontsize=7)
        for a in aa.flat:a.set_xlabel('Policy step (20ms)');a.set_ylabel('Signed reward');a.grid(alpha=.2)
        f.suptitle(context+f' / original alpha{alpha} / zero external force / all actual active ticks',fontsize=10);f.savefig(out/f'reward_pair_{i:02d}.png',dpi=120);plt.close(f)
    fig.legend(handles=[Line2D([],[],color=c,ls='-' if r=='zero_bias' else '--',label=r) for r,c in colors.items()],loc='outside lower center',ncol=2)
    fig.suptitle('Frozen full middle + bounded progress bias: fixed nominal wiring check\nAll16 actual trajectories; 1m solid /0.5m dashed original-goal circles; true endpoints',fontsize=12)
    fig.savefig(out/'all_xy.png',dpi=140);plt.close(fig)




def render_tracking(out,rows,paths):
    """Ancillary requested/reference/body/wheel proxy, steering, roll and goal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)/'tracking';out.mkdir(exist_ok=False)
    groups={}
    for r in rows:groups.setdefault((r['context'],r['alpha']),{})[r['role']]=r
    for i,((context,alpha),g) in enumerate(groups.items()):
        f,axes=plt.subplots(4,1,figsize=(12,10),layout='constrained');path=paths[context]
        point=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,j]) for j in range(2)])
        for role,r in g.items():
            with np.load(r['raw_file']) as z:d={k:z[k] for k in z.files}
            a=d['active_tick'].astype(bool);t=(d['physical_tick'][a]+1)*.005;color='#333333' if role=='zero_bias' else '#0072B2'
            axes[0].plot(t,d['actual_forward_speed'][a],color=color,label=role+' body')
            axes[0].plot(t,d['reference'][a,0],color=color,ls='--',label=role+' reference')
            axes[0].plot(t,d['wheel_speed_proxy'][a],color=color,ls=':',alpha=.7,label=role+' rear-wheel proxy')
            if role=='zero_bias':axes[0].plot(t,d['requested_speed'][a],color='gray',ls='-.',label='Original requested speed')
            axes[1].plot(t,d['actual_delta'][a],color=color,label=role+' measured')
            axes[1].plot(t,d['reference'][a,1],color=color,ls='--',label=role+' reference')
            axes[2].plot(t,d['phi'][a],color=color,label=role)
            axes[3].plot(t,np.linalg.norm(d['actual_xy'][a]-point,axis=1),color=color,label=role)
        axes[3].axhline(1.,color='gray',ls='--',label='1m goal radius')
        for ax,y in zip(axes,['Speed (m/s)','Steer (rad)','Roll (rad)','Original-goal distance (m)']):
            ax.set_ylabel(y);ax.set_xlabel('Original task time (s)');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=7)
        f.suptitle(context+f' / alpha{alpha} / nominal wiring probe only',fontsize=10)
        f.savefig(out/f'tracking_pair_{i:02d}.png',dpi=120);plt.close(f)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--recovered',action='store_true');a=p.parse_args();analyze(a.root,recovered=a.recovered)
