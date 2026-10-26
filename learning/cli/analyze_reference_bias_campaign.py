"""All selected reference outcomes and fixed strong controls, no reselection."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.reference_bias_runtime import score_raw,encode_objective
from sttw_control.reference_bias_optimization import bias_targets


def analyze(root):
    root=root.resolve();prod=root/'production'
    status=json.loads((prod/'status.json').read_text())
    if status['phase']!='completed' or status['evaluation_cases']!=120:raise ValueError('complete production required')
    contract=json.loads((prod/'contract.json').read_text())
    if digest(prod/'contract.json')!=json.loads((prod/'seal.json').read_text())['contract_sha256']:raise ValueError('contract changed')
    if digest(prod/'cached_baselines.json')!=contract['cached_baselines_sha256']:raise ValueError('baseline catalog changed')
    frozen=json.loads((prod/'selection_frozen.json').read_text())
    if digest(prod/'selected_references.json')!=frozen['selected_references_sha256']:raise ValueError('nominal decisions changed')
    out=prod/'analysis';out.mkdir(exist_ok=False)
    plan=json.loads((root/'plan.json').read_text());cfg=json.loads((root/'common_environment_config.json').read_text());paths={}
    for name,c in plan['contexts'].items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('original path changed')
        with np.load(c['path']) as z:paths[name]={k:z[k] for k in z.files}
    new=json.loads((prod/'evaluation_results.json').read_text());old=json.loads((prod/'cached_baselines.json').read_text())
    if len(new)!=120 or len(old)!=240:raise ValueError('all120new+240cached required')
    rows=[];audits=[]
    for inherited,rr in [(False,new),(True,old)]:
        for r in rr:
            if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('raw identity changed')
            with np.load(r['raw_file']) as z:raw={k:z[k] for k in z.files}
            ctx=plan['contexts'][r['context']]
            extra={} if inherited else dict(targets=bias_targets(r['vector']),positions=ctx['bias_positions_m'],feedback_spec=plan['feedback_spec'])
            q=score_raw(raw,r,cfg,paths[r['context']],**extra)
            if abs(q['scored_cost']-r['scored_cost'])>1e-8:raise ValueError('original cost changed')
            a=raw['active_tick'].astype(bool)
            q.update(raw_file=r['raw_file'],raw_sha256=r['raw_sha256'],inherited=inherited,
                     actual_applied_impulse=np.asarray(raw['applied_force_impulse'])[a].sum(axis=0).tolist(),
                     privileged_onset_controller=r['role']=='neural_after_only')
            rows.append(encode_objective(q));audits.append(dict(case_id=r['case_id'],raw_sha256=r['raw_sha256'],original_reward_and_world_goal_recomputed=True,
                bias_audit=not inherited,scope='Original cached control audit inherited, no new bias fields invented.' if inherited else 'New bias/feedback/gate audit repeated.'))
    roles=['goal_full_feedback','goal_zero_feedback','goal_steer_feedback','neural_after_only','cem','random']
    groups={}
    for r in rows:
        g=groups.setdefault((r['context'],r['alpha'],r['event_id']),{})
        if r['role'] in g:raise ValueError('duplicate role/event')
        g[r['role']]=r
    if len(groups)!=60 or any(set(g)!=set(roles) for g in groups.values()):raise ValueError('all six controls on every original event required')
    report={}
    for split in ['nominal','disturbed']:
        selected=[g for g in groups.values() if bool(g['cem']['force_N'])==(split=='disturbed')]
        counts={role:dict(n=len(selected),arrived={str(t):sum(g[role]['goals'][str(t)]['goal_arrived'] for g in selected) for t in [.5,.75,1.]},
            survived=sum(g[role]['physical_outcome']=='survived' for g in selected),
            arrived_1m_and_survived=sum(g[role]['goals']['1.0']['goal_arrived'] and g[role]['physical_outcome']=='survived' for g in selected),
            own_user_cost_by_alpha={str(a):float(np.mean([g[role]['scored_cost'] for g in selected if g[role]['alpha']==a])) for a in [0.,.5,1.]}) for role in roles}
        pairs={}
        for role in ['cem','random']:
            for baseline in ['goal_full_feedback','goal_zero_feedback','goal_steer_feedback','neural_after_only','random']:
                if role==baseline:continue
                pairs[role+'_vs_'+baseline]=dict(n=len(selected),
                    arrival_gain=sum(g[role]['goals']['1.0']['goal_arrived'] and not g[baseline]['goals']['1.0']['goal_arrived'] for g in selected),
                    arrival_loss=sum(g[baseline]['goals']['1.0']['goal_arrived'] and not g[role]['goals']['1.0']['goal_arrived'] for g in selected),
                    survival_gain=sum(g[role]['physical_outcome']=='survived' and g[baseline]['physical_outcome']!='survived' for g in selected),
                    survival_loss=sum(g[baseline]['physical_outcome']=='survived' and g[role]['physical_outcome']!='survived' for g in selected))
        report[split]=dict(counts=counts,paired=pairs)
    matrices={}
    for role in roles:
        rr=[r for r in rows if r['role']==role and r['force_N']==0]
        matrices[role]=[[float(np.mean([r['cross_user_costs'][str(user)] for r in rr if r['alpha']==executed])) for executed in [0.,.5,1.]] for user in [0.,.5,1.]]
    batch_status=[json.loads(f.read_text()) for f in (prod/'searches').glob('search_*/generation_*/status.json')]+[json.loads(f.read_text()) for f in (prod/'evaluation').glob('batch_*/status.json')]
    if len(batch_status)!=104 or sum(s['completed_cases'] for s in batch_status)!=1656:raise ValueError('complete96 search+8event batches required')
    for field in ['active_lower_ticks','allocated_lower_slots']:
        if sum(s[field] for s in batch_status)!=status[field]:raise ValueError('cost accounting mismatch')
    atomic_json(out/'results.json',rows);atomic_json(out/'raw_audit.json',audits)
    atomic_json(out/'summary.json',dict(results=report,cross_user_matrix_rows_user_cols_executed=matrices,
        production_cost=status,inherited_engineering_lower_ticks=32000,total_project_new_lower_ticks=status['active_lower_ticks']+32000,
        warning='Nominal fit and same seen-DEV hidden-event evaluation, not independent TEST, training repeats or online planner. Matrix reweights fixed executed trajectories, not counterfactual dynamics. After-only uses privileged onset.',visual_review_complete=False))
    render(out,rows,paths,roles)
    lines=['# 定向参考优化与同预算随机对照','',
        '冻结完整中层和原用户alpha；只用名义原目标选偏置，然后锁定24个过程面对全部原事件。0.5/0.75/1m原world目标进入与全窗存活分开；全部损失保留。研究者已见DEV，不能称独立测试或部署优化。','',
        '|条件|方法|例数|1m到达|原10s全存活|','|---|---|---:|---:|---:|']
    for split,s in report.items():
        for role,c in s['counts'].items():lines.append(f"|{split}|{role}|{c['n']}|{c['arrived']['1.0']}|{c['survived']}|")
    lines+=['','neural_after_only是特权机制对照，不是可部署控制器。每alpha原目标成本、交叉矩阵、各baseline配对新增/丢失、计算代价见summary.json；不因名义fit下降就声称抗扰增量。','',
            '全部360条真实XY按原力档分五张图，原固定路径/1m目标圈/真实终点/等米比例。图已生成不等于目视，实际图审另记录。']
    for force in [0.,-16.,16.,-32.,32.]:lines.append(f'\n[全部XY force {force:g}N](all_xy_force_{force:+g}.png)')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'status.json',dict(phase='completed',new_physics_steps=0,numeric_cases=360,optimizer_updates=0,visual_review_complete=False))
    print(json.dumps(report,ensure_ascii=False))


def render(out,rows,paths,roles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    colors=['#222222','#888888','#D55E00','#AA4499','#0072B2','#009E73']
    for force in [0.,-16.,16.,-32.,32.]:
        f,axes=plt.subplots(4,3,figsize=(16,15),layout='constrained')
        for i,(context,path) in enumerate(paths.items()):
            point=[np.interp(float(path['goal']),path['s'],path['xy'][:,k]) for k in range(2)]
            for j,alpha in enumerate([0.,.5,1.]):
                ax=axes[i,j];p=path['s']<=float(path['goal']);ax.plot(path['xy'][p,0],path['xy'][p,1],color='gray',ls=':',lw=.8)
                ax.add_patch(Circle(point,1.,fill=False,ec='gray',lw=.6));ax.scatter(*point,color='black',marker='*',s=20)
                rr=[r for r in rows if r['context']==context and r['alpha']==alpha and r['force_N']==force]
                if len(rr)!=6:raise ValueError('all six methods per panel required')
                for role,color in zip(roles,colors):
                    r=next(r for r in rr if r['role']==role)
                    with np.load(r['raw_file']) as z:xy=z['actual_xy'][z['active_tick'].astype(bool)]
                    ax.plot(xy[:,0],xy[:,1],color=color,ls='--' if role in ['random','neural_after_only'] else '-',lw=1)
                    ax.scatter(*xy[-1],color=color,marker='x' if r['physical_outcome']=='contact' else 'o',s=15)
                ax.set_title(context+f' / alpha{alpha}',fontsize=7);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.2)
        handles=[Line2D([],[],color=c,ls='--' if r in ['random','neural_after_only'] else '-',label=r+(' (privileged)' if r=='neural_after_only' else '')) for r,c in zip(roles,colors)]
        f.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=8)
        f.suptitle(f'Nominal-selected bounded reference bias / all360 seen-DEV endpoints / force {force:g}N, 0.1s\nAll original alpha; 1m original-goal circle; real contact/true horizon endpoints',fontsize=12)
        f.savefig(out/f'all_xy_force_{force:+g}.png',dpi=140);plt.close(f)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
