"""Read all360 matched traces; retain relaxed arrival and physical outcomes separately."""
import argparse
import json
from pathlib import Path
import numpy as np

ROLES = ['goal_zero_feedback', 'lineage_h75', 'goal_full_feedback']
LABELS = ['Zero neural', 'Frozen H75', 'Current free_GAE64']


def review(root, analysis_name="analysis"):
    root=Path(root).resolve();analysis=root/analysis_name
    if json.loads((analysis/'status.json').read_text())['phase']!='completed':
        raise ValueError('complete original audited analysis required; do not rerun physics')
    plan=json.loads((root/'plan.json').read_text())
    if plan['experiment']!='h75_disturbed_common_interface_v1':raise ValueError('declared H75 comparison required')
    rows=json.loads((analysis/'results.json').read_text())
    key=lambda r:(r['context'],r['alpha'],r['event_id'])
    byrole={role:{key(r):r for r in rows if r['role']==role} for role in ROLES}
    if len(rows)!=360 or any(len(group)!=120 for group in byrole.values()):raise ValueError('all360 unique cases required')
    if any(set(group)!=set(byrole[ROLES[0]]) for group in byrole.values()):raise ValueError('missing paired case')
    if any(r['physical_outcome']=='unknown' or r['goal_arrived'] is None for r in rows):raise ValueError('unknown outcomes must remain explicit')
    result={}
    for split in ['nominal','disturbed']:
        subset=[r for r in rows if (r['force_N']!=0)==(split=='disturbed')]
        report={}
        for radius in [.5,.75,1.]:
            counts={}
            for role in ROLES:
                rr=[r for r in subset if r['role']==role]
                counts[role]=dict(n=len(rr),arrived=sum(r['minimum_upright_goal_distance_m']<=radius for r in rr),
                                  survived=sum(r['physical_outcome']=='survived' for r in rr),
                                  arrived_and_survived=sum(r['minimum_upright_goal_distance_m']<=radius and r['physical_outcome']=='survived' for r in rr),
                                  median_min_goal_distance_m=float(np.median([r['minimum_upright_goal_distance_m'] for r in rr])),
                                  mean_scored_cost=float(np.mean([r['scored_cost'] for r in rr])),
                                  mean_reference_speed_variation=float(np.mean([r['reference_speed_variation_m_s'] for r in rr])),
                                  mean_reference_steer_variation=float(np.mean([r['reference_steer_variation_rad'] for r in rr])))
            pairs={}
            for candidate,baseline in [('goal_full_feedback','lineage_h75'),('lineage_h75','goal_zero_feedback'),('goal_full_feedback','goal_zero_feedback')]:
                details=[]
                for k,c in byrole[candidate].items():
                    if (c['force_N']!=0)!=(split=='disturbed'):continue
                    b=byrole[baseline][k]
                    details.append(dict(context=c['context'],alpha=c['alpha'],force_N=c['force_N'],event_id=c['event_id'],
                        arrival_candidate=c['minimum_upright_goal_distance_m']<=radius,arrival_baseline=b['minimum_upright_goal_distance_m']<=radius,
                        survived_candidate=c['physical_outcome']=='survived',survived_baseline=b['physical_outcome']=='survived',
                        candidate_distance_m=c['minimum_upright_goal_distance_m'],baseline_distance_m=b['minimum_upright_goal_distance_m'],
                        original_alpha_cost_change=c['scored_cost']-b['scored_cost']))
                pairs[candidate+'_vs_'+baseline]=dict(n=len(details),
                    arrival_gain=sum(d['arrival_candidate'] and not d['arrival_baseline'] for d in details),
                    arrival_loss=sum(d['arrival_baseline'] and not d['arrival_candidate'] for d in details),
                    survival_gain=sum(d['survived_candidate'] and not d['survived_baseline'] for d in details),
                    survival_loss=sum(d['survived_baseline'] and not d['survived_candidate'] for d in details),details=details)
            endpoint_counts={}
            for alpha in [0.,1.]:
                endpoint_counts[str(alpha)]={}
                for role in ROLES:
                    rr=[r for r in subset if r['role']==role and r['alpha']==alpha]
                    endpoint_counts[str(alpha)][role]=dict(n=len(rr),
                        arrived=sum(r['minimum_upright_goal_distance_m']<=radius for r in rr),
                        survived=sum(r['physical_outcome']=='survived' for r in rr),
                        arrived_and_survived=sum(r['minimum_upright_goal_distance_m']<=radius and r['physical_outcome']=='survived' for r in rr),
                        mean_same_user_scored_cost=float(np.mean([r['scored_cost'] for r in rr])),
                        median_min_goal_distance_m=float(np.median([r['minimum_upright_goal_distance_m'] for r in rr])))
            report[str(radius)]=dict(counts=counts,pairs=pairs,by_executed_and_original_user_alpha=endpoint_counts)
        result[split]=report
    out=analysis/'h75_review';out.mkdir(exist_ok=False)
    (out/'review.json').write_text(json.dumps(dict(results=result,new_physics_steps=0,optimizer_updates=0,
        scope='All360 related DEV traces, three frozen middle-weight families with the same goal/feedback/short-support stack. Radius .5 primary working endpoint; .75/1 posthoc sensitivity. No independent repeats, native reproduction, new safety method or automatic adoption.'),indent=2)+'\n')
    lines=['# H75是否覆盖当前恢复收益：匹配强对照','',
        '96新H75受扰＋264认证缓存，三族各24无扰/96受扰。同一原目标、完整进入状态、用户α和控制栈，只换神经权重。整窗存活与进入目标分开；不加末段保持，不将到达后触地藏掉。','',
        '|条件|目标半径/m|Zero到达|H75到达|Current到达|Zero/H75/Current整窗存活|','|---|---:|---:|---:|---:|---|']
    for split,rr in result.items():
        for radius,d in rr.items():
            c=d['counts'];n=c[ROLES[0]]['n'];values=[f"{c[r]['arrived']}/{n}" for r in ROLES]
            lines.append('|'+split+'|'+radius+'|'+'|'.join(values)+'|'+'/'.join(str(c[r]['survived']) for r in ROLES)+'|')
    lines+=['','## 两个用户偏好的分项（0.5米工作口径）','',
            '|条件|原用户α|方法|到达|整窗存活|到达且整窗存活|同用户成本均值|',
            '|---|---:|---|---:|---:|---:|---:|']
    for split,rr in result.items():
        for alpha,group in rr['0.5']['by_executed_and_original_user_alpha'].items():
            for role,c in group.items():
                lines.append(f"|{split}|{alpha}|{role}|{c['arrived']}/{c['n']}|{c['survived']}/{c['n']}|{c['arrived_and_survived']}/{c['n']}|{c['mean_same_user_scored_cost']:.4f}|")
    lines+=['','不同α的原始成本尺度不同，分项只在相同原用户α下比较；不以混合α成本均值排名偏好能力。','', '0.75/1米只是敏感性，不倒写预注册；配对新增/丢失、每例距离/成本及参考变化见review.json和原分析。未知不强填失败；所有物理失败都保留。共享祖先与同快照多力档不算独立样本，不作因果训练归因。',
            '', 'XY图每个力档一张，包含12进入情境×两端α×三族的全部实际轨迹；灰线为原世界路径，原0.5m圆。点标真实终点，x为触地。无平滑/坐标重定位/成功筛选。仅工作图，不声明期刊规范。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    render(out,rows,plan)
    print(json.dumps(dict(review=str(out),disturbed=result['disturbed']['0.5']['counts'])))


def render(out, rows, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    colors=['#666666','#D55E00','#0072B2'];audit=[]
    for force in [0.,-16.,16.,-32.,32.]:
        fig,axes=plt.subplots(4,3,figsize=(16,15),layout='constrained')
        for ax,(context,spec) in zip(axes.flat,plan['contexts'].items()):
            with np.load(spec['path']) as p:
                s=p['s'];xy=p['xy'];goal=float(p['goal']);point=[np.interp(goal,s,xy[:,i]) for i in [0,1]]
                ax.plot(xy[s<=goal,0],xy[s<=goal,1],color='black',ls=':',lw=1)
            ax.add_patch(Circle(point,.5,fill=False,ec='black',lw=.7));ax.scatter(*point,c='black',marker='*',s=22)
            rr=[r for r in rows if r['context']==context and r['force_N']==force]
            if len(rr)!=6:raise ValueError('all two-alpha/three-role traces required in each force/context panel')
            for role,color in zip(ROLES,colors):
                for alpha,ls in [(0.,'-'),(1.,'--')]:
                    r=next(r for r in rr if r['role']==role and r['alpha']==alpha)
                    with np.load(r['raw_file']) as raw:a=raw['active_tick'].astype(bool);actual=raw['actual_xy'][a]
                    ax.plot(actual[:,0],actual[:,1],color=color,ls=ls,lw=1)
                    ax.scatter(*actual[-1],color=color,s=15,marker='x' if r['physical_outcome']=='contact' else 'o' if alpha==0 else 's')
                    audit.append(dict(case_id=r['case_id'],raw_sha256=r['raw_sha256'],active_samples=len(actual)))
            prefix,geo=context.split('__');ax.set_title(prefix+' / '+geo,fontsize=8)
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel('World X (m)',fontsize=8);ax.set_ylabel('World Y (m)',fontsize=8);ax.grid(alpha=.2);ax.tick_params(labelsize=7)
        handles=[Line2D([],[],color=c,label=l) for c,l in zip(colors,LABELS)]
        handles += [Line2D([],[],color='black',ls='-',marker='o',label='alpha0'),Line2D([],[],color='black',ls='--',marker='s',label='alpha1'),Line2D([],[],color='black',ls=':',marker='*',label='Original path / target')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle(f'Frozen H75 / current / zero middle: lateral force {force:g} N, duration 0.1s\nAll12 entry contexts; original event times / fixed user alpha / common reference stack',fontsize=12)
        fig.savefig(out/f'all_xy_force_{force:+g}.png',dpi=140);plt.close(fig)
    if len(audit)!=360 or len({a['case_id'] for a in audit})!=360:raise ValueError('complete360 plotted cases required')
    (out/'visual_review.json').write_text(json.dumps(dict(actual_visual_review_complete=False,plotted_cases=audit,
        transform='All actual active world XY; no smoothing/reanchoring/outcome exclusions; equal meter axes in every panel.',publisher_requirements='Pending exact Transactions venue; provisional work figure.'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--analysis-name',default='analysis')
    args=parser.parse_args();review(args.run,args.analysis_name)
