"""Finite paired endpoint preference readout after the frozen lineage panel.

Cross-score executed alpha0/1 trajectories under each unchanged user objective;
never select a policy on the same outcomes or omit contacts from cost means.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def review(root, *, analysis_name="analysis"):
    root=Path(root).resolve()
    analysis=root/analysis_name
    status=json.loads((analysis/'status.json').read_text())
    if status['phase']!='completed':raise ValueError('completed original analysis required')
    plan=json.loads((root/'plan.json').read_text())
    if not plan.get('lineage_comparison') or plan['analysis_cases']!=120:raise ValueError('fixed120 lineage panel required')
    rows=json.loads((analysis/'results.json').read_text())
    with (analysis/'fixed_trace_objective_queries.csv').open() as f:queries=list(csv.DictReader(f))
    if len(rows)!=120 or len({r['case_id'] for r in rows})!=120 or len(queries)!=600:
        raise ValueError('complete120 trajectories and600 objective queries required')
    roles=plan['evaluation_roles']; contexts=set(plan['contexts'])
    if len(roles)!=5 or len(contexts)!=12:raise ValueError('five families and12 matched contexts required')
    bykey={(r['role'],r['context'],r['alpha']):r for r in rows}
    if len(bykey)!=120 or any(r['force_N']!=0 for r in rows):raise ValueError('unique nominal matched endpoints required')
    values={(q['case_id'],float(q['objective_alpha'])):float(q['scored_cost']) for q in queries}
    if len(values)!=600:raise ValueError('duplicate cross-objective query')
    reports={}
    for role in roles:
        matrix=[];regret=[];counts=[];ties=[];details=[]
        for objective in [0.,1.]:
            pairs=[]
            for context in sorted(contexts):
                pair=[bykey[(role,context,executed)] for executed in [0.,1.]]
                costs=np.array([values[(r['case_id'],objective)] for r in pair])
                if not np.isfinite(costs).all():raise ValueError('unknown/nonfinite cost cannot silently disappear')
                own=costs[int(objective)];other=costs[1-int(objective)]
                pairs.append(costs)
                details.append(dict(context=context,user_alpha=objective,executed0_cost=float(costs[0]),executed1_cost=float(costs[1]),
                    own_endpoint_cost_gap_to_measured_pair_min=float(own-min(costs))))
            a=np.asarray(pairs);matrix.append(a.mean(0).tolist())
            gap=a[:,int(objective)]-a.min(1)
            regret.append(float(gap.mean()))
            # Equality tolerance is arithmetic reporting, not an adoption gate.
            own=a[:,int(objective)];other=a[:,1-int(objective)]
            counts.append(int(np.sum(own < other-1e-6)))
            ties.append(int(np.sum(abs(own-other)<=1e-6)))
        endpoint=[]
        for alpha in [0.,1.]:
            rr=[bykey[(role,c,alpha)] for c in sorted(contexts)]
            if any(r['goal_arrived'] is None or r['physical_outcome']=='unknown' for r in rr):raise ValueError('unknown must be resolved as unknown, not success/failure')
            endpoint.append(dict(executed_alpha=alpha,n=12,arrived=sum(r['goal_arrived'] is True for r in rr),
                survived=sum(r['physical_outcome']=='survived' for r in rr),
                median_min_goal_distance_m=float(np.median([r['minimum_upright_goal_distance_m'] for r in rr])),
                mean_path_rmse_m=float(np.mean([r['task_path_rmse_m'] for r in rr])),
                mean_speed_rmse_m_s=float(np.mean([r['task_speed_rmse_m_s'] for r in rr]))))
        reports[role]=dict(cost_matrix_rows_user_alpha_cols_executed_alpha=matrix,
            mean_own_endpoint_cost_gap_to_measured_pair_min=regret,
            own_endpoint_strictly_cheaper_contexts=counts,arithmetic_ties=ties,
            endpoint_outcomes=endpoint,all_context_details=details)
    comparisons={}
    for role in roles:
        gain=loss=0
        for context in contexts:
            for alpha in [0.,1.]:
                a=bykey[(role,context,alpha)]['goal_arrived']; b=bykey[('goal_full_feedback',context,alpha)]['goal_arrived']
                gain+=int(a and not b); loss+=int(b and not a)
        comparisons[role]=dict(arrival_gains_vs_current=gain,arrival_losses_vs_current=loss)
    out=analysis/'lineage_nominal_review';out.mkdir(exist_ok=False)
    report=dict(families=reports,paired_vs_current=comparisons,trajectories=120,objective_queries=600,new_physics_steps=0,optimizer_updates=0,
        scope='All matched nominal DEV traces, including failures. Finite two-executed-endpoint comparisons, not global regret, an online selector, architecture-only ablation, native historical reproduction or independent TEST.')
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 冻结迭代链名义偏好核对','',report['scope'],'',
        '12个相同完整进入情境，各执行alpha0/1。用户alpha原样用于重评分。表中成本包含全部失败；RMSE是实际观察窗口统计，若触地导致窗口不同则不作固定窗因果性能排序。0.5m目标到达与整窗存活分开。','',
        '|模型|执行α|到达/12|整窗存活/12|路径RMSE均值/m|速度RMSE均值/(m/s)|',
        '|---|---:|---:|---:|---:|---:|']
    for role,d in reports.items():
        for r in d['endpoint_outcomes']:
            lines.append(f"|{role}|{r['executed_alpha']:g}|{r['arrived']}|{r['survived']}|{r['mean_path_rmse_m']:.4f}|{r['mean_speed_rmse_m_s']:.4f}|")
    lines+=['','## 同一用户目标下的成本','',
        '每个2×2矩阵的行为执行来自真实仿真，固定轨迹只重新计算用户目标；不能改用户alpha替策略打分。行=user0/1，列=执行0/1。所谓差距是逐情境与两个已测端点最小值之差的均值，不是全局最优性指标或要求必须为零。','']
    for role,d in reports.items():
        lines += [f"- {role}: 矩阵 {np.round(d['cost_matrix_rows_user_alpha_cols_executed_alpha'],4).tolist()}；自身端点相对已测最小值的均值差 {np.round(d['mean_own_endpoint_cost_gap_to_measured_pair_min'],4).tolist()}。"]
    lines+=['','## 比较边界','',
        '所有权重使用近期共同扩展参考接口，低速请求低于旧原生速度下限，因此旧模型结果是迁移诊断，不证明原生能力失效，也不把整个迭代链当架构单因素消融。72新、48认证缓存分别计费。共享bank祖先/相关进入前缀不算独立训练重复。新旧到达用途和失败均保留；不按此表自动重映射alpha、换采用模型或开启续训。', '',
        '全部12情境实际/原参考XY图在上一级all_xy_*.png；连续指标/固定原alpha成本/真实终点留在原分析，不用只报成功率代替任务代价。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis-name',default='analysis');a=p.parse_args()
    result=review(a.run,analysis_name=a.analysis_name);print(json.dumps(dict(trajectories=result['trajectories'],families=len(result['families']))))
