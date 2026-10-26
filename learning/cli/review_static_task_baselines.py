"""Paired offline policy-bank diagnostics; no future-aware deployment claims."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sttw_control.preference_best import digest

ROLES=('goal_zero_feedback','lineage_h75','goal_full_feedback')
KEYS=('context','alpha','event_id')


def evaluate(root):
    root=Path(root).resolve();source=root/'analysis_reconstructed_v2'
    if json.loads((source/'status.json').read_text())['phase']!='completed':
        raise ValueError('completed original numeric audit required')
    rows=json.loads((source/'results.json').read_text())
    key=lambda r:tuple(r[k] for k in KEYS)
    banks={role:{key(r):r for r in rows if r['role']==role} for role in ROLES}
    if len(rows)!=360 or any(len(g)!=120 for g in banks.values()):raise ValueError('complete360 unique cases required')
    if any(set(g)!=set(banks[ROLES[0]]) for g in banks.values()):raise ValueError('unmatched policy-bank events')
    for k in banks[ROLES[0]]:
        values=[g[k] for g in banks.values()]
        for field in ['force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s','feedback_gain_rad','causal_support']:
            if any(v[field]!=values[0][field] for v in values[1:]):raise ValueError('event/control permissions differ: '+field)
        for role,v in zip(ROLES,values):
            if v['source_keep_channels'] != ([False,False] if role=='goal_zero_feedback' else [True,True]):
                raise ValueError('neural mask differs from declared baseline mode')
        if any(v['physical_outcome']=='unknown' or v['goal_arrived'] is None for v in values):raise ValueError('unknown endpoint cannot be coerced')
    report={}
    details=[]
    for split in ['nominal','disturbed']:
        keys=[k for k,r in banks[ROLES[0]].items() if bool(r['force_N'])==(split=='disturbed')]
        candidates={role:[banks[role][k] for k in keys] for role in ROLES}
        # Rule selected after these outcomes were inspected: DEV diagnostic only.
        # Only fixed geometric task identity is used; all forces/onsets share it.
        composite=[]
        for k in keys:
            c=banks['goal_full_feedback'][k]
            role='goal_zero_feedback' if c['geometry_name']=='left55_R1p8_v1p1' else 'goal_full_feedback'
            if c['geometry_name'] not in ['left55_R1p8_v1p1','right120_R1p6_v1p4']:
                raise ValueError('outside declared fixed task lookup')
            chosen=banks[role][k];composite.append(chosen)
            details.append(dict(split=split,context=c['context'],alpha=c['alpha'],event_id=c['event_id'],force_N=c['force_N'],
                selected_role=role,selected_case_id=chosen['case_id'],selection_information='fixed geometry_name only; posthoc rule; no realized event input',
                arrived=chosen['goal_arrived'],survived=chosen['physical_outcome']=='survived'))
        candidates['static_left_zero_right_current']=composite
        counts={}
        for name,rr in candidates.items():
            counts[name]=dict(n=len(rr),arrived=sum(r['goal_arrived'] for r in rr),survived=sum(r['physical_outcome']=='survived' for r in rr),
                arrived_and_survived=sum(r['goal_arrived'] and r['physical_outcome']=='survived' for r in rr),
                same_user_cost_by_alpha={str(a):float(np.mean([r['scored_cost'] for r in rr if r['alpha']==a])) for a in [0.,1.]})
        unions={}
        for selected in [ROLES[1:],ROLES[::2],ROLES]:
            unions[' + '.join(selected)]=dict(n=len(keys),
                hindsight_whole_episode_arrival_union=sum(any(banks[role][k]['goal_arrived'] for role in selected) for k in keys),
                hindsight_whole_episode_survival_union=sum(any(banks[role][k]['physical_outcome']=='survived' for role in selected) for k in keys))
        partitions={}
        for name,fields in [('global',()),('original_alpha',('alpha',)),('geometry_and_original_alpha',('geometry_name','alpha')),('complete_entry_and_original_alpha',('context','alpha'))]:
            groups=defaultdict(lambda:defaultdict(list))
            for k in keys:
                for role in ROLES:
                    r=banks[role][k];groups[tuple(r[f] for f in fields)][role].append(r)
            groups_out=[]
            for group,values in groups.items():
                counts_by_role={role:dict(arrived=sum(r['goal_arrived'] for r in rr),survived=sum(r['physical_outcome']=='survived' for r in rr)) for role,rr in values.items()}
                groups_out.append(dict(group=list(group),counts=counts_by_role))
            partitions[name]=dict(groups=groups_out,
                separate_objective_hindsight_arrival_max=sum(max(g['counts'][role]['arrived'] for role in ROLES) for g in groups_out),
                separate_objective_hindsight_survival_max=sum(max(g['counts'][role]['survived'] for role in ROLES) for g in groups_out),
                warning='Independent per-objective posthoc maxima; generally different selectors, not a simultaneous achieved pair. Entry identity is not assumed deployed task information.')
        # Compare actual chosen trajectories, rather than combining independent maxima.
        paired={}
        for name in candidates:
            if name=='goal_full_feedback':continue
            pairs=list(zip(candidates[name],candidates['goal_full_feedback']))
            paired[name+'_vs_current']=dict(arrival_gain=sum(a['goal_arrived'] and not b['goal_arrived'] for a,b in pairs),
                arrival_loss=sum(b['goal_arrived'] and not a['goal_arrived'] for a,b in pairs),
                survival_gain=sum(a['physical_outcome']=='survived' and b['physical_outcome']!='survived' for a,b in pairs),
                survival_loss=sum(b['physical_outcome']=='survived' and a['physical_outcome']!='survived' for a,b in pairs))
        report[split]=dict(counts=counts,whole_episode_bank_diagnostics=unions,partition_diagnostics=partitions,paired=paired)
    out=root/'static_task_baseline_review';out.mkdir(exist_ok=False)
    scope='Retrospective DEV trajectory reuse, no new physics/optimization or fitted online controller. Whole-episode bank union is NOT an upper bound on within-trajectory switching, hybrid reference control or physical viability. Marginal arrival/survival unions may require different selectors. All outcomes preserved; fixed-geometry lookup uses no realized force or onset, but was selected after these data and is not independent confirmation.'
    output=dict(source_results=str(source/'results.json'),source_results_sha256=digest(source/'results.json'),source_raw_audit=str(source/'raw_audit.json'),
                protocol=str(root/'plan.json'),results=report,selected_cases=details,scope=scope,new_physics_steps=0,optimizer_updates=0)
    (out/'review.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# 固定任务选择能覆盖多少收益？','',scope,'',
           '|条件|模型/规则|到达|整窗存活|','|---|---|---:|---:|']
    for split,d in report.items():
        for role,c in d['counts'].items():lines.append(f"|{split}|{role}|{c['arrived']}/{c['n']}|{c['survived']}/{c['n']}|")
    lines+=['','## 可以得出的判断','',
        'H75/current整回合事后到达并集：受扰72/96，与current相同；不能以整回合模型选择声称新增到达。但这不是时序混合、参考协调或其它控制器的性能上界，不能据此判定它们无效。',
        'zero/current整回合到达并集76/96、存活并集92/96均为未来结局可见的诊断量，不能当可部署成绩，也不能默认同一个选择能同时取得两项并集。',
        '事后选定的固定几何规则“左弯zero、右弯current”受扰到达72/96、存活90/96；无扰到达22/24、存活24/24。相对current受扰到达4增4失，存活7增2失；名义到达0增2失。总数相同并不表示逐例能力相同。该规则以固定原任务几何从task0选权重，所有force/onset一视同仁，是可执行简单基线的旧轨迹回顾性读回，没有新增切换仿真或独立测试。',
        '', '## 对下一方法的约束','',
        '应研究在同一任务条件下何时有用地保留、减弱或调整神经参考修正，同时保留原用户alpha的效用；不能只证明检测到外力或在不同速度面板换权重。H75保留原生偏好的价值不因低速并集结果消失。简单几何规则必须作为强对照，零中层仍含同一goal-preview/短支持/反馈与冻结local300，不是裸底层。',
        '已有数据只证明机会存在于有限已测整回合轨迹并集，并未证明在线干预可获得该机会。先判断四个zero-only事件及对应current-only事件的可观测差异和参考动作作用，再决定是否值得一个有界闭环机会验证；不拟合后验阈值、扩参数库或加训来维持叙事。',
        '', '## 原始XY与复用','',
        '全部360轨迹的五张按力档总览在 ../analysis_reconstructed_v2/h75_review/，本次不重复绘制或改变坐标。每条固定规则选择见review.json selected_cases，原case身份、事件、alpha与结局可追溯。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),counts={k:d['counts'] for k,d in report.items()})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();evaluate(args.run)
