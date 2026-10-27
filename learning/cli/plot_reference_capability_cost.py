"""Observed capability/cost points, with original-alpha and shared ancestry explicit.

Read-only derived figure: no fitting, interpolation, simulation or selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def plot(root,*,confirmation=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    root=root.resolve()
    source=root/('preservation_timecheck_v1/analysis' if confirmation else 'production/analysis')
    state=json.loads((source/'status.json').read_text())
    if state['phase']!='completed':raise ValueError('completed audited results required')
    file=source/'results.json';rows=json.loads(file.read_text())
    methods=(['full','context_choice','geometry_alpha','alpha_only'] if confirmation else
             ['goal_full_feedback','goal_zero_feedback','goal_steer_feedback','neural_after_only','cem','random'])
    labels=(['Full middle','Context + alpha choice','Geometry + alpha choice','Alpha-only role choice'] if confirmation else
            ['Full middle','Zero middle','Steering only','After-only (privileged onset)','Nominal CEM','Nominal random'])
    colors=['#222222','#0072B2','#D55E00','#009E73'] if confirmation else ['#222222','#777777','#D55E00','#AA4499','#0072B2','#009E73']
    markers=['o','s','^','D','P','X'];points=[]
    expected_disturbed=32 if confirmation else 16
    for alpha in [0.,.5,1.]:
        baseline=[r for r in rows if r['role']==methods[0] and r['alpha']==alpha and not r['force_N']]
        if len(baseline)!=4:raise ValueError('four original nominal contexts required')
        base=float(np.mean([r['scored_cost'] for r in baseline]))
        if not np.isfinite(base) or base<=0:raise ValueError('positive baseline cost required for ratio')
        for method,label in zip(methods,labels):
            nominal=[r for r in rows if r['role']==method and r['alpha']==alpha and not r['force_N']]
            disturbed=[r for r in rows if r['role']==method and r['alpha']==alpha and r['force_N']]
            if len(nominal)!=4 or len(disturbed)!=expected_disturbed:raise ValueError('complete fixed panel required')
            if any(r['goals']['1.0']['goal_arrived'] is None or r['physical_outcome'] not in ('survived','contact') for r in disturbed):
                raise ValueError('unknown outcomes cannot become failures in a point plot')
            cost=float(np.mean([r['scored_cost'] for r in nominal]))
            points.append(dict(alpha=alpha,method=method,label=label,nominal_cost=cost,
                original_alpha_nominal_cost_ratio=cost/base,nominal_n=4,disturbed_n=len(disturbed),
                arrived=sum(r['goals']['1.0']['goal_arrived'] for r in disturbed),
                survived=sum(r['physical_outcome']=='survived' for r in disturbed),
                nominal_case_ids=[r['case_id'] for r in nominal],disturbed_case_ids=[r['case_id'] for r in disturbed]))
    out=source/'capability_cost_points'
    if out.exists():
        previous=json.loads((out/'provenance.json').read_text())
        if previous['source_results_sha256']!=digest(file):raise ValueError('do not overwrite a figure from changed source results')
    out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    maximum=max(p['original_alpha_nominal_cost_ratio'] for p in points)
    for j,alpha in enumerate([0.,.5,1.]):
        selected=[p for p in points if p['alpha']==alpha]
        for i,metric in enumerate(['arrived','survived']):
            ax=axes[i,j];full=selected[0]
            for p,color,marker in zip(selected,colors,markers):
                ax.scatter(p['original_alpha_nominal_cost_ratio'],p[metric],edgecolors=color,marker=marker,s=70,
                           facecolors='none' if marker in ('o','s','D') else color,linewidths=1.4,zorder=3)
            ax.axvline(1.,c='gray',ls=':',lw=.8);ax.axhline(full[metric],c='gray',ls=':',lw=.8)
            ax.set(xlabel='Nominal original-alpha cost / full-middle cost',
                   ylabel=f'{"Arrivals within 1 m" if metric=="arrived" else "Survived full 10 s"} / {expected_disturbed}',
                   title=f'Original user alpha = {alpha:g}',xlim=(0,maximum*1.08),ylim=(-.5,expected_disturbed+.5))
            ax.grid(alpha=.2)
    handles=[Line2D([],[],color=c,marker=m,ls='',label=label,markersize=7) for c,m,label in zip(colors,markers,labels)]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Observed task-cost / disturbance-capability points\n'
                 'Lower cost and higher capability preferred; fixed DEV panel, shared ancestry, no independent-repeat CI',fontsize=12)
    fig.savefig(out/'observed_points.png',dpi=150,facecolor='white');plt.close(fig)
    atomic_json(out/'points.json',points)
    atomic_json(out/'provenance.json',dict(source_results=str(file),source_results_sha256=digest(file),
        script_sha256=digest(__file__),new_physics_steps=0,optimizer_updates=0,
        transformations='Four-context mean nominal cost within each original user alpha, divided by same-alpha full-middle mean; disturbed success counts. No smoothing, interpolation, jitter, hypothesis test or fitted frontier.',
        limitations='Nominal costs are not disturbed task costs. Correlated fixed DEV branches, not independent starts/seeds; no population confidence interval. Overlapping points retained without displacement. No continuous robustness or safety guarantee.',
        figure_status='Research diagnostic; exact Transactions journal and export requirements pending.'))
    (out/'README.md').write_text('# 实测任务代价与抗扰点图\n\n'
        '[全部原alpha的实测点](observed_points.png)；底层数字与逐点case IDs见points.json，来源SHA见provenance.json。\n\n'
        '横轴是无扰原alpha任务成本相对完整中层的比值，越小越好；纵轴分别是受扰到达和全10秒不触地，越高越好。'
        '不把无扰成本当受扰成本，不把二者相乘为新分数。每alpha含4个相关场景；共享初态与重复分支不当独立训练种子，'
        '不画人口置信区间、插值线或连续Pareto边界；重合实测点保留不抖动。原alpha不同列，不横向比较原始成本。'
        '无新增物理、训练或选择；该图不是已满足期刊规范的声明。\n')
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--confirmation',action='store_true');args=parser.parse_args()
    plot(args.root,confirmation=args.confirmation)
