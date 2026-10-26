"""Matched alpha consequences on completed entry-history evaluation traces.

No cost comparison across different alpha objectives; no independent-sample CI.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def review(root):
    root=Path(root).resolve()
    for file in ['status.json','analysis/status.json']:
        if json.loads((root/file).read_text())['phase']!='completed':
            raise ValueError('complete physical execution and raw analysis required')
    plan=json.loads((root/'plan.json').read_text())
    if plan['experiment']!='entry_history_confirmation_v1':raise ValueError('fixed entry-history panel required')
    results=root/'analysis/results.json';rows=json.loads(results.read_text())
    groups={}
    for row in rows:
        key=(row['context'],row['role'],row['force_N'],row['event_id'])
        cell=groups.setdefault(key,{})
        if row['alpha'] in cell:raise ValueError('duplicate alpha event')
        cell[row['alpha']]=row
    if len(rows)!=540 or len(groups)!=180:raise ValueError('all fixed540 cases required')
    records=[]
    for key,cell in groups.items():
        if set(cell)!={0.,.5,1.}:raise ValueError('matched three alpha required')
        for field in ['pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s']:
            if any(cell[a][field]!=cell[0.][field] for a in (.5,1.)):
                raise ValueError('alpha events differ: '+field)
        first,last=cell[0.],cell[1.]
        valid=all(r['physical_outcome']=='survived' and r['complete_task_observation'] for r in cell.values())
        records.append(dict(context=key[0],role=key[1],force_N=key[2],event_id=key[3],
            prefix_name=first['prefix_name'],geometry_name=first['geometry_name'],
            common_full_window_survivors=valid,
            alpha1_arrival_gain=last['goal_arrived'] is True and first['goal_arrived'] is False,
            alpha1_arrival_loss=first['goal_arrived'] is True and last['goal_arrived'] is False,
            alpha1_survival_gain=last['physical_outcome']=='survived' and first['physical_outcome']=='contact',
            alpha1_survival_loss=first['physical_outcome']=='survived' and last['physical_outcome']=='contact',
            values={str(a):dict(speed_rmse_m_s=r['task_speed_rmse_m_s'],path_rmse_m=r['task_path_rmse_m'],
                arrived=r['goal_arrived'],physical_outcome=r['physical_outcome']) for a,r in cell.items()}))
    def summarize(records):
        common=[r for r in records if r['common_full_window_survivors']]
        result=dict(total_matched_events=len(records),common_full_window_events=len(common),
                    excluded_continuous_metric_events=len(records)-len(common))
        for field in ['alpha1_arrival_gain','alpha1_arrival_loss','alpha1_survival_gain','alpha1_survival_loss']:
            result[field]=sum(r[field] for r in records)
        result['means_on_same_common_events']={str(a):{metric:float(np.mean([r['values'][str(a)][metric] for r in common])) if common else None
            for metric in ['speed_rmse_m_s','path_rmse_m']} for a in [0.,.5,1.]}
        return result
    summary={split:{role:summarize([r for r in records if r['role']==role and (r['force_N']!=0)==disturbed])
                    for role in plan['evaluation_roles']} for split,disturbed in [('nominal',False),('disturbed',True)]}
    out=root/'analysis/entry_preference_review.json'
    if out.exists():raise ValueError('keep old derived analysis immutable')
    result=dict(summary=summary,records=records,source_results_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0,optimizer_updates=0,
        scope='Descriptive matched-alpha consequences; all event labels retained. Continuous means use the same events surviving for all three alpha over10s within each role, with exclusions counted. Different roles may have different survivor sets; the plotted points are not fair cross-role rankings. Do not compare different-alpha objective costs, infer a Pareto frontier or independent confidence intervals. Shared ancestor and frozen Actor; no independent training repeat.',visual_review_complete=False)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    render(root,result,plan['evaluation_roles'])
    print(json.dumps(summary,ensure_ascii=False))


def render(root,result,roles,*,readable=False):
    summary=result['summary']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    names={'goal_zero_feedback':'Zero neural middle','goal_steer_feedback':'Neural steering only','goal_full_feedback':'Full neural middle'}
    for row,split in enumerate(['nominal','disturbed']):
        for col,role in enumerate(roles):
            ax=axes[row,col];data=summary[split][role]
            for a,color,marker in [(0.,'#0072B2','o'),(.5,'#009E73','s'),(1.,'#D55E00','^')]:
                point=data['means_on_same_common_events'][str(a)]
                if point['path_rmse_m'] is not None:
                    ax.scatter(point['path_rmse_m'],point['speed_rmse_m_s'],color=color,marker=marker,s=55,label=f'alpha={a:g}')
            ax.set(title=f'{names[role]} / {split}\nSame events: {data["common_full_window_events"]}/{data["total_matched_events"]}',
                   xlabel='Mean whole-task path RMSE (m)',ylabel='Mean whole-task speed RMSE (m/s)')
            if readable:
                # Do not visually magnify float32 noise in alpha-invariant
                # zero-middle baselines. Keep each original point unchanged.
                points=list(data['means_on_same_common_events'].values())
                for axis,key,minimum in [('x','path_rmse_m',.05),('y','speed_rmse_m_s',.02)]:
                    values=[p[key] for p in points if p[key] is not None]
                    if values:
                        center=(max(values)+min(values))/2
                        half=max(minimum,max(values)-min(values))*0.65
                        (ax.set_xlim if axis=='x' else ax.set_ylim)(max(0.,center-half),center+half)
                ax.ticklabel_format(style='plain',axis='both',useOffset=False)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Within-role matched alpha consequences — survivor sets may differ across roles\nDiscrete DEV points; arrival and contact losses retained in JSON; shared initial ancestor')
    name='entry_preference_consequences_readable.png' if readable else 'entry_preference_consequences.png'
    fig.savefig(root/'analysis'/name,dpi=150,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--render-readable',action='store_true')
    args=parser.parse_args()
    if args.render_readable:
        root=args.run.resolve()
        result=json.loads((root/'analysis/entry_preference_review.json').read_text())
        plan=json.loads((root/'plan.json').read_text())
        render(root,result,plan['evaluation_roles'],readable=True)
    else:review(args.run)
