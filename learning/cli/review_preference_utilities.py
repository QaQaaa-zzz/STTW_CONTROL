"""Matched executed-alpha utility audit on already completed540 trajectories."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def review(root):
    from sttw_control.preference_trace_rescore import rescore_trace_alpha
    from sttw_control.path_command_reward import configured_cost_keys
    root=Path(root).resolve();out=root/'analysis'
    if json.loads((out/'status.json').read_text())['phase']!='completed':
        raise ValueError('completed original raw audit required')
    source=out/'results.json';query_file=out/'fixed_trace_objective_queries.csv'
    rows=json.loads(source.read_text());queries=list(csv.DictReader(query_file.open()))
    if len(rows)!=540 or len(queries)!=2700:raise ValueError('original complete540 and2700 queries required')
    q={}
    for row in queries:
        k=(row['case_id'],float(row['objective_alpha']))
        if k in q:raise ValueError('duplicate utility')
        q[k]=float(row['scored_cost'])
    groups={}
    for row in rows:
        cell=groups.setdefault((row['context'],row['role'],row['event_id']),{})
        if row['alpha'] in cell:raise ValueError('duplicate executed alpha')
        cell[row['alpha']]=row
    if len(groups)!=180 or any(set(c)!={0.,.5,1.} for c in groups.values()):
        raise ValueError('all matched executed-alpha triples required')
    records=[]
    for key,c in groups.items():
        for field in ['force_N','pulse_onset_task_tick','world_direction']:
            if any(c[a][field]!=c[0.][field] for a in [.5,1.]):raise ValueError('unmatched event')
        for user in [0.,.5,1.]:
            costs={str(a):q[(c[a]['case_id'],user)] for a in [0.,.5,1.]}
            own=costs[str(user)];best=min(costs.values())
            records.append(dict(context=key[0],role=key[1],event_id=key[2],force_N=c[0.]['force_N'],
                user_alpha=user,executed_costs=costs,requested_alpha_cost=own,
                minimum_cost_among_three=best,regret_vs_measured_three=max(0.,own-best),
                outcomes={str(a):dict(arrived=c[a]['goal_arrived'],physical=c[a]['physical_outcome']) for a in [0.,.5,1.]}))
    roles=list(dict.fromkeys(r['role'] for r in rows))
    summaries={}
    for split,forced in [('nominal',False),('disturbed',True)]:
        summaries[split]={}
        for role in roles:
            selected=[r for r in records if r['role']==role and bool(r['force_N'])==forced]
            summaries[split][role]={str(u):dict(matched_events=sum(r['user_alpha']==u for r in selected),
                mean_cost_by_executed_alpha={str(a):float(np.mean([r['executed_costs'][str(a)] for r in selected if r['user_alpha']==u])) for a in [0.,.5,1.]},
                mean_regret_vs_measured_three=float(np.mean([r['regret_vs_measured_three'] for r in selected if r['user_alpha']==u]))) for u in [0.,.5,1.]}
    cfg=json.loads((root/'common_environment_config.json').read_text());keys=configured_cost_keys(cfg)
    preference_indices=[keys.index(k) for k in ['speed','path','primary']]
    components=[]
    # All nominal trajectories survive; no outcome-based subselection here.
    for row in rows:
        if row['force_N']:continue
        if row['physical_outcome']!='survived':raise ValueError('nominal component cohort changed')
        with np.load(row['raw_file'],allow_pickle=False) as z:
            data={k:z[k] for k in ['active_tick','effective_costs','chi','actual_forward_speed',
                'requested_speed','path_cross_track','path_heading_error','contact_failure']}
        for user in [0.,.5,1.]:
            score=rescore_trace_alpha(data,alpha=user,config=cfg)
            expected=q[(row['case_id'],user)]
            np.testing.assert_allclose(score['scored_cost'],expected,rtol=0,atol=1e-6)
            pref=float(score['effective_costs'][:,preference_indices].sum()*cfg['path_reward']['scale']*cfg['timing']['lower_dt_s'])
            components.append(dict(case_id=row['case_id'],role=row['role'],context=row['context'],
                executed_alpha=row['alpha'],user_alpha=user,preference_dependent_cost=pref,
                alpha_independent_cost=expected-pref,total_cost=expected))
    result=dict(summary=summaries,records=records,nominal_components=components,
        source_results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_queries_sha256=hashlib.sha256(query_file.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0,optimizer_updates=0,
        scope='Exploratory after RMSE and task outcomes were seen. Same event and original user objective compares three actually executed alpha trajectories, not predicted dynamics. Regret is only relative to measured three, not global optimality or deployment selection. All disturbed failures/contact costs retained; nominal component analysis includes all108 nominal traces, none selected by outcome. Independent costs are alpha-independent for rescoring one fixed trace, but vary with actual policy behavior. Related DEV, no independent CI, new rollout, alpha remapping or fit. Tiny zero-neural differences are numerical variation, not preference ability.',visual_review_complete=False)
    dest=out/'entry_preference_utility_review.json'
    if dest.exists():raise ValueError('preserve previous utility audit')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    render(out,result)
    print(json.dumps(summaries['nominal'],ensure_ascii=False))


def render(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full=[r for r in result['nominal_components'] if r['role']=='goal_full_feedback']
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['total_cost','preference_dependent_cost','alpha_independent_cost'],
                           ['Complete original cost','Preference channels only','Other running channels']):
        values=np.array([[np.mean([r[key] for r in full if r['user_alpha']==u and r['executed_alpha']==a]) for a in [0.,.5,1.]] for u in [0.,.5,1.]])
        im=ax.imshow(values,cmap='Blues',aspect='equal',vmin=0)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>.6*values.max() else 'black')
        ax.set(xticks=range(3),xticklabels=['0','.5','1'],yticks=range(3),yticklabels=['0','.5','1'],
            xlabel='Actually executed alpha',ylabel='User objective alpha',title=title)
        fig.colorbar(im,ax=ax,shrink=.7,label='Mean cost (panel-specific scale)')
    fig.suptitle('Full neural middle: same12 nominal events, three measured alpha trajectories\n'
        'No global optimum, remapping or independent-repeat claim; other channels vary with behavior',fontsize=10)
    fig.savefig(out/'entry_preference_objective_matrix.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    review(p.parse_args().run)
