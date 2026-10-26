"""Describe the complete fixed four-combination steering intervention."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

ORDER=['goal_zero_feedback','neural_before_only','neural_after_only','goal_steer_feedback']
EDGES={'early_when_late_off':(1,0),'early_when_late_on':(3,2),
       'late_when_early_off':(2,0),'late_when_early_on':(3,1)}


def audit_prefixes(root,groups):
    """Describe actual matched-early prefixes; never demand exact replay."""
    fields=['actual_xy','actual_forward_speed','phi','phi_dot','actual_delta','nominal_reference']
    records=[];samples=[]
    for key,cell in groups.items():
        loaded={}
        for role,row in cell.items():
            with np.load(row['raw_file'],allow_pickle=False) as z:
                loaded[role]={name:z[name] for name in fields+['active_tick','physical_tick','applied_force_impulse']}
            raw=loaded[role];onset=row['pulse_onset_task_tick']
            at=np.flatnonzero(raw['active_tick']&(raw['physical_tick']==onset-1))
            forced=np.flatnonzero(raw['active_tick']&(np.abs(raw['applied_force_impulse']).sum(axis=-1)>0))
            first_force=int(raw['physical_tick'][forced[0]]) if len(forced) else None
            samples.append(dict(case_id=row['case_id'],role=role,
                sample_task_tick=onset-1,available=bool(len(at)),
                first_measured_force_task_tick=first_force,
                sample_precedes_first_measured_force=(onset-1<first_force) if first_force is not None else None,
                values={name:np.asarray(raw[name][at[0]]).tolist() for name in fields} if len(at) else None))
        for target,baseline in [('neural_before_only','goal_steer_feedback'),('neural_after_only','goal_zero_feedback')]:
            a,b=loaded[target],loaded[baseline]
            active=a['active_tick']&b['active_tick']&(a['physical_tick']<cell[target]['pulse_onset_task_tick'])
            if np.any(a['physical_tick'][active]!=b['physical_tick'][active]):raise ValueError('different logged prefix clocks')
            differences={name:float(np.max(np.abs(a[name][active]-b[name][active]))) if np.any(active) else None for name in fields}
            records.append(dict(context=key[0],alpha=key[1],event_id=key[2],target=target,baseline=baseline,
                common_planned_preswitch_ticks=int(active.sum()),max_abs_recorded_differences=differences))
    result=dict(comparisons=records,precut_samples=samples,physics_steps=0,optimizer_updates=0,
        scope='Read-only recorded samples just before the planned switch and max differences of pairs sharing an early request policy. Prefix slicing uses planned onset; actual first measured impulse is separately recorded, not assumed. Raw identities already authenticated by full analysis. Small batch/replay variation is accepted; no new equality gate or forced onset state matching. Physical fields are post-tick measurements; nominal_reference is the issued pre-tick reference. Logs are not complete snapshots for state swapping or proof of one-variable causation.')
    file=root/'analysis/stage_prefix_audit.json'
    if file.exists():raise ValueError('keep original prefix audit immutable')
    file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def review(root):
    root=Path(root).resolve()
    for file in ['status.json','analysis/status.json']:
        if json.loads((root/file).read_text())['phase']!='completed':raise ValueError('complete analysis required')
    plan=json.loads((root/'plan.json').read_text())
    if plan['experiment']!='steering_stage_mechanism_v1':raise ValueError('fixed stage experiment required')
    source=root/'analysis/results.json';rows=json.loads(source.read_text());groups={}
    for row in rows:
        key=(row['context'],row['alpha'],row['event_id'])
        cell=groups.setdefault(key,{})
        if row['role'] in cell:raise ValueError('duplicate stage')
        cell[row['role']]=row
        if row['physical_outcome'] not in ['survived','contact'] or row['goal_arrived'] not in [True,False]:
            raise ValueError('unknown retained; complete endpoint assessment required')
    if len(rows)!=720 or len(groups)!=180:raise ValueError('all720 trajectories required')
    for cell in groups.values():
        if set(cell)!=set(ORDER):raise ValueError('complete factorial cell required')
        for field in ['force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s']:
            if any(cell[r][field]!=cell[ORDER[0]][field] for r in ORDER):raise ValueError('mismatched event '+field)
    prefix_audit=audit_prefixes(root,groups)
    def summarize(cells):
        def value(row,metric):return int(row['goal_arrived'] if metric=='arrival' else row['physical_outcome']=='survived')
        result=dict(total_matched_events=len(cells),counts={},effects={})
        for role in ORDER:
            result['counts'][role]=dict(total=len(cells),arrived=sum(value(c[role],'arrival') for c in cells),
                survived=sum(value(c[role],'survival') for c in cells))
        for metric in ['arrival','survival']:
            effects={}
            for name,(target,base) in EDGES.items():
                delta=[value(c[ORDER[target]],metric)-value(c[ORDER[base]],metric) for c in cells]
                effects[name]=dict(gains=sum(x>0 for x in delta),losses=sum(x<0 for x in delta),
                    net=sum(delta),mean_difference=float(np.mean(delta)) if delta else None)
            interaction=[value(c[ORDER[3]],metric)-value(c[ORDER[1]],metric)-value(c[ORDER[2]],metric)+value(c[ORDER[0]],metric) for c in cells]
            effects['interaction']=dict(sum=sum(interaction),mean=float(np.mean(interaction)) if interaction else None)
            result['effects'][metric]=effects
        return result
    cells=list(groups.values());zero=lambda c:c[ORDER[0]]
    splits=dict(nominal=[c for c in cells if zero(c)['force_N']==0],disturbed=[c for c in cells if zero(c)['force_N']!=0],
                prior_loss_cluster_left55_negative32=[c for c in cells if zero(c)['geometry_name']=='left55_R1p8_v1p1' and zero(c)['force_N']==-32])
    result=dict(summary={k:summarize(v) for k,v in splits.items()},
        by_alpha={str(a):summarize([c for c in splits['disturbed'] if zero(c)['alpha']==a]) for a in [0.,.5,1.]},
        source_results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0,optimizer_updates=0,
        prefix_comparisons=len(prefix_audit['comparisons']),precut_samples=len(prefix_audit['precut_samples']),
        scope='Request-stage total effects, including persistent filtering/slew/history effects. Exact switch is privileged experimental intervention, not deployable detection. Different early policies generate their own real onset states.360 cached endpoints plus360 new hybrid traces; shared ancestor, correlated cases. No independent confidence intervals, onset scan, learned selector or continuous disturbance certification.',visual_review_complete=False)
    out=root/'analysis/stage_factorial_review.json'
    if out.exists():raise ValueError('keep previous derived review immutable')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,split in enumerate(['nominal','disturbed']):
        for col,metric in enumerate(['arrived','survived']):
            ax=axes[row,col];data=result['summary'][split]['counts'];n=result['summary'][split]['total_matched_events']
            values=[data[r][metric] for r in ORDER]
            ax.bar(range(4),np.asarray(values)/n,color=['#0072B2','#E69F00','#009E73','#CC79A7'])
            for i,v in enumerate(values):ax.text(i,v/n+.025,f'{v}/{n}',ha='center')
            ax.set(xticks=range(4),xticklabels=['00\nCached zero','10\nBefore only','01\nAfter only','11\nCached steering'],
                title=split+' / '+metric,ylabel='Fraction of fixed paired events',ylim=(0,1.12));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Privileged steering-request stage intervention — same task0, real forward histories\nPersistent reference/history effects retained; not a deployable switching policy')
    fig.savefig(root/'analysis/stage_factorial_outcomes.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);review(p.parse_args().run)
