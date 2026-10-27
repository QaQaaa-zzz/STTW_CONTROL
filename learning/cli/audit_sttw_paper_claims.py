"""Reconcile existing STTW claims without pooling incompatible DEV cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def groups(rows,roles):
    result={}
    for r in rows:
        key=(r['context'],r['alpha'],r['event_id'],r['force_N'])
        g=result.setdefault(key,{})
        if r['role'] in g:raise ValueError('duplicate case within one cohort')
        g[r['role']]=r
    if any(set(g)!=set(roles) for g in result.values()):raise ValueError('incomplete matched cohort')
    for g in result.values():
        for field in ['pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s']:
            if any(r[field]!=next(iter(g.values()))[field] for r in g.values()):
                raise ValueError('unmatched actual event/interface')
    return list(result.values())


def endpoints(matched,roles,radius):
    disturbed=[g for g in matched if next(iter(g.values()))['force_N']]
    arrived=lambda r:r['minimum_upright_goal_distance_m']<=radius
    alive=lambda r:r['physical_outcome']=='survived'
    counts={m:dict(n=len(disturbed),arrived=sum(arrived(g[m]) for g in disturbed),
        survived=sum(alive(g[m]) for g in disturbed)) for m in roles}
    candidate=roles[-1];pairs={}
    for baseline in roles[:-1]:
        detail=[dict(case_id=g[candidate]['case_id'],baseline_case_id=g[baseline]['case_id'],
            arrival_change=int(arrived(g[candidate]))-int(arrived(g[baseline])),
            survival_change=int(alive(g[candidate]))-int(alive(g[baseline]))) for g in disturbed]
        pairs[candidate+'_vs_'+baseline]=dict(arrival_gain=sum(d['arrival_change']==1 for d in detail),
            arrival_loss=sum(d['arrival_change']==-1 for d in detail),
            survival_gain=sum(d['survival_change']==1 for d in detail),
            survival_loss=sum(d['survival_change']==-1 for d in detail),details=detail)
    return dict(radius_m=radius,counts=counts,pairs=pairs)


def main(root):
    root=root.resolve();out=root/'analysis/paper_claim_audit';out.mkdir(exist_ok=False)
    sources={}
    def read(file):
        file=Path(file);data=json.loads(file.read_text());sources[str(file.resolve())]=digest(file)
        return data
    native=read('runs/current_native_preference_v1_attempt2/analysis/review.json')
    matrices={};native_counts={};tradeoffs={}
    for family in ['H75','current']:
        rr=[r for r in native['records'] if r['family']==family]
        if len(rr)!=8 or any(not r['native_complete_horizon'] for r in rr):raise ValueError('all8 native trajectories required')
        matrix=[[float(np.mean([r['user_objective_costs'][str(user)] for r in rr if r['alpha']==executed]))
            for executed in [0.,1.]] for user in [0.,1.]]
        np.testing.assert_allclose(matrix,native['matrices_rows_user_cols_executed'][family],atol=1e-10)
        matrices[family]=matrix
        native_counts[family]={str(radius):dict(n=8,arrived=sum(r['minimum_goal_distance_before_native_stop_m']<=radius for r in rr),
            survived=sum(r['native_physical_survived_horizon'] for r in rr)) for radius in [.5,.75,1.]}
        contexts={}
        for r in rr:contexts.setdefault((r['route'],r['speed']),{})[r['alpha']]=r
        if len(contexts)!=4 or any(set(g)!={0.,1.} for g in contexts.values()):raise ValueError('native endpoint pairing differs')
        tradeoffs[family]=dict(contexts=4,path_priority_better_path=sum(g[0.]['path_rmse_m']<g[1.]['path_rmse_m'] for g in contexts.values()),
            speed_priority_better_speed=sum(g[1.]['speed_rmse_m_s']<g[0.]['speed_rmse_m_s'] for g in contexts.values()))
    lowfile='runs/h75_disturbed_common_interface_v1/analysis_reconstructed_v2/results.json'
    low=read(lowfile);roles=['goal_zero_feedback','lineage_h75','goal_full_feedback']
    if len(low)!=360:raise ValueError('complete360 low-transfer cohort required')
    matched=groups(low,roles);low_readback={str(t):endpoints(matched,roles,t) for t in [.5,.75,1.]}
    old=read('runs/h75_disturbed_common_interface_v1/analysis_reconstructed_v2/h75_review/review.json')
    for radius,q in low_readback.items():
        prior=old['results']['disturbed'][radius]
        for role,values in q['counts'].items():
            if any(values[k]!=prior['counts'][role][k] for k in ['n','arrived','survived']):raise ValueError('low endpoint count conflict')
        for pair,v in q['pairs'].items():
            if any(v[k]!=prior['pairs'][pair][k] for k in ['arrival_gain','arrival_loss','survival_gain','survival_loss']):raise ValueError('paired claim conflict')
    pause=read('runs/same_state_reference_pause_v1/analysis_partial_block_fix_v1/results.json')
    if len(pause)!=360:raise ValueError('complete360 pause cohort required')
    pause_roles=['full_current','pause_neural_both','pause_neural_steer'];pm=groups(pause,pause_roles)
    pause_readback={};witnesses=[]
    for radius in [.5,.75,1.]:
        counts={};pairs={}
        for candidate in pause_roles[1:]:
            q=endpoints([{k:g[k] for k in ['full_current',candidate]} for g in pm],['full_current',candidate],radius)
            counts.update(q['counts']);pairs.update(q['pairs'])
        pause_readback[str(radius)]=dict(counts=counts,pairs=pairs)
    for g in pm:
        if not g['full_current']['force_N']:continue
        if len({r['fork_task_tick'] for r in g.values()})!=1 or len({r['prefix_index'] for r in g.values()})!=1:
            raise ValueError('same-state pause prefix mismatch')
        for method in pause_roles[1:]:
            b,c=g['full_current'],g[method]
            arrival_gain=c['minimum_upright_goal_distance_m']<=1<b['minimum_upright_goal_distance_m']
            survival_gain=c['physical_outcome']=='survived' and b['physical_outcome']=='contact'
            if arrival_gain or survival_gain:
                if not (c['common_state_branch'] and c['reached_fork']):raise ValueError('unreached branch cannot witness intervention')
                witnesses.append(dict(method=method,context=c['context'],alpha=c['alpha'],event_id=c['event_id'],
                    arrival_gain_1m=arrival_gain,survival_gain=survival_gain,fork_task_tick=c['fork_task_tick'],
                    prefix_index=c['prefix_index'],baseline_case_id=b['case_id'],candidate_case_id=c['case_id'],
                    original_distance_m=b['minimum_upright_goal_distance_m'],candidate_distance_m=c['minimum_upright_goal_distance_m'],
                    pause_lower_ticks=c['pause_lower_ticks'],scope='Existing audited complete-state fork; event-locked timing, not a deployable detector'))
    teaching=read('docs/evidence/tolerance_progress_20261022/long_baseline_transfer/results.json')
    teaching_totals=dict(n=sum(p['seed918501']['n'] for p in teaching['panels'].values()),
        ordinary_ppo=sum(p['seed918501']['long75'] for p in teaching['panels'].values()),
        teaching=sum(p['seed918501']['other'] for p in teaching['panels'].values()),
        teaching_gain=sum(p['seed918501']['other_gains'] for p in teaching['panels'].values()),
        teaching_loss=sum(p['seed918501']['other_losses'] for p in teaching['panels'].values()),
        scope='Historical task-success contract; not physical falls or current1m arrival')
    for f,sha in teaching['sources'].items():
        if digest(f)!=sha:raise ValueError('historical teaching summary identity differs')
    timing=read('runs/reference_bias_opportunity_v1/preservation_timecheck_v1/analysis/summary.json')
    permission=read(root/'analysis/summary.json')
    result=dict(native=dict(matrices_rows_user_cols_executed=matrices,counts=native_counts,tradeoffs=tradeoffs,
        scope='Shared four native high-demand contexts;8 traces per model; not independent repeats or causal training ablation'),
        low_transfer=low_readback,pause=pause_readback,existing_same_state_positive_witnesses=witnesses,
        historical_teaching=teaching_totals,new_time_lookup=timing['methods'],permission_factor=permission['disturbed'],
        source_sha256=sources,new_physics_steps=0,optimizer_updates=0,
        scope='Readback of existing authenticated case summaries plus matching cohort metadata. Raw physics audits inherited, not repeated. Do not pool cohorts, turn historical task labels into falls, or infer training causality from frozen weight comparisons.')
    atomic_json(out/'claims_readback.json',result)
    print(json.dumps(dict(native=native_counts,low_1m=low_readback['1.0']['counts'],same_state_positive_witnesses=len(witnesses),teaching=teaching_totals),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
