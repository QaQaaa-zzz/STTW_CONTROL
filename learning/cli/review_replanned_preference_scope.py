"""Separate original-route geometry, executed-route tracking and reference TV.

No scalar utility is chosen while the user meaning of path preference after
replanning remains unresolved. All outputs are retrospective DEV diagnostics.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from sttw_control.goal_arrival_analysis import digest
from sttw_control.rejoin_comparison import geometric_task_trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.worktree.resolve()
    source=root/'runs/goal_replanning_h75_v1/combined_analysis/results.json'
    rows=json.loads(source.read_text())
    if len(rows)!=72:raise ValueError('Complete72 required')
    campaigns={};inputs={};records=[];switches=[]
    for name in ['goal_replanning_github_v1','goal_replanning_h75_v1']:
        base=root/'runs'/name
        plan=json.loads((base/'plan.json').read_text())
        cfg=json.loads((base/'common_environment_config.json').read_text())
        index=json.loads((base/'physical/case_index.json').read_text())
        with (base/'physical/initial_carry.pkl').open('rb') as f:initial=pickle.load(f)
        # TeleopEnv.observe -> RecoveryEnv.pose uses actual qpos[0:2]. Its
        # bookkeeping reference_pose can be metres behind and is NOT actual XY.
        initial_xy=np.asarray(initial.state.physical.data.qpos)[:,:2]
        progress=np.asarray(initial.state.projection.progress)
        if np.any(progress!=0):raise ValueError('This shadow helper assumes initial projection0')
        initial_reference=np.asarray(initial.state.physical.governor.current_reference)
        if len(index)!=len(initial_xy):raise ValueError('Initial lane and case index mismatch')
        origins={r['case_id']:(initial_xy[i],initial_reference[i]) for i,r in enumerate(index)}
        decisions={r['case_id']:r for r in json.loads((base/'physical/replanning_decisions.json').read_text())}
        campaigns[name]=(plan,cfg,origins,decisions)
        for rel in ['plan.json','common_environment_config.json','physical/case_index.json',
                    'physical/initial_carry.pkl','physical/replanning_decisions.json']:
            f=base/rel;inputs[str(f)]=digest(f)
    for number,row in enumerate(rows):
        name='goal_replanning_h75_v1' if row['model']=='H75' else 'goal_replanning_github_v1'
        plan,cfg,origins,decisions=campaigns[name]
        context=plan['contexts'][row['context']]
        if digest(context['path'])!=context['path_sha256'] or digest(row['raw_file'])!=row['raw_sha256']:
            raise ValueError('Original path or raw changed')
        with np.load(context['path'],allow_pickle=False) as z:path={k:z[k] for k in z.files}
        with np.load(row['raw_file'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        active=raw['active_tick'].astype(bool)
        n=int(active.sum())
        if not np.array_equal(raw['physical_tick'][active],np.arange(n)):
            raise ValueError('Actual physical prefix required')
        if np.any(active&~raw['task_reward_valid'].astype(bool)):
            raise ValueError('No current-path cost claims for an invalid reward tail')
        start_xy,start_reference=origins[row['case_id']]
        shadow=geometric_task_trace(raw,path,cfg,initial_xy=start_xy)
        fatal=np.flatnonzero(active&(raw['contact_failure']|raw['nonfinite']|raw['lower_fault']|raw['policy_fault']))
        stop=int(fatal[0]) if len(fatal) else n
        arrival=row['horizons']['20']['first_arrival_s']
        cut=min(stop,int(round(arrival/.005))) if arrival is not None else stop
        if not cut:raise ValueError('Nonempty measured task prefix required')
        speed=raw['actual_forward_speed'][:cut]-row['requested_speed_m_s']
        reference=np.asarray(raw['reference'][:cut],float)
        variation=np.abs(np.diff(np.vstack([start_reference,reference]),axis=0)).sum(axis=0)
        rms=lambda a:float(np.sqrt(np.mean(np.asarray(a)[:cut]**2)))
        record=dict(case_id=row['case_id'],model=row['model'],planning=row['planning'],
            context=row['context'],alpha=row['alpha'],event_id=row['event_id'],force_N=row['force_N'],
            arrived=arrival is not None,survived=row['horizons']['20']['physical_outcome']=='survived',
            task_prefix_s=cut*.005,original_route_cross_track_rmse_m=rms(shadow['path_cross_track']),
            executed_route_cross_track_rmse_m=rms(raw['path_cross_track']),
            original_route_heading_rmse_rad=rms(shadow['path_heading_error']),
            executed_route_heading_rmse_rad=rms(raw['path_heading_error']),
            speed_rmse_m_s=float(np.sqrt(np.mean(speed**2))),speed_mae_m_s=float(np.mean(abs(speed))),
            reference_speed_total_variation_m_s=float(variation[0]),
            reference_steer_total_variation_rad=float(variation[1]),
            no_replan_shadow_cross_track_max_error_m=(float(np.max(abs(shadow['path_cross_track'][:n]-raw['path_cross_track'][:n]))) if not row['replanned'] else None),
            raw_sha256=row['raw_sha256'])
        records.append(record)
        decision=decisions.get(row['case_id'])
        if bool(decision)!=row['replanned']:raise ValueError('Full decision identity changed')
        if decision:
            i=decision['task_tick']
            if not 0<i<cut:
                # A genuine plan before eventual contact can have no completed
                # task; retain it provided the immediate measured step exists.
                if not 0<i<n:raise ValueError('No observed post-rebind tick')
            switches.append(dict(case_id=row['case_id'],replan_s=i*.005,
                measured_xy_step_m=float(np.linalg.norm(raw['actual_xy'][i]-raw['actual_xy'][i-1])),
                old_logged_cross_track_m=float(raw['path_cross_track'][i-1]),
                new_logged_cross_track_m=float(raw['path_cross_track'][i]),
                original_route_cross_track_same_post_tick_m=float(shadow['path_cross_track'][i]),
                reference_step= (raw['reference'][i]-raw['reference'][i-1]).tolist()))
        if (number+1)%24==0:print('geometry_records_completed',number+1,flush=True)
    comparisons=[]
    metrics=['task_prefix_s','original_route_cross_track_rmse_m','executed_route_cross_track_rmse_m',
             'original_route_heading_rmse_rad','executed_route_heading_rmse_rad','speed_rmse_m_s',
             'reference_speed_total_variation_m_s','reference_steer_total_variation_rad']
    for candidate in ['H75','current64']:
        for alpha in [0.,1.]:
            aa=[r for r in records if r['model']==candidate and r['planning']=='replan' and r['alpha']==alpha]
            for control_planning in ['original','replan']:
                details=[]
                for a in aa:
                    b=next(b for b in records if b['model']=='zero' and b['planning']==control_planning and
                           (b['context'],b['alpha'],b['event_id'])==(a['context'],a['alpha'],a['event_id']))
                    if a['arrived'] and b['arrived'] and a['survived'] and b['survived']:
                        details.append(dict(case_id=a['case_id'],control_case_id=b['case_id'],
                            **{k:a[k]-b[k] for k in metrics}))
                comparisons.append(dict(candidate=candidate+'/replan',control='zero/'+control_planning,alpha=alpha,
                    both_arrived_and_survived=len(details),details=details,
                    means={k:float(np.mean([r[k] for r in details])) for k in metrics}))
    args.output.mkdir(exist_ok=False)
    with (args.output/'records.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(records=records,reference_switches=switches,comparisons=comparisons,
        preference_semantics='User clarification pending: original versus replanned route priority. No scalar utility or canonical alpha ranking chosen.',
        source_sha256={**inputs,str(source):digest(source),str(Path(__file__)):digest(__file__),
                       str(root/'learning/src/sttw_control/rejoin_comparison.py'):digest(root/'learning/src/sttw_control/rejoin_comparison.py')},
        raw_sha_checked=72,new_physics_steps=0,optimizer_updates=0,
        limits='Original-route causal geometric projection is a shadow diagnostic, not counterfactual controller execution. Executed route changes at replan; its error has a different target. TV measures actual governed references, not motor effort. Own first-arrival prefixes differ; common-success subsets can differ, so no global Pareto or causal ranking. Old scored reward is not recomputed as a common user utility.')
    (args.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(records=72,switches=len(switches),comparisons=[{k:v for k,v in r.items() if k!='details'} for r in comparisons]),indent=2))


if __name__=='__main__':main()
