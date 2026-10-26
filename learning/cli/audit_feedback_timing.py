"""Reconstruct force timing and feedback intrusion from authenticated old logs."""
import argparse
import json
from pathlib import Path

import numpy as np

from sttw_control.finite_value_evaluation_report import evaluation_inputs
from sttw_control.preference_best import digest
from sttw_control.feedback_timing_audit import force_window,cost_partition,angle_delta,describe

PAIRS=[('feedback_010','free_GAE'),('parent_feedback_010','parent_causal')]
FIELDS=['active_tick','physical_tick','actual_xy','actual_yaw','actual_forward_speed','phi','phi_dot',
        'actual_delta','reference','applied_force_impulse','scored_tick_reward','feedback_requested_steer_rad']


def audit(root,output):
    root=root.resolve()
    if output.exists():raise ValueError('preserve prior derived audit')
    for name in ['status.json','analysis/status.json']:
        if json.loads((root/name).read_text())['phase']!='completed':raise ValueError('completed raw analysis required')
    plan=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('changed sealed plan')
    rows=json.loads((root/'analysis/results.json').read_text())
    _,_,indices,_=evaluation_inputs(root,plan);index={r['case_id']:r for r in indices}
    selected={role for pair in PAIRS for role in pair};raws={};loaded=[]
    for row in rows:
        if row['role'] not in selected:continue
        if row['physical_outcome']=='unknown' or row['goal_arrived'] is None:
            raise ValueError('unknown outcomes need explicit separate analysis')
        item=index[row['case_id']];file=root/item['file']
        if digest(file)!=item['sha256'] or item['sha256']!=row['raw_sha256']:raise ValueError('raw identity changed')
        with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in FIELDS}
        n=int(raw['active_tick'].sum())
        if not np.array_equal(raw['physical_tick'][raw['active_tick']],np.arange(n)):
            raise ValueError('non-prefix physical clock')
        raw={k:v[raw['active_tick']] for k,v in raw.items()}
        np.testing.assert_allclose(-raw['scored_tick_reward'].sum(),row['scored_cost'],rtol=2e-5,atol=2e-5)
        raws[row['case_id']]=raw;loaded.append(row)
    cells={};zeros={}
    for row in loaded:
        if row['force_N']:
            cells.setdefault((row['context'],row['alpha'],row['event_id']),{})[row['role']]=row
        else:zeros[row['context'],row['alpha'],row['role']]=row
    records=[];unexposed=[]
    for cell in cells.values():
        if set(cell)!=selected:raise ValueError('complete four-role event required')
        for target,baseline in PAIRS:
            a,b=cell[target],cell[baseline]
            for key in ['force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s']:
                if a[key]!=b[key]:raise ValueError('mismatched real event')
            da,db=raws[a['case_id']],raws[b['case_id']]
            wa,wb=[force_window(x['active_tick'],x['applied_force_impulse']) for x in [da,db]]
            if wa is None or wb is None:
                unexposed.append(dict(target=a['case_id'],baseline=b['case_id'],observed_windows=[wa,wb]));continue
            if wa[0]!=wb[0] or wa[0]==0:raise ValueError('common measured pre-force prefix required')
            start=wa[0]
            za=zeros[a['context'],a['alpha'],target];zb=zeros[b['context'],b['alpha'],baseline]
            na,nb=raws[za['case_id']],raws[zb['case_id']]
            if len(na['physical_tick'])!=2000 or len(nb['physical_tick'])!=2000:
                raise ValueError('complete unfiltered nominal counterpart required')
            if force_window(na['active_tick'],na['applied_force_impulse']) or force_window(nb['active_tick'],nb['applied_force_impulse']):
                raise ValueError('nominal counterpart contains force')
            costs=[cost_partition(-x['scored_tick_reward'],start) for x in [da,db,na,nb]]
            delta={key:costs[0][key]-costs[1][key] for key in ['before','after','total']}
            nominal={key:costs[2][key]-costs[3][key] for key in ['before','after','total']}
            prefix={}
            for tag,x,z in [('feedback',da,na),('baseline',db,nb)]:
                prefix[tag]={key:float(np.max(np.abs(angle_delta(x[key][:start],z[key][:start])
                    if key=='actual_yaw' else x[key][:start]-z[key][:start])))
                    for key in ['actual_xy','actual_yaw','actual_forward_speed','phi','phi_dot','reference']}
            request=np.abs(da['feedback_requested_steer_rad']);dt=plan['lower_dt_s']
            i=start-1
            records.append(dict(comparison=target+'_vs_'+baseline,case_id=a['case_id'],context=a['context'],alpha=a['alpha'],
                force_N=a['force_N'],observed_force_first_tick=start,scheduled_force_tick=a['pulse_onset_task_tick'],
                observed_minus_scheduled_ticks=start-a['pulse_onset_task_tick'],
                pre_force_s=start*dt,feedback_nominal_prefix_difference=prefix,
                entry_xy_separation_m=float(np.linalg.norm(da['actual_xy'][i]-db['actual_xy'][i])),
                entry_yaw_delta_rad=float(angle_delta(da['actual_yaw'][i],db['actual_yaw'][i])),
                entry_speed_delta_m_s=float(da['actual_forward_speed'][i]-db['actual_forward_speed'][i]),
                entry_roll_delta_rad=float(da['phi'][i]-db['phi'][i]),
                entry_roll_rate_delta_rad_s=float(da['phi_dot'][i]-db['phi_dot'][i]),
                pre_force_mean_abs_requested_steer_rad=float(request[:start].mean()),
                pre_force_request_integral_rad_s=float(request[:start].sum()*dt),
                full_request_integral_rad_s=float(request.sum()*dt),
                disturbed_cost_delta=delta,nominal_cost_delta=nominal,
                force_associated_excess_cost_delta=delta['total']-nominal['total'],
                both_survived=a['physical_outcome']==b['physical_outcome']=='survived',
                arrival_gain=bool(a['goal_arrived'] and not b['goal_arrived']),
                arrival_loss=bool(b['goal_arrived'] and not a['goal_arrived']),
                physics_outcomes=[a['physical_outcome'],b['physical_outcome']]))
    summary={}
    for target,baseline in PAIRS:
        name=target+'_vs_'+baseline;rr=[r for r in records if r['comparison']==name];ss=[r for r in rr if r['both_survived']]
        result=dict(events=len(rr),both_survived_events=len(ss),arrival_gains=sum(r['arrival_gain'] for r in rr),
                    arrival_losses=sum(r['arrival_loss'] for r in rr))
        for key in ['entry_xy_separation_m','entry_yaw_delta_rad','entry_speed_delta_m_s','entry_roll_delta_rad',
                    'entry_roll_rate_delta_rad_s','pre_force_mean_abs_requested_steer_rad']:
            result[key]=describe([r[key] for r in rr])
        result['pre_force_request_fraction_of_observed_integral']=sum(r['pre_force_request_integral_rad_s'] for r in rr)/sum(r['full_request_integral_rad_s'] for r in rr)
        for key in ['disturbed_cost_delta','nominal_cost_delta']:
            result[key]={phase:describe([r[key][phase] for r in ss]) for phase in ['before','after','total']}
        result['force_associated_excess_cost_delta_both_survived']=describe([r['force_associated_excess_cost_delta'] for r in ss])
        result['prefix_replay_max_abs_difference']={key:max(r['feedback_nominal_prefix_difference'][tag][key] for r in rr for tag in ['feedback','baseline'])
            for key in ['actual_xy','actual_yaw','actual_forward_speed','phi','phi_dot','reference']}
        result['observed_minus_scheduled_tick_values']=sorted({r['observed_minus_scheduled_ticks'] for r in rr})
        summary[name]=result
    result=dict(summary=summary,records=records,unexposed_cases=unexposed,authenticated_raw_count=len(loaded),
        results_sha256=digest(root/'analysis/results.json'),plan_sha256=digest(root/'plan.json'),
        raw_sources=[dict(case_id=r['case_id'],sha256=r['raw_sha256']) for r in loaded],
        script_sha256=digest(__file__),module_sha256=digest(Path(__file__).parents[1]/'src/sttw_control/feedback_timing_audit.py'),
        physics_steps=0,optimizer_updates=0,
        scope='Exploratory timing/entry-state diagnostic. Actual first nonzero logged force impulse defines the boundary. Future force timing is used only offline, never supplied to a controller. Pre-force nominal replay differences are reported, not used as a new exact-replay gate.',
        limitations='Different closed-loop prefix states confound after-force policy attribution. Force-associated excess subtracts each policy own zero trajectory, not a causal prefix/suffix intervention. Costs summarized on both-survived cases; all physical/arrival outcomes remain separate. Reused nominal traces and force branches are not independent replicates.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.run,args.output);print(json.dumps(result['summary'],indent=2))
