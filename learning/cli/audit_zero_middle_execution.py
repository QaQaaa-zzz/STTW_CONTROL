"""Verify actual zero-source execution and describe correlated alpha branches."""
import argparse
import json
from pathlib import Path
import numpy as np

from sttw_control.preference_best import digest

SIGNALS=['actual_xy','actual_yaw','actual_forward_speed','phi','phi_dot','actual_delta','reference','applied_force_impulse']


def compare_alpha_traces(traces):
    """Describe observed differences; do not impose an exact replay gate."""
    lengths=[len(t['actual_xy']) for t in traces]
    common=min(lengths)
    return dict(observed_lengths=lengths,common_ticks=common,
                max_abs_difference={key:max(float(np.max(np.abs(t[key][:common]-traces[0][key][:common])))
                    for t in traces) for key in SIGNALS} if common else {})


def audit(root,output):
    if output.exists():raise ValueError('keep existing audit immutable')
    root=root.resolve()
    if json.loads((root/'analysis/status.json').read_text())['phase']!='completed':
        raise ValueError('complete authenticated analysis required')
    rows=json.loads((root/'analysis/results.json').read_text())
    selected=[r for r in rows if r.get('actor_source_zero')]
    if len(selected)!=120:raise ValueError('complete120 zero-source trajectories required')
    groups={};legal_total=native_nonzero=0;max_source=0.
    for row in selected:
        file=Path(row['raw_file'])
        if digest(file)!=row['raw_sha256']:raise ValueError('changed zero-source raw')
        with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        legal=raw['actor_action_valid'].astype(bool)
        np.testing.assert_array_equal(raw['actor_source_zero'],True)
        np.testing.assert_array_equal(raw['source_actor_latent'][legal],0.)
        legal_total+=int(legal.sum())
        native_nonzero+=int(np.any(raw['original_actor_latent'][legal]!=0,axis=1).sum())
        if legal.any():max_source=max(max_source,float(np.max(np.abs(raw['source_actor_latent'][legal]))))
        active=raw['active_tick'].astype(bool)
        key=(row['role'],row['context'],row['event_id'])
        groups.setdefault(key,[]).append((row,{k:raw[k][active] for k in SIGNALS}))
    records=[]
    for (role,context,event),items in groups.items():
        items.sort(key=lambda item:item[0]['alpha'])
        if [row['alpha'] for row,_ in items]!=[0.,.5,1.]:raise ValueError('three original alpha branches required')
        for field in ['force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s']:
            if any(row[field]!=items[0][0][field] for row,_ in items):raise ValueError('mismatched alpha event')
        records.append(dict(role=role,context=context,event_id=event,alphas=[row['alpha'] for row,_ in items],
            physical_outcomes=[row['physical_outcome'] for row,_ in items],arrivals=[row['goal_arrived'] for row,_ in items],
            original_alpha_costs=[row['scored_cost'] for row,_ in items],**compare_alpha_traces([trace for _,trace in items])))
    result=dict(authenticated_zero_raws=len(selected),legal_ticks=legal_total,native_nonzero_ticks=native_nonzero,
        max_abs_executed_source_latent=max_source,alpha_groups=records,
        max_alpha_trace_difference={key:max(r['max_abs_difference'][key] for r in records if r['common_ticks']) for key in SIGNALS},
        source=dict(results_sha256=digest(root/'analysis/results.json'),script_sha256=digest(__file__)),
        physics_steps=0,optimizer_updates=0,
        scope='Zero neural middle retains frozen learned lower/follower/short support. Native NN output is logged but does not drive the selected source. Alpha-branch equality is descriptive and not a replay gate or independent repetition claim.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.run,args.output)
    print(json.dumps({k:v for k,v in r.items() if k!='alpha_groups'},indent=2))
