"""Read-only geometric-stage decomposition of the frozen native32 panel."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    plan=json.loads((source/'plan.json').read_text());old=json.loads((source/'analysis/review.json').read_text())
    assert len(old['records'])==32
    out.mkdir(exist_ok=False)
    cfg=plan['config'];c=cfg['path'];prefix=c['straight_prefix_m'];ramp=c['curvature_transition_length_m']
    turn_end=prefix+abs(c['turn_angle_rad'])*c['turn_radius_m']+ramp
    boundaries=[prefix,prefix+ramp,turn_end-ramp,turn_end]
    phases=['straight_prefix','entry_ramp','constant_curvature','exit_ramp','outgoing_straight']
    design=dict(question='Does the existing H75 working-roll advantage occur only at geometric transitions, or also with sustained reference differences?',
        data_role='Posthoc DEV analysis after global native rewards/goal-prefix components were already seen',
        boundaries_m=boundaries,phases=phases,windows=['before_first_goal1m_or_observed_end','after_first_goal1m'],
        source_review_sha256=digest(source/'analysis/review.json'),source_plan_sha256=digest(source/'plan.json'),
        metric_policy='No new weights, filtering, fitted model, interpolation or peak-selected windows. Mean rates and durations separate from integrals; no causal attribution.',
        stop='Exactly existing32; no new simulation/training/library or automatic matched-mean intervention',new_physics_steps=0)
    (out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    paths={}
    for route in {x['route'] for x in old['records']}:
        file=source/'evaluation'/(route+'_path.npz')
        with np.load(file) as z:
            # This older path export has xy/s/goal only, not turn_end.
            np.testing.assert_allclose(float(z['goal'])-c['goal_section_offset_after_turn_m'],turn_end,atol=1e-5,rtol=0)
        paths[str(file)]=digest(file)
    records=[];traces={};raw_identity={}
    for item in old['records']:
        file=Path(item['raw_file']);assert digest(file)==item['raw_sha256'];raw_identity[str(file)]=item['raw_sha256']
        with np.load(file) as z:d={k:z[k] for k in z.files}
        active=d['active_tick'].astype(bool);n_all=len(active);assert active.all() and 0<n_all<=2000
        assert n_all==item['active_lower_ticks']
        np.testing.assert_allclose(d['time'],np.arange(n_all)*.005,atol=1e-5,rtol=0)
        assert not (d['physical_failure']|d['lower_fault']|d['policy_fault']).any()
        progress=d['path_progress'];assert np.isfinite(progress).all() and np.all(np.diff(progress)>=-1e-6)
        assert d['effective_costs'].shape[1]==10
        bins=np.searchsorted(boundaries,progress,side='right')
        cut=int(round(item['first_arrival_s']/.005)) if item['first_arrival_s'] is not None else len(active)
        rowkey=(item['route'],item['speed'],item['alpha'],item['family']);traces[rowkey]=d
        ref_change=np.abs(np.diff(d['governed'],axis=0,prepend=d['governed'][:1]))
        for window,wmask in [('before_first_goal1m_or_observed_end',np.arange(n_all)<cut),('after_first_goal1m',np.arange(n_all)>=cut)]:
            for i,phase in enumerate(phases):
                mask=wmask&(bins==i);n=int(mask.sum())
                row=dict(route=item['route'],speed=item['speed'],alpha=item['alpha'],family=item['family'],phase=phase,window=window,
                    ticks=n,duration_s=n*.005,goal1m_arrived=item['goal1m_before_native_stop'],observed_end_s=n_all*.005,
                    domain_stop=item['native_domain_failure'],
                    substep_peak_roll_rad=float(d['peak_roll'][mask].max()) if n else None,
                    reference_speed_TV_m_s=float(ref_change[mask,0].sum()),reference_steer_TV_rad=float(ref_change[mask,1].sum()))
                quantities={'abs_roll_rad':abs(d['phi']),'abs_roll_rate_rad_s':abs(d['phi_dot']),
                    'speed_request_error_m_s':d['actual_forward_speed']-item['speed'],
                    'path_error_m':d['path_cross_track'],'actual_speed_m_s':d['actual_forward_speed'],
                    'actual_steer_rad':d['actual_delta'],'reference_speed_m_s':d['governed'][:,0],
                    'reference_steer_rad':d['governed'][:,1]}
                for name,values in quantities.items():
                    row['mean_'+name]=float(values[mask].mean()) if n else None
                    row['rms_'+name]=float(np.sqrt(np.mean(values[mask]**2))) if n else None
                    row['max_abs_'+name]=float(abs(values[mask]).max()) if n else None
                for name,k in [('roll',3),('working_roll',4)]:
                    values=d['effective_costs'][:,k]
                    row[name+'_mean_cost_rate']=float(values[mask].mean()) if n else None
                    row[name+'_cost_integral']=float(values[mask].sum(dtype=np.float64)*.005*cfg['path_reward']['scale'])
                records.append(row)
    assert len(records)==320
    lookup={(x['route'],x['speed'],x['alpha'],x['family'],x['phase'],x['window']):x for x in records}
    paired=[]
    for x in records:
        if x['family']!='H75':continue
        for baseline in ('zero','equilibrium'):
            b=lookup[x['route'],x['speed'],x['alpha'],baseline,x['phase'],x['window']]
            row={k:x[k] for k in ('route','speed','alpha','phase','window')};row.update(baseline=baseline,H75_duration_s=x['duration_s'],baseline_duration_s=b['duration_s'])
            for name in ('mean_abs_roll_rad','max_abs_abs_roll_rad','mean_abs_roll_rate_rad_s','mean_reference_speed_m_s','mean_reference_steer_rad','rms_path_error_m','rms_speed_request_error_m_s','roll_mean_cost_rate','working_roll_mean_cost_rate','working_roll_cost_integral'):
                row['delta_'+name]=None if x[name] is None or b[name] is None else x[name]-b[name]
            paired.append(row)
    # Keep each context/alpha comparison visible; no correlated rows treated as
    # independent repetitions and no combined new scalar score.
    summary=[]
    for baseline in ('zero','equilibrium'):
        for phase in phases:
            xs=[x for x in paired if x['baseline']==baseline and x['phase']==phase and x['window']=='before_first_goal1m_or_observed_end']
            usable=[x for x in xs if x['delta_mean_abs_roll_rad'] is not None]
            summary.append(dict(baseline=baseline,phase=phase,n_related_pairs=len(usable),
                lower_mean_abs_roll=sum(x['delta_mean_abs_roll_rad']<0 for x in usable),
                mean_abs_roll_delta_range=[min(x['delta_mean_abs_roll_rad'] for x in usable),max(x['delta_mean_abs_roll_rad'] for x in usable)] if usable else None,
                reference_speed_delta_range=[min(x['delta_mean_reference_speed_m_s'] for x in usable),max(x['delta_mean_reference_speed_m_s'] for x in usable)] if usable else None))
    for name,rows in [('records.csv',records),('pairs.csv',paired)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(design=design,records=records,pairs=paired,stage_summary=summary,input_raw_sha256=raw_identity,input_path_sha256=paths,
        script_sha256=digest(__file__),new_physics_steps=0,optimizer_updates=0,
        limits='Stage durations and reference levels differ endogenously. Follower monotone projection can diverge from physical path distance. First-goal prefixes differ in duration; never-arrived rows are observed failures. Missing stages stay null. Zero2 domain stops at9.78s leave unknown tail, not physical failure. Mean post-tick roll and substep peak roll are distinct; neither is disturbance tolerance. TV includes change from preceding logged tick at phase entry; initial governor-to-first-log excluded. No dynamic compensation causality established.')
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
