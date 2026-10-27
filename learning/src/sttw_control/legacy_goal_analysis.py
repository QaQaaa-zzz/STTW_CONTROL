"""Read-only outcomes and full-panel figures for the fixed legacy90 run."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    a=parser.parse_args();root=a.root.resolve();source=root/'evaluation';out=root/'analysis'
    status=json.loads((source/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=90:
        raise ValueError('Require completed fixed90; never launch or restart collection here')
    index=json.loads((source/'case_index.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    preparation=json.loads((root/'plan.json').read_text())
    assert manifest['horizon_s']==20 and manifest['lower_dt_s']==.005 and len(index)==90
    assert manifest['plan_sha256']==sha(root/'plan.json')
    seal=json.loads((root/'execution_seal.json').read_text())
    assert manifest['execution_plan_sha256']==seal['execution_plan_sha256']==sha(root/'execution_plan.json')
    inventory=json.loads((root/'inventory.json').read_text())
    assert manifest['cases']==inventory
    expected_cases={x['case_id']:x for x in inventory}
    assert len({x['case_id'] for x in index})==90
    assert {x['case_id'] for x in index}==set(expected_cases)
    assert all(x['case']==expected_cases[x['case_id']] for x in index)
    paths={}
    for route in ('left','right'):
        assert manifest['config']==json.loads(Path(preparation['routes'][route]['config']).read_text())
        with np.load(source/(route+'_original_path.npz')) as z:paths[route]={k:z[k] for k in z.files}
        assert sha(preparation['routes'][route]['path'])==preparation['routes'][route]['path_sha256']
        with np.load(preparation['routes'][route]['path']) as z:
            for k in z.files:np.testing.assert_array_equal(paths[route][k],z[k])
    initial={x['case_id']:x['physical_xy'] for x in json.loads((root/'input_audit.json').read_text())['records']}
    out.mkdir(exist_ok=False)
    from sttw_control.rejoin_comparison import geometric_task_trace
    records=[];traces={};inputs={}
    for item in index:
        case=item['case'];cid=item['case_id'];file=Path(item['raw_file'])
        assert sha(file)==item['raw_sha256'];inputs[str(file)]=item['raw_sha256']
        with np.load(file,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        active=d['active_tick'].astype(bool);indices=np.flatnonzero(active);ticks=d['physical_tick'][active]
        assert len(active)==4000 and len(indices)>0 and np.array_equal(indices,np.arange(len(indices)))
        np.testing.assert_array_equal(ticks,np.arange(len(ticks)))
        assert int(active.sum())==item['active_lower_ticks']
        np.testing.assert_allclose(d['time'][active],ticks*.005,atol=1e-5,rtol=0)
        xy=d['actual_xy'].astype(float);goal=np.asarray(item['original_goal_xy'])
        p=paths[case['route_group']]
        np.testing.assert_allclose(goal,[np.interp(float(p['goal']),p['s'],p['xy'][:,k]) for k in (0,1)],atol=1e-6)
        contact=d['contact_failure'].astype(bool)&active
        fault=(d['nonfinite']|d['lower_fault']|d['policy_fault']).astype(bool)&active
        fault|=active&~np.isfinite(xy).all(axis=1)
        errors=np.flatnonzero(fault)
        if len(errors):raise ValueError('Completed run contains engineering UNKNOWN '+cid)
        fatal=np.flatnonzero(contact|fault);stop=int(fatal[0]) if len(fatal) else len(indices)
        assert not contact.any() or np.flatnonzero(contact)[0]==len(indices)-1
        distance=np.linalg.norm(xy-goal,axis=1);valid=active&(np.arange(4000)<stop)
        physical='contact' if contact.any() else 'survived' if len(indices)==4000 else 'unknown'
        if physical=='unknown':raise ValueError('Unexplained incomplete case '+cid)
        # The pulse is keyed to task time. Integrator logging sums substep
        # impulses over each actual 5ms control interval, including contact tick.
        t=ticks*.005;onset=case['pulse_onset_task_tick']*.005
        exposure=float(np.maximum(0,np.minimum(t+.005,onset+case['pulse_duration_s'])-np.maximum(t,onset)).sum())
        expected=case['force_N']*exposure*np.asarray(case['world_direction'])
        actual_impulse=np.asarray(d['applied_force_impulse'][active]).sum(axis=0)
        # Keep the previously accepted 1% substep-boundary tolerance. This is
        # an engineering exposure check, not a new numerical replay gate.
        np.testing.assert_allclose(actual_impulse,expected,atol=1e-5,rtol=.01,err_msg=cid)
        tick_exposure=np.maximum(0,np.minimum(t+.005,onset+case['pulse_duration_s'])-np.maximum(t,onset))
        expected_profile=tick_exposure[:,None]*case['force_N']*np.asarray(case['world_direction'])
        impulse_profile_max_error=float(np.linalg.norm(d['applied_force_impulse'][active]-expected_profile,axis=1).max())
        ref=d['reference'][active]
        assert np.isfinite(ref).all()
        assert ref[:,0].min()>=case['reference_speed_floor_m_s']-2e-5 and ref[:,0].max()<=3.5+2e-5
        assert abs(ref[:,1]).max()<=.65+2e-5
        geometry=geometric_task_trace(d,p,manifest['config'],initial_xy=initial[cid])
        traveled=float(np.linalg.norm(np.diff(np.vstack([initial[cid],xy[active]]),axis=0),axis=1).sum())
        traces[cid]=d
        for horizon in (10.,20.):
            window=active&(np.arange(4000)<round(horizon/.005));upright=valid&window
            physical_h='contact' if np.any(contact&window) else 'survived' if window.sum()==round(horizon/.005) else 'unknown'
            for radius in (.5,.75,1.):
                hits=np.flatnonzero(upright&(distance<=radius));arrived=bool(len(hits))
                record=dict(case_id=cid,route=case['route_group'],role=case['role'],alpha=case['alpha'],event=case['event_id'],
                    force_N=case['force_N'],disturbed=bool(case['force_N']),horizon_s=horizon,radius_m=radius,
                    arrived=arrived,physical=physical_h,arrived_and_survived=arrived and physical_h=='survived',
                    arrival_time_s=float((hits[0]+1)*.005) if len(hits) else None,
                    minimum_prefatal_distance_m=float(distance[upright].min()) if upright.any() else None,
                    final_observed_distance_m=float(distance[np.flatnonzero(window)[-1]]),
                    active_lower_ticks=int(window.sum()),
                    original_route_max_abs_cross_track_m=float(abs(geometry['path_cross_track'][window]).max()),
                    executed_route_max_abs_cross_track_m=float(abs(d['path_cross_track'][window]).max()),
                    speed_request_rmse_m_s=float(np.sqrt(np.mean((d['actual_forward_speed'][window]-case['requested_speed_m_s'])**2))),
                    traveled_distance_full_observed_m=traveled,
                    reference_speed_TV_between_logged_ticks_full_observed_m_s=float(abs(np.diff(ref[:,0])).sum()),
                    reference_steer_TV_between_logged_ticks_full_observed_rad=float(abs(np.diff(ref[:,1])).sum()),
                    pulse_exposure_full_observed_s=exposure,pulse_profile_max_tick_error_N_s=impulse_profile_max_error,
                    raw_sha256=item['raw_sha256'])
                records.append(record)
    assert status['active_lower_ticks']==sum(x['active_lower_ticks'] for x in index)
    assert status['allocated_lower_slots']==360000
    groups=defaultdict(list)
    for r in records:groups[r['role'],r['disturbed'],r['horizon_s'],r['radius_m']].append(r)
    summaries=[]
    for (role,disturbed,horizon,radius),xs in sorted(groups.items()):
        summaries.append(dict(role=role,disturbed=disturbed,horizon_s=horizon,radius_m=radius,n=len(xs),
            arrived=sum(r['arrived'] for r in xs),survived=sum(r['physical']=='survived' for r in xs),
            contact=sum(r['physical']=='contact' for r in xs),unknown=sum(r['physical']=='unknown' for r in xs),
            arrived_and_survived=sum(r['arrived_and_survived'] for r in xs)))
    lookup={(r['route'],r['alpha'],r['event'],r['role'],r['horizon_s'],r['radius_m']):r for r in records}
    pairs=[]
    for r in records:
        if r['role']!='physical_credit':continue
        for role in ('task_only','parent','floor','zero_middle'):
            base=lookup[r['route'],None if role=='zero_middle' else r['alpha'],r['event'],role,r['horizon_s'],r['radius_m']]
            pairs.append(dict(target_case_id=r['case_id'],baseline_case_id=base['case_id'],baseline=role,
                disturbed=r['disturbed'],horizon_s=r['horizon_s'],radius_m=r['radius_m'],
                arrival_delta=int(r['arrived'])-int(base['arrived']),
                survival_delta=int(r['physical']=='survived')-int(base['physical']=='survived'),
                joint_delta=int(r['arrived_and_survived'])-int(base['arrived_and_survived']),
                zero_reuse_dependency=role=='zero_middle'))
    pg=defaultdict(list)
    for r in pairs:pg[r['baseline'],r['disturbed'],r['horizon_s'],r['radius_m']].append(r)
    paired=[]
    for (base,disturbed,horizon,radius),xs in sorted(pg.items()):
        paired.append(dict(baseline=base,disturbed=disturbed,horizon_s=horizon,radius_m=radius,n_logical_pairs=len(xs),
            arrival_gain=sum(r['arrival_delta']==1 for r in xs),arrival_loss=sum(r['arrival_delta']==-1 for r in xs),
            survival_gain=sum(r['survival_delta']==1 for r in xs),survival_loss=sum(r['survival_delta']==-1 for r in xs),
            joint_gain=sum(r['joint_delta']==1 for r in xs),joint_loss=sum(r['joint_delta']==-1 for r in xs),
            zero_reuse_dependency=base=='zero_middle'))
    table(out/'records.csv',records);table(out/'summaries.csv',summaries);table(out/'pairs.csv',pairs)
    write(out/'review.json',dict(summaries=summaries,paired_summaries=paired,records=records,pairs=pairs,
        input_raw_sha256=inputs,new_physics_steps=0,optimizer_updates=0,analysis_script_sha256=sha(__file__),
        source_manifest_sha256=sha(source/'manifest.json'),limitations='Seen DEV families, one old training pair, shared ancestors and explicit zero reuse; no independent test/repeats or scalar Jalpha. Full-observed distance/variation diagnostics are not window-normalized. Reference TV excludes initial governor-to-first-log transition. Original route uses causal projection; executed route switches geometry at replan. Pulse total tolerance1%, per-tick residual diagnostic only.'))
    figures=plot(out,index,traces,paths)
    write(out/'figure_manifest.json',dict(figures=figures,source_data='Existing raw case_index, records.csv and original paths; replanned geometry is not drawn',
        encoding='All routes/alphas/events/roles, equal XY scale per event; no smoothing. Contact x, surviving endpoint o, goal circle1m; zero physically shared explicitly.',
        destination='Internal DEV report, provisional dimensions; not journal submission certification'))
    write(out/'status.json',dict(phase='completed',unique_cases=90,records=len(records),figures=len(figures),new_physics_steps=0))
    print(json.dumps([r for r in summaries if r['horizon_s']==20 and r['radius_m']==1],indent=2))


def plot(out,index,traces,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    colors={'parent':'#0072B2','task_only':'#D55E00','physical_credit':'#009E73','floor':'#882255','zero_middle':'#444444'}
    styles={'parent':'--','task_only':'-.','physical_credit':'-','floor':':','zero_middle':(0,(4,1,1,1))}
    figures=[]
    for route in ('left','right'):
        p=paths[route];goal=np.array([np.interp(float(p['goal']),p['s'],p['xy'][:,k]) for k in (0,1)])
        for event in sorted({x['case']['event_id'] for x in index}):
            selected=[x for x in index if x['case']['route_group']==route and x['case']['event_id']==event]
            fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
            cloud=np.vstack([traces[x['case_id']]['actual_xy'][traces[x['case_id']]['active_tick']] for x in selected]+[goal[None]])
            lo=cloud.min(axis=0)-1.1;hi=cloud.max(axis=0)+1.1
            for ax,alpha in zip(axes,(.5,.75)):
                valid=p['s']<=float(p['goal']);ax.plot(*p['xy'][valid].T,color='.65',linestyle=':',label='Original path')
                ax.add_patch(Circle(goal,1.,fill=False,color='black',linewidth=1));ax.plot(*goal,'k*',markersize=8)
                for item in selected:
                    c=item['case']
                    if c['alpha'] not in (alpha,None):continue
                    d=traces[item['case_id']];active=d['active_tick'];xy=d['actual_xy'][active];role=c['role']
                    ax.plot(*xy.T,color=colors[role],linestyle=styles[role],label=role+(' (shared)' if c['alpha'] is None else ''))
                    ax.scatter(*xy[-1],color=colors[role],marker='x' if d['contact_failure'][active].any() else 'o',s=25)
                    on=c['pulse_onset_task_tick']*.005;t=d['physical_tick'][active]*.005;mask=(t>=on)&(t<on+c['pulse_duration_s'])
                    if c['force_N'] and mask.any():ax.plot(*xy[mask].T,color=colors[role],linewidth=4)
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)',title=f'{route}/{event}, user alpha={alpha:g}')
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle('Legacy frozen actors, common20s + one causal replan; dot=survived20s, x=contact; thick=pulse')
            name=f'xy_{route}_{event}.png';file=out/name;fig.savefig(file,dpi=130,facecolor='white');plt.close(fig)
            with Image.open(file) as im:figures.append(dict(file=name,pixels=list(im.size),dpi=list(im.info['dpi']),mode=im.mode))
    return figures


if __name__=='__main__':main()
