"""Audit the fixed six nominal-program suffixes and eighteen cached controls."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from sttw_control.preference_history_task_review import paired_complete_metrics, shadow_cross_track


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();plan=json.loads((root/'plan.json').read_text());status=json.loads((root/'status.json').read_text())
    assert status['phase']=='completed' and status['completed_cases']==6
    out=root/'analysis';out.mkdir(exist_ok=False)
    physical=json.loads((root/'results.json').read_text())['cases'];assert len(physical)==6
    source=Path(plan['source_run']);task=json.loads((source.parent/'preference_history_task_review_v1/review.json').read_text())
    declaration=json.loads(Path(plan['source_declaration']).read_text());cfg=declaration['config']
    from sttw_control.geometric_path import build_path
    paths={};origins={}
    for route in declaration['routes']:
        with np.load(task['path_reconstruction'][route['name']]['original_xy_file']) as z:xy=z['xy'].astype(float)
        heading=float(np.arctan2(*(xy[1]-xy[0])[::-1]))
        built=build_path(route['builder'],[*xy[0],heading],cfg)
        paths[route['name']]=dict(xy=xy,s=np.asarray(built.s),heading=np.asarray(built.heading))
    for row in plan['cases']:
        if row['state_id'] not in origins:
            state=pickle.loads(Path(row['snapshot']).read_bytes())
            origins[row['state_id']]=dict(xy=np.asarray(state.physical.data.qpos)[:2].astype(float),progress=float(state.projection.progress))
    records=[];traces={};program_receipts=[]
    inventory=[dict(r,role='nominal_program',raw_file=str(root/(r['name']+'.npz'))) for r in physical]
    inventory+=plan['cached_cases']
    assert len(inventory)==24
    for row in inventory:
        file=Path(row['raw_file']);sha=digest(file)
        if 'raw_sha256' in row:assert sha==row['raw_sha256']
        with np.load(file) as z:log={k:z[k] for k in z.files}
        active=log['active_tick'].astype(bool);assert np.array_equal(active,np.arange(len(active))<sum(active))
        t=log['time'][active].astype(float);xy=log['actual_xy'][active].astype(float)
        assert np.allclose(t,np.arange(len(t))*.005,atol=3e-6,rtol=0)
        fatal=np.logical_or.reduce([log[k][active].astype(bool) for k in ['contact_failure','nonfinite','lower_fault','policy_fault']])
        end=int(np.flatnonzero(fatal)[0]) if fatal.any() else len(t)
        unknown=bool(row.get('final_fault',False)) or bool(np.logical_or.reduce([log[k][active].astype(bool) for k in ['nonfinite','lower_fault','policy_fault']]).any())
        label='UNKNOWN' if unknown else 'CONTACT' if log['contact_failure'][active].any() else 'PASS' if len(t)==800 else 'UNKNOWN'
        if 'label' in row:assert label==row['label']
        goal=np.array(task['path_reconstruction'][row['route']]['original_goal_world_xy_m'])
        origin=origins[row['state_id']]
        ey,_=shadow_cross_track(xy,paths[row['route']],cfg,initial_xy=origin['xy'],initial_progress=origin['progress'])
        distance=np.linalg.norm(xy[:end]-goal,axis=1);hit=np.flatnonzero(distance<=1.)
        role=row['role'];key=(row['state_id'],row['force_N'],role)
        traces[key]=(xy,goal,label)
        records.append(dict(state_id=row['state_id'],force_N=row['force_N'],role=role,physical_label=label,
            raw_file=str(file),raw_sha256=sha,observed_lower_ticks=len(t),
            goal_hit_before_fatal=bool(len(hit)),first_suffix_goal_s=float(t[hit[0]]) if len(hit) else None,
            minimum_goal_distance_m=float(distance.min()) if len(distance) else None,
            path_shadow_rmse_m=float(np.sqrt(np.mean(ey**2))),
            original_speed_mae_m_s=float(np.mean(abs(log['actual_forward_speed'][active].astype(float)-1.5)))))
        if role=='nominal_program':
            with np.load(row['nominal_program_file']) as z:program=z['bare_latents']
            assert digest(row['nominal_program_file'])==row['nominal_program_sha256']
            used=log['nominal_program_used'][active].astype(bool);assert np.array_equal(used,log['upper_active'][active].astype(bool))
            indices=np.rint(t[used]/.005).astype(int)//4;assert ((indices>=0)&(indices<200)).all()
            expected=program[indices]
            np.testing.assert_array_equal(log['nominal_program_latent'][active][used],expected)
            np.testing.assert_array_equal(log['executed_policy_latent'][active][used],expected)
            shadow=log['original_policy_latent'][active][used];finite=np.isfinite(shadow).all(axis=1)
            delta=np.max(abs(shadow[finite]-expected[finite])) if finite.any() else None
            impulse=log['applied_force_impulse'][active].sum(axis=0)
            expected_impulse=row['force_N']*.1*np.asarray(row['world_direction'])
            error=float(np.linalg.norm(impulse-expected_impulse))
            assert error<=.01*max(1,abs(row['force_N'])*.1)
            program_receipts.append(dict(state_id=row['state_id'],force_N=row['force_N'],used_lower_ticks=int(used.sum()),
                held_lower_ticks=int(log['fallback_active'][active].sum()),shadow_finite_ticks=int(finite.sum()),
                max_shadow_program_difference= float(delta) if delta is not None else None,
                issued_program_exact_match=True,force_impulse_error_Ns=error,
                note='Shadow nonfinite comparisons excluded, not physical labels; unused program zeros not treated as publications'))
    roles=['future0','nominal_program','zero_latent','speed_support']
    summary={role:dict(n=6,survived=sum(r['physical_label']=='PASS' for r in records if r['role']==role),
        contact=sum(r['physical_label']=='CONTACT' for r in records if r['role']==role),
        unknown=sum(r['physical_label']=='UNKNOWN' for r in records if r['role']==role),
        goal_hits=sum(r['goal_hit_before_fatal'] for r in records if r['role']==role)) for role in roles}
    index={(r['state_id'],r['force_N'],r['role']):r for r in records};paired=[]
    for state,force in sorted({(r['state_id'],r['force_N']) for r in records}):
        live=index[state,force,'future0'];program=index[state,force,'nominal_program']
        paired.append(dict(state_id=state,force_N=force,live=live['physical_label'],program=program['physical_label'],
            live_goal=live['goal_hit_before_fatal'],program_goal=program['goal_hit_before_fatal']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors=['#0072B2','#D55E00','#009E73','#CC79A7'];styles=['-','--','-.',':']
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for i,state in enumerate(sorted({r['state_id'] for r in records})):
        route=next(r['route'] for r in inventory if r['state_id']==state)
        with np.load(task['path_reconstruction'][route]['original_xy_file']) as z:path=z['xy']
        for j,force in enumerate([0.,-32.,32.]):
            ax=axes[i,j];points=[]
            for role,color,style in zip(roles,colors,styles):
                xy,goal,label=traces[state,force,role];points.extend(xy)
                ax.plot(xy[:,0],xy[:,1],style,color=color,label=role)
                ax.scatter(*xy[-1],color=color,marker='x' if label=='CONTACT' else 'o',s=25)
            points.extend([goal-1.2,goal+1.2]);points=np.asarray(points);lo=points.min(0)-.2;hi=points.max(0)+.2;center=(lo+hi)/2;side=max(hi-lo)
            ax.plot(path[:,0],path[:,1],':',color='0.6',label='original path')
            ax.add_patch(Circle(goal,1,fill=False,color='0.4'));ax.scatter(*goal,c='k',marker='*',label='original goal / 1m')
            ax.set(xlim=(center[0]-side/2,center[0]+side/2),ylim=(center[1]-side/2,center[1]+side/2),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)',title=f'{state}, F={force:g}N')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Original alpha0: live H75 vs nominal neural program and simple controls\nAll 24 observed suffixes; x contact, circle observed end; four-second window, no tail extrapolation')
    fig.savefig(out/'all_xy.png',dpi=140);plt.close(fig)
    result=dict(summary=summary,records=records,paired_live_program=paired,program_receipts=program_receipts,
        common_survivor_metrics=paired_complete_metrics(records,'future0','nominal_program',['original_speed_mae_m_s','path_shadow_rmse_m']),
        new_physics_ticks=status['active_lower_ticks'],new_allocated_slots=status['logical_slots']*6,
        interpretation='Two alpha0 source-eligible states, DEV mechanism only; not alpha1/general neural necessity, deployable program, continuous force threshold or 20-second task failure')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,paired=paired),indent=2))


if __name__=='__main__':main()
