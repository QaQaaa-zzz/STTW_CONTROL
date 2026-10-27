"""Audit all sealed actual traces after the fixed native comparison completes.

No physics, inference, training, interpolation of dynamics, or case selection.
Logged effective components and scored rewards are separate accounting objects.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(raw, goal, *, scale, dt, keys):
    active = np.asarray(raw['active_tick'], bool)
    if active.ndim != 1 or not active.all() or not 0 < len(active) <= 2000:
        raise ValueError('final per-lane raw must contain only actual active ticks')
    np.testing.assert_allclose(raw['time'], np.arange(len(active)) * dt, atol=3e-6, rtol=0)
    fault = np.asarray(raw['policy_fault'], bool) | np.asarray(raw['lower_fault'], bool)
    physical = np.asarray(raw['physical_failure'], bool)
    domain = np.asarray(raw['domain_exit'], bool)
    terminal = physical | domain | fault
    stop = int(np.flatnonzero(terminal)[0]) if terminal.any() else len(active)
    distances = np.linalg.norm(raw['actual_xy'][:stop] - goal, axis=1)
    hits = np.flatnonzero(distances <= 1.)
    effective = np.asarray(raw['effective_costs'], float)
    reward = np.asarray(raw['scored_tick_reward'], float)
    if effective.shape != (len(active), len(keys)) or not np.isfinite(effective).all() or not np.isfinite(reward).all():
        raise ValueError('unknown or inconsistent actual cost data')
    component_totals = scale * dt * effective.sum(axis=0)
    running = -scale * dt * effective.sum(axis=1)
    # Native failure substitutes its last upper block with a remaining-task
    # penalty. Do not call that difference another running cost component.
    ordinary = np.arange(len(active)) < ((stop // 4) * 4 if terminal.any() else len(active))
    error = float(np.max(np.abs(reward[ordinary] - running[ordinary]))) if ordinary.any() else None
    if error is not None and error > 3e-5:
        raise ValueError('ordinary scored reward differs from logged components')
    return dict(active_lower_ticks=len(active), observed_s=len(active)*dt,
        native_physical_failure=bool(physical.any()), native_domain_exit=bool(domain.any()),
        engineering_fault=bool(fault.any()), complete_horizon=len(active)==2000 and not terminal.any(),
        goal1m_before_native_stop=bool(len(hits)),
        first_arrival_s=float(raw['time'][hits[0]]) if len(hits) else None,
        minimum_goal_distance_before_native_stop_m=float(distances.min()) if len(distances) else None,
        scored_cost=float(-reward.sum()), running_cost=float(-running.sum()),
        terminal_replacement_adjustment=float(running.sum()-reward.sum()),
        task_channels=float(sum(component_totals[keys.index(k)] for k in ('speed','path','primary'))),
        components={k:float(v) for k,v in zip(keys,component_totals)},
        ordinary_reward_max_abs_error=error,
        path_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track']**2))),
        speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed']-raw['v_user'])**2))))


def main(root):
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=16:
        raise ValueError('all16 completed cases required; no interim outcome analysis')
    plan=json.loads((root/'plan.json').read_text())
    assert digest(root/'plan.json')==json.loads((root/'seal.json').read_text())['plan_sha256']
    out=root/'analysis';out.mkdir(exist_ok=False)
    manifest=json.loads((root/'evaluation/manifest.json').read_text());keys=manifest['reward_components']
    cfg=plan['config'];dt=cfg['timing']['lower_dt_s'];scale=cfg['path_reward']['scale']
    new=json.loads((root/'evaluation/results.json').read_text())['cases']
    cached=json.loads((root/'cached_records.json').read_text());assert len(new)==16 and len(cached)==24
    inputs={str(root/'plan.json'):digest(root/'plan.json')};records=[];traces={};geometries={}
    for item in new+cached:
        identity=dict(item)
        identity['role']=item.get('role',dict(H75='H75_50hz',zero='zero',equilibrium='equilibrium').get(item.get('family')))
        file=Path(item['raw_file']);assert digest(file)==item['raw_sha256'];inputs[str(file)]=digest(file)
        with np.load(file) as z:raw={k:z[k] for k in z.files}
        geomfile=root/'evaluation'/(item['route']+'_path.npz')
        with np.load(geomfile) as z:geo={k:z[k] for k in z.files}
        inputs[str(geomfile)]=digest(geomfile);geometries[item['route']]=geo
        goal=np.array([np.interp(float(geo['goal']),geo['s'],geo['xy'][:,k]) for k in (0,1)])
        np.testing.assert_allclose(raw['v_user'],item['speed'],atol=1e-6,rtol=0)
        assert not np.any(raw['applied_force_impulse'])
        row={k:identity[k] for k in ('role','route','speed','alpha','raw_file','raw_sha256')}
        row.update(summarize(raw,goal,scale=scale,dt=dt,keys=keys));records.append(row)
        traces[(row['role'],row['route'],row['speed'],row['alpha'])]=raw
        if item in cached:
            np.testing.assert_allclose(row['scored_cost'],item['user_objective_costs'][str(item['alpha'])],atol=5e-4,rtol=3e-5)
            assert row['goal1m_before_native_stop']==item['goal1m_before_native_stop']
    assert len(traces)==40
    comparisons=[]
    for c in plan['cases']:
        paired=[r for r in records if (r['route'],r['speed'],r['alpha'])==(c['route'],c['speed'],c['alpha'])]
        proposed=next(r for r in paired if r['role']=='reference_grid_5hz')
        for baseline in paired:
            if baseline is proposed:continue
            comparisons.append(dict(route=c['route'],speed=c['speed'],alpha=c['alpha'],baseline=baseline['role'],
                grid_minus_baseline_cost=proposed['scored_cost']-baseline['scored_cost'],
                grid_minus_baseline_task_channels=proposed['task_channels']-baseline['task_channels'],
                grid_goal=proposed['goal1m_before_native_stop'],baseline_goal=baseline['goal1m_before_native_stop']))
    actual=0;slots=0
    for f in sorted((root/'evaluation/chunks').glob('round_*.npz')):
        with np.load(f) as z:actual+=int(z['active_tick'].sum());slots+=z['active_tick'].size
    prediction=0;pslots=0
    for f in sorted((root/'evaluation/forecasts').glob('round_*.npz')):
        with np.load(f) as z:
            prediction+=int(z['observed_lower_ticks'].sum());pslots+=8*26*200
            assert np.array_equal(np.argmin(z['original_user_cost'],axis=1),
                json.loads((root/'evaluation/timings.json').read_text())[int(f.stem.split('_')[-1])]['selected'])
    assert actual==status['new_actual_lower_ticks']==sum(r['active_lower_ticks'] for r in records[:16])
    assert slots==status['allocated_actual_slots'] and prediction==status['new_prediction_lower_ticks'] and pslots==status['allocated_prediction_slots']
    timings=json.loads((root/'evaluation/timings.json').read_text())
    result=dict(records=records,comparisons=comparisons,accounting=dict(actual_lower_ticks=actual,actual_slots=slots,
        prediction_lower_ticks=prediction,prediction_slots=pslots,compile_s=status['compile_s'],
        elapsed_s=status['elapsed_s'],forecast_s=sum(t['forecast_s'] for t in timings),
        actor_s=sum(t['actor_s'] for t in timings),advance_s=sum(t['advance_s'] for t in timings),
        new_training_transitions=0,cached_cases=24),source_sha256=inputs,
        scope='All40 actual seen-DEV trajectories, no independent training repeats; ordinary privileged offline comparator, no perturbation/tolerance or real-time claim',
        cost_audit='Logged effective components and scored reward independently summed; full original cost formula reconstruction still a separate review')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('comparisons',comparisons),('records',[{k:v for k,v in r.items() if k!='components'} for r in records])]:
        with (out/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(out,plan,traces,geometries,keys,scale,dt)
    print(json.dumps(result['accounting'],indent=2))


def plot(out,plan,traces,geometries,keys,scale,dt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    styles={'reference_grid_5hz':('#0072B2','-'),'H75_5hz':('#D55E00','--'),
            'H75_50hz':('#009E73','-.'),'zero':('.45',':'),'equilibrium':('#CC79A7',(0,(5,2)))}
    for i,c in enumerate(plan['cases']):
        fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained');a=ax.flat
        geo=geometries[c['route']];a[0].plot(*geo['xy'].T,':',color='black',label='Original path')
        goal=np.array([np.interp(float(geo['goal']),geo['s'],geo['xy'][:,k]) for k in (0,1)])
        a[0].scatter(*goal,marker='*',color='black');a[0].add_patch(Circle(goal,1,fill=False,color='.5',lw=.7))
        for role,(color,style) in styles.items():
            r=traces[(role,c['route'],c['speed'],c['alpha'])];t=r['time'];cost=-np.cumsum(r['scored_tick_reward'],dtype=float)
            for panel,x,y in [(0,r['actual_xy'][:,0],r['actual_xy'][:,1]),(1,t,r['scored_tick_reward']),
                (2,t,cost),(3,t,r['path_cross_track']),(4,t,r['actual_forward_speed']),(5,t,r['phi'])]:
                a[panel].plot(x,y,color=color,ls=style,lw=1,label=role)
            a[0].scatter(*r['actual_xy'][-1],color=color,s=20,marker='x' if r['physical_failure'].any() else 'D' if r['domain_exit'].any() else 's')
        a[0].set(xlabel='World X (m)',ylabel='World Y (m)');a[0].set_aspect('equal',adjustable='box')
        for p,label in enumerate(['Scored reward / lower tick','Cumulative original cost','Path error (m)','Forward speed (m/s)','Roll (rad)'],1):
            a[p].set(xlabel='Task time (s)',ylabel=label);a[p].grid(alpha=.2)
        a[4].axhline(c['speed'],color='black',lw=.6,ls=':')
        handles,labels=a[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle(f"Native nominal10s / {c['route']} / requested{c['speed']:g}m/s / user alpha{c['alpha']:g}\nAll actual traces; circle=goal1m; square=horizon, diamond=domain, x=physical; offline computation delay unmodeled")
        for ext in ('png','pdf'):fig.savefig(out/f'case_{i:02d}.{ext}',dpi=150)
        plt.close(fig)
        fig,axes=plt.subplots(2,5,figsize=(16,6),layout='constrained')
        for role,(color,style) in styles.items():
            r=traces[(role,c['route'],c['speed'],c['alpha'])]
            for k,panel in zip(range(len(keys)),axes.flat):
                panel.plot(r['time'],scale*dt*r['effective_costs'][:,k],color=color,ls=style,lw=.8,label=role)
                panel.set(title=keys[k],xlabel='Task time (s)',ylabel='Running cost / lower tick');panel.grid(alpha=.2)
        fig.suptitle(f"Case{i:02d}: all original capped running components; terminal replacement separately reported")
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
        for ext in ('png','pdf'):fig.savefig(out/f'components_{i:02d}.{ext}',dpi=150)
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
