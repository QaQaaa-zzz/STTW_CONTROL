"""Read original-goal and unweighted task diagnostics for the fixed 48 suffixes."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from sttw_control.geometric_path import build_path
from sttw_control.preference_history_task_review import shadow_cross_track, paired_complete_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(exist_ok=False)
    plan = json.loads((source / 'plan.json').read_text())
    review = json.loads((source / 'analysis/review.json').read_text())
    assert len(review['records']) == 48
    declaration = json.loads(Path(plan['source_declaration']).read_text())
    cfg = declaration['config']
    requested_speeds = {}
    for case in plan['cases']:
        speed = float(case['requested_speed'])
        assert case['state_id'] not in requested_speeds or requested_speeds[case['state_id']] == speed
        requested_speeds[case['state_id']] = speed
    old = Path(plan['source_run'])
    lineage = json.loads((old / 'snapshot_lineage.json').read_text())['cases']
    origins = {}
    inputs = {str(p): digest(p) for p in [source/'plan.json', source/'analysis/review.json',
              old/'snapshot_lineage.json', Path(plan['source_declaration']), Path(__file__).resolve()]}
    for row in lineage:
        if row['future_alpha'] != 0.:
            continue
        path = Path(row['snapshot'])
        assert digest(path) == row['snapshot_sha256']
        state = pickle.loads(path.read_bytes())
        origins[row['state_id']] = dict(xy=np.asarray(state.physical.data.qpos)[:2].astype(float),
                                        progress=float(state.projection.progress),
                                        user_alpha=row['entry_alpha'])
        inputs[str(path)] = digest(path)
    assert len(origins) == 4
    paths, path_receipts = {}, {}
    for route in declaration['routes']:
        file = Path(plan['path_evidence_directory']) / (route['name'] + '_path.npz')
        with np.load(file) as z:
            xy = z['xy'].astype(float)
        heading0 = float(np.arctan2(*(xy[1]-xy[0])[::-1]))
        # Reconstruct only intrinsic s/heading/goal from the original builder;
        # retained recorded world XY is authoritative, including float rounding.
        built = build_path(route['builder'], [*xy[0], heading0], cfg)
        path = {key: np.asarray(getattr(built, key)) for key in ('s', 'xy', 'heading', 'goal', 'turn_end')}
        reconstruction_error = float(np.max(abs(path['xy']-xy)))
        assert path['xy'].shape == xy.shape
        path['xy'] = xy
        goal = np.array([np.interp(float(path['goal']), path['s'], xy[:, j]) for j in range(2)])
        paths[route['name']] = (path, goal)
        inputs[str(file)] = digest(file)
        path_receipts[route['name']] = dict(original_xy_file=str(file), sha256=digest(file),
              inferred_initial_heading_rad=heading0, rebuilt_xy_max_difference_m=reconstruction_error,
              original_goal_s_m=float(path['goal']), original_turn_end_s_m=float(path['turn_end']),
              original_goal_world_xy_m=goal.tolist(), world_xy_source='retained recorded array; not rebuilt XY')
    (out/'design.json').write_text(json.dumps(dict(question='Do matched simple rules also cover four-second task diagnostics?',
        data_role='posthoc DEV; physical outcomes and sources already seen',
        metrics=['original path shadow RMSE/MAE/peak','original requested-speed MAE','travel length','first original-goal 1m entry'],
        no_scalar_Jalpha=True,new_physics_ticks=0,new_training_steps=0,
        initial_projection='genuine snapshot progress, bounded monotone local projection every logged lower tick',
        compare_only='pair-specific common full survivors, plus explicit four-rule common subset; fatal/excluded records retained'),indent=2)+'\n')
    records, traces = [], {}
    for row in review['records']:
        file = Path(row['raw_file'])
        assert digest(file) == row['raw_sha256']
        with np.load(file) as z:
            raw = {key: z[key] for key in z.files}
        active = raw['active_tick'].astype(bool)
        xy, t = raw['actual_xy'][active].astype(float), raw['time'][active].astype(float)
        assert len(t) == row['active_ticks'] and np.allclose(t, np.arange(len(t))*.005, atol=3e-6)
        origin = origins[row['state_id']]
        requested_speed = requested_speeds[row['state_id']]
        path, goal = paths[row['route']]
        ey, progress = shadow_cross_track(xy, path, cfg, initial_xy=origin['xy'], initial_progress=origin['progress'])
        distance = np.linalg.norm(xy-goal, axis=1)
        initial_inside = bool(np.linalg.norm(origin['xy']-goal) <= 1.)
        fatal = np.logical_or.reduce([raw[key][active].astype(bool) for key in
                                      ('contact_failure','nonfinite','lower_fault','policy_fault')])
        events = np.flatnonzero(fatal)
        stop = int(events[0]) if len(events) else len(t)
        hit = np.flatnonzero(distance[:stop] <= 1.)
        first = 0. if initial_inside else float(t[hit[0]]) if len(hit) else None
        v = raw['actual_forward_speed'][active].astype(float)
        records.append(dict(state_id=row['state_id'], role=row['role'], route=row['route'],
            force_N=row['force_N'], original_user_alpha=origin['user_alpha'],
            physical_label=row['label'], observed_ticks=len(t), observed_suffix_s=len(t)*.005,
            path_shadow_rmse_m=float(np.sqrt(np.mean(ey**2))), path_shadow_mae_m=float(np.mean(abs(ey))),
            path_shadow_peak_abs_m=float(np.max(abs(ey))),
            original_speed_mae_m_s=float(np.mean(abs(v-requested_speed))), original_requested_speed_m_s=requested_speed,
            travel_length_m=float(np.linalg.norm(np.diff(np.vstack([origin['xy'],xy]),axis=0),axis=1).sum()),
            initial_projection_m=origin['progress'], final_shadow_progress_m=float(progress[-1]),
            initially_inside_goal_1m=initial_inside, first_suffix_goal_1m_s=first,
            goal_hit_before_fatal=first is not None, goal_and_full_survival=first is not None and row['label']=='PASS',
            minimum_goal_distance_before_fatal_m=float(distance[:stop].min()) if stop else None,
            raw_sha256=row['raw_sha256']))
        traces[(row['state_id'],row['force_N'],row['role'])] = (xy,t)
    metrics = ['path_shadow_rmse_m', 'original_speed_mae_m_s', 'travel_length_m']
    pairs = [paired_complete_metrics(records, 'future1', baseline, metrics)
             for baseline in ('future0', 'zero_latent', 'speed_support')]
    roles = ('future0','future1','zero_latent','speed_support')
    index = {(r['state_id'],r['force_N'],r['role']):r for r in records}
    common = [dict(state_id=s,force_N=f) for s,f in sorted({(r['state_id'],r['force_N']) for r in records})
              if all(index[(s,f,role)]['physical_label']=='PASS' for role in roles)]
    common_summary = {}
    for role in roles:
        selected = [index[(e['state_id'],e['force_N'],role)] for e in common]
        common_summary[role] = {key: float(np.mean([r[key] for r in selected])) if selected else None for key in metrics}
    summary = {role: dict(n=12, any_suffix_goal_hits=sum(r['goal_hit_before_fatal'] for r in records if r['role']==role),
             goal_hits_with_full_survival=sum(r['goal_and_full_survival'] for r in records if r['role']==role),
             nominal_goal_hits=sum(r['goal_hit_before_fatal'] for r in records if r['role']==role and not r['force_N']),
             disturbed_goal_hits=sum(r['goal_hit_before_fatal'] for r in records if r['role']==role and r['force_N'])) for role in roles}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors = dict(zip(roles, ('#0072B2','#D55E00','#009E73','#CC79A7')))
    styles = dict(zip(roles, ('-','--','-.',':')))
    states = sorted(origins)
    fig, axes = plt.subplots(4,3,figsize=(15,17),layout='constrained')
    for i,sid in enumerate(states):
        route = next(r['route'] for r in records if r['state_id']==sid)
        path,goal = paths[route]
        for j,force in enumerate((0.,-32.,32.)):
            ax = axes[i,j]
            points = [goal+np.array([-1.25,-1.25]),goal+np.array([1.25,1.25])]
            for role in roles:
                xy,t = traces[(sid,force,role)]
                points.extend(xy)
                record = index[(sid,force,role)]
                ax.plot(xy[:,0],xy[:,1],color=colors[role],ls=styles[role],label=role)
                ax.scatter(*xy[-1],color=colors[role],marker='x' if record['physical_label']=='CONTACT' else 'o',s=24)
                if record['first_suffix_goal_1m_s'] is not None:
                    k=int(np.argmin(abs(t-record['first_suffix_goal_1m_s'])))
                    ax.scatter(*xy[k],color=colors[role],marker='D',s=20)
            points=np.asarray(points);lo,hi=points.min(axis=0)-.25,points.max(axis=0)+.25
            center=(lo+hi)/2;side=max(hi-lo)
            ax.plot(path['xy'][:,0],path['xy'][:,1],':',color='0.6',lw=1,label='original path')
            ax.add_patch(Circle(goal,1.,fill=False,color='0.35',lw=1))
            ax.scatter(*goal,color='black',marker='*',s=40,label='original goal / 1m')
            ax.set(xlim=(center[0]-side/2,center[0]+side/2),ylim=(center[1]-side/2,center[1]+side/2),
                   aspect='equal',xlabel='World X (m)',ylabel='World Y (m)',
                   title=f"{route}, entry alpha={origins[sid]['user_alpha']:g}, F={force:g}N")
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('All 48 observed four-second suffixes and the ORIGINAL goal (1m)\n'
                 'Diamond: first goal entry; x: contact; circle: observed endpoint; no tail extrapolation')
    fig.savefig(out/'all_xy_original_goal.png',dpi=140)
    plt.close(fig)
    result = dict(summary=summary, records=records, pair_specific_common_survivors=pairs,
        four_rule_common_survivors=common, four_rule_common_means=common_summary,
        path_reconstruction=path_receipts, inputs=inputs,
        interpretation_limits=['Suffix four-second diagnostics, not full task failure or 20-second completion',
            'Posthoc survivor subsets, no population dominance or interchangeable pair means',
            'Unweighted original geometry/speed measures, not V3 reward or canonical Jalpha',
            'After-domain shadow is geometry only, not an observation supplied back to the controller'])
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,common=len(common),pairs=[dict(candidate=p['candidate'],baseline=p['baseline'],n=p['common_complete'],means=p['mean_differences']) for p in pairs]),indent=2))


if __name__ == '__main__':
    main()
