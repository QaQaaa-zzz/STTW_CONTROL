"""Prospective, same-slot evaluation on a wide bend, narrow bend and exit ridge."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from pcrp_review_batches import LAB, load, write
from sttw_control.isaac_path_reporting import replay_rewards, static_case_metrics
from sttw_control.route_pebbles import project_route

GYM = Path('/home/qy/miniconda3/envs/go2_isaacgym')
METHODS = {'wide_to_narrow': (1., 0.), 'fixed0': (0., 0.), 'fixed05': (.5, .5),
           'fixed1': (1., 1.), 'Zero': (0., 0.)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    source = root.parent / 'pcrp_local300_p0_20261121/full_ground/s_bend_pulse'
    controls = {g: load(source / f'flat_{g}/contract.json') for g in range(101, 106)}
    base = controls[101]
    bundle = Path(base['bundle'])
    route_file = bundle / 'route.npz'
    route = np.load(route_file)
    assert digest(route_file) == base['identity']['files']['route.npz']
    assert np.allclose(route['xy'][0], base['identity']['world_anchor'][:2], atol=1e-5)
    gate = 7.
    center = 12.5
    assert route['curvature'][route['s'] < gate].max() > .6
    assert route['curvature'][route['s'] > gate].min() < -.6 and float(route['turn_end']) < center < float(route['goal'])
    root.mkdir(parents=True, exist_ok=False)
    geom = load(root.parent/'h75_theory_iteration/calibration/single04_scene.json')
    # Use the mature SDK mesh path. Compute its original-route coordinates once.
    geom.update(route=str(route_file), bounds_m=[-5.,40.], center_s_m=center, switch_s_m=gate,
                projection_cache=str(root/'grid_projection.npz'), role='Predeclared corridor exit ridge')
    axis = np.arange(geom['bounds_m'][0], geom['bounds_m'][1] + .5*geom['horizontal_scale_m'], geom['horizontal_scale_m'])
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    section = route['s'] <= float(route['turn_end']) + 3
    progress, lateral = project_route(np.column_stack((gx.ravel(), gy.ravel())),
                                     {k: route[k][section] for k in ('xy','s')})
    np.savez(geom['projection_cache'], s=progress, n=lateral)
    write(root/'ridge.json', geom)
    raw_mesh, terrain = root/'grid_mesh', root/'terrain'
    subprocess.run([GYM/'bin/python', 'learning/cli/place_isaac_terrain.py', '--terrain', 'bend_bump',
                    '--scene-config', root/'ridge.json', '--output', raw_mesh], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, LD_LIBRARY_PATH=str(GYM/'lib')+os.pathsep+os.getenv('LD_LIBRARY_PATH','')))
    subprocess.run([sys.executable, 'learning/cli/compact_isaac_terrain.py', '--source', raw_mesh,
                    '--output', terrain], check=True)
    meta = load(terrain/'identity.json')
    with np.load(terrain/'terrain.npz') as mesh:
        assert np.isclose(mesh['vertices'][:,2].max(), geom['height_m'])
        assert np.min(mesh['vertices'][:,:2]) <= -5 and np.max(mesh['vertices'][:,:2]) >= 40
    config = {k: base['settings'][k] for k in ('asset','controller_config','lower_actor','lower_sha256',
                                                  'plant_options','initial_state','selected_lower')}
    config.update(auto_reset_horizon=False, episode_s=10., record_control_commands=True,
                  record_controller_terms=True, omit_diagnostic_fields=['observation','policy_measurement'],
                  evaluation_role='Prospective continuous-corridor DEV')
    identities = {str(bundle/name):sha for name,sha in base['identity']['files'].items()}
    for f in (config['lower_actor'], config['initial_state'], config['asset'], config['controller_config']):
        path = Path(f).resolve()
        identities[str(path)] = digest(path)
    assert digest(Path(config['lower_actor'])) == config['lower_sha256']
    identities[str(terrain/'terrain.npz')] = meta['sha256']
    identities[str(Path(__file__).resolve())] = digest(Path(__file__))
    batches = {}
    jobs = []
    for method, (prefix, target) in METHODS.items():
        folder = root/method
        folder.mkdir()
        cases = []
        for speed in (1.9, 2.3):
            for group, c in controls.items():
                assert c['bundle'] == str(bundle) and c['settings']['initial_state'] == config['initial_state']
                cases.append(dict(seed=group, user_speed_m_s=speed, randomization=c['settings']['randomization'],
                                  terrain_path=str(terrain), output=str(folder/f'v{speed}_g{group}'),
                                  alphas=[prefix], zero_middle=[method=='Zero'], case_labels=[method],
                                  alpha_switch=dict(progress_s_m=gate, progress_frame='original_route',
                                                    target_alpha=[target], execution_modes=['independent'])))
        write(folder/'cases.json', cases)
        write(folder/'settings.json', config)
        identities[str(folder/'settings.json')] = digest(folder/'settings.json')
        batches[method] = dict(lanes=len(cases), bundle=str(bundle), manifest_sha256=digest(folder/'cases.json'))
        jobs.append(dict(config=str(folder/'settings.json'),case_manifest=str(folder/'cases.json'),
                         output=str(folder/'batch'),terrain=str(terrain)))
    write(root/'jobs.json', jobs)
    identities[str(root/'jobs.json')] = digest(root/'jobs.json')
    write(root/'protocol.json', dict(experiment='PCRP prospective continuous corridor', data_role='exploratory DEV',
          code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          identities=identities, batches=batches, lanes=50, max_physics_lane_steps=2500000, optimizer_updates=0,
          route=str(route_file), gate_s_m=gate, half_width_m=[.25,.10], ridge_center_s_m=center,
          ridge_height_m=geom['height_m'], request_m_s=[1.9,2.3], goal_s_m=float(route['goal']),
          seconds=10, physics_dt=.0002, lower_dt=.005, policy_dt=.02, groups=list(controls),
          source_cases=str(source), pairing='One resident USD scene, five PhysX resets, identical 10 clone slots and order, asset, '
          'original hot start, group properties and no pulse. All five arms evaluate the location trigger. '
          'Only the wide_to_narrow arm changes preference at the first forward crossing. No reset at crossing.',
          primary='Survive the full 10 s, cross the original goal plane by 10 s and satisfy the geometric half-width '
          'at every observed 200 Hz sample including the last partial physical step. Bounds constrain the tracked '
          'vehicle reference point. No physical wall collisions. Deadline and geometric outcomes separate.',
          secondary='Report first crossing time, signed error, first violation, actual endpoint, cause flags, '
          'final hold, speed error, under/over time, reward components and each paired gain/loss.',
          stop='One pass through five arms or first process/output error. No parameter selection or retry.'))
    write(root/'status.json',dict(stage='prepared',episodes=50,physics_started=False))


def score_lane(data, contract, protocol):
    trace = data['lower_trace'][:,:,0].reshape(-1,11)
    ids = np.flatnonzero(trace[:,6])
    active = trace[ids]
    time = ids*contract['lower_dt'] + active[:,7]*contract['physics_dt']
    route = np.load(protocol['route'])
    progress, ey = project_route(active[:,:2].astype(float), route)
    limit = np.where(progress < protocol['gate_s_m'], *protocol['half_width_m'])
    violation = np.flatnonzero(np.abs(ey) > limit)
    goal = protocol['goal_s_m']
    goal_point = [np.interp(goal,route['s'],axis) for axis in route['xy'].T]
    heading = np.interp(goal,route['s'],route['heading'])
    cross = np.flatnonzero((data['progress'][ids,0]>=goal) &
                           ((active[:,:2]-goal_point)@[np.cos(heading),np.sin(heading)]>=0))
    switched = data['switch_tick'][ids[-1],0]
    alpha = data['alpha'][ids,0]
    s = contract['settings']
    expected = np.where((switched>=0)&(ids>=switched), s['alpha_switch']['target_alpha'][0], s['alphas'][0])
    np.testing.assert_array_equal(alpha, expected)
    metrics = static_case_metrics(data,contract)[0]
    full = not metrics['failed'] and abs(time[-1]-protocol['seconds'])<1e-6
    assert bool(len(cross))==metrics['goal_crossed']
    record = dict(full=full, hold=metrics['final_hold'], goal=bool(len(cross)), goal_s=time[cross[0]] if len(cross) else None,
                  deadline_missed=not len(cross), in_band=not len(violation),
                  task=full and bool(len(cross)) and not len(violation),
                  first_out_s=time[violation[0]] if len(violation) else None,
                  first_out_progress_m=progress[violation[0]] if len(violation) else None,
                  switch_time_s=switched*contract['lower_dt'] if switched>=0 else None,
                  out_duration_s=np.sum(active[np.abs(ey)>limit,7])*contract['physics_dt'],
                  end_s=time[-1], peak_cross_m=np.max(abs(ey)), speed_rmse_m_s=metrics['speed_rmse_m_s'],
                  underspeed_s=active[active[:,2]<s['user_speed_m_s'],7].sum()*contract['physics_dt'],
                  overspeed_s=active[active[:,2]>s['user_speed_m_s'],7].sum()*contract['physics_dt'])
    for key in ('contact_failure','roll_limit_failure','policy_fault','domain_exit','numerical_failure'):
        record[key] = bool(data[key][ids,0].any()) if key in data else None
    return record, dict(time_s=time, xy=active[:,:2], progress_m=progress, cross_m=ey, half_width_m=limit,
                        speed_m_s=active[:,2], wheel_m_s=active[:,10], alpha=alpha)


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = load(root/'protocol.json')
    route = np.load(spec['route'])
    report = root/'analysis'
    report.mkdir(exist_ok=True)
    cases = {name:load(root/name/'cases.json') for name in spec['batches']}
    records, gallery = [], []
    checkpoint = load(Path(spec['batches']['Zero']['bundle'])/'identity.json')['files']['actor.npz'][:8]
    for slot, base in enumerate(cases['Zero']):
        fig,axes = plt.subplots(3,2,figsize=(11,9),layout='constrained')
        xy, error, speed, preference, reward, cumulative = axes.flat
        cut = route['s']<=26
        arc = route['s'][cut]
        gate = spec['gate_s_m']
        # Repeat the junction to draw the width step at exactly the registered section.
        arc = np.r_[arc[arc<gate],gate,gate,arc[arc>gate]]
        center = np.column_stack([np.interp(arc,route['s'],col) for col in route['xy'].T])
        angle = np.interp(arc,route['s'],route['heading'])
        width = np.where(arc<gate,*spec['half_width_m'])
        width[np.flatnonzero(arc==gate)[0]] = spec['half_width_m'][0]
        normal = width[:,None]*np.column_stack((-np.sin(angle),np.cos(angle)))
        xy.plot(*center.T,'k:',label='Original route')
        for sign in (-1,1): xy.plot(*(center+sign*normal).T,color='.6',lw=1)
        ridge = np.array([np.interp(spec['ridge_center_s_m'],route['s'],axis) for axis in route['xy'].T])
        xy.scatter(*ridge,marker='+',color='black',label='Ridge center')
        matches = []
        for method, rows in cases.items():
            row = rows[slot]
            folder = Path(row['output'])
            contract = load(folder/'contract.json')
            assert row['seed']==base['seed'] and row['user_speed_m_s']==base['user_speed_m_s']
            matches.append([contract[k] for k in ('origin','randomization_readback','delay_steps')])
            with np.load(folder/'trajectory.npz') as data:
                record, signals = score_lane(data, contract, spec)
                cfg = load(Path(contract['bundle'])/'config.json')
                _, parts, closure = replay_rewards(data,cfg)
                steps = np.flatnonzero(data['valid'][:,0])
                values = data['reward'][steps,0]
                np.savez_compressed(folder/'corridor_scores.npz',**signals,**parts,policy_steps=steps+1,
                                    policy_reward=values)
            records.append(dict(method=method, speed=row['user_speed_m_s'], group=row['seed'], slot=slot,
                                raw=str(folder), reward_replay_max_error=closure, **record))
            x = signals['xy']
            line, = xy.plot(*x.T,label=method,ls='--' if method=='Zero' else '-')
            color = line.get_color()
            xy.plot(*x[-1],marker='o' if record['full'] else 'x',color=color)
            t = signals['time_s']
            for ax,y in ((error,signals['cross_m']), (speed,signals['speed_m_s']-row['user_speed_m_s']), (preference,signals['alpha'])):
                ax.plot(t,y,color=color,ls=line.get_linestyle())
            for sign in (-1,1):error.plot(t,sign*signals['half_width_m'],color=color,lw=.6,ls=':')
            reward.plot(steps+1,values,color=color); cumulative.plot(steps+1,np.cumsum(values),color=color)
        assert all(m==matches[0] for m in matches), 'Physical slot readbacks differ across methods'
        xy.axis('equal'); xy.set(xlabel='World x (m)',ylabel='World y (m)')
        for ax,label in ((error,'Signed cross-track error (m)'),(speed,'Speed error (m/s)'),(preference,'Preference')):
            ax.set(xlabel='Time (s)',ylabel=label)
        reward.set(xlabel='Policy step',ylabel='Recorded reward'); cumulative.set(xlabel='Policy step',ylabel='Cumulative reward')
        fig.legend(*xy.get_legend_handles_labels(),loc='outside lower center',ncol=4)
        name=f"v{base['user_speed_m_s']}_g{base['seed']}"
        fig.suptitle(f"Wide bend, narrow bend and exit ridge | {name} | H75 {checkpoint} | circle 10 s, cross early stop")
        fig.savefig(report/(name+'.png'),dpi=150); plt.close(fig)
        gallery.append(f'- [{name}]({name}.png)')
    with (report/'episodes.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary=['# Continuous corridor, matched DEV','','[Per-trajectory outcomes](episodes.csv)','',
             '| Method | Full | Goal by 10 s | In band until endpoint | Task | Hold | Switch gains/losses on task |',
             '|---|---:|---:|---:|---:|---:|---:|']
    switched=[r for r in records if r['method']=='wide_to_narrow']
    for method in cases:
        arm=[r for r in records if r['method']==method]
        counts=[f'{sum(bool(r[k]) for r in arm)}/{len(arm)}' for k in ('full','goal','in_band','task','hold')]
        pairs=list(zip(switched,arm,strict=True))
        summary.append('| '+' | '.join([method,*counts,
            f"{sum(a['task'] and not b['task'] for a,b in pairs)}/{sum(b['task'] and not a['task'] for a,b in pairs)}"])+' |')
    (report/'README.md').write_text('\n'.join([*summary,'',*gallery])+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','analyze'))
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    root=a.output.resolve()
    if a.action=='prepare': prepare(root)
    else:
        if a.action=='run':
            for path,expected in load(root/'protocol.json')['identities'].items():
                if digest(Path(path))!=expected: raise ValueError(f'Frozen input changed: {path}')
            spec=load(root/'protocol.json')
            for name,stage in spec['batches'].items():
                if digest(root/name/'cases.json')!=stage['manifest_sha256']:
                    raise ValueError(f'Frozen cases changed: {name}')
            write(root/'status.json',dict(stage='running',episodes=50))
            subprocess.run([sys.executable,'learning/scripts/notify_sttw_job.py','--message',
                            'PCRP 连续走廊开始，五种方法依次复用相同十个物理槽。',
                            '--receipt',str(root/'start_notice.json')])
            bundle=spec['batches']['Zero']['bundle']
            with (root/'simulation.log').open('x') as log:
                cmd=[LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',bundle,
                     '--seconds','10','--resident-jobs',str(root/'jobs.json')]
                rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            for name in spec['batches']:
                status=root/name/'batch/status.json'
                if not status.exists() or load(status).get('policy_steps')!=500:
                    rc=rc or 1
            if rc:
                write(root/'status.json',dict(stage='physics_error',exit_code=rc))
                subprocess.run([sys.executable,'learning/scripts/notify_sttw_job.py','--message',
                                f'PCRP 连续走廊仿真异常，查看 {root}/simulation.log',
                                '--receipt',str(root/'physics_error_notice.json')])
                raise SystemExit(rc)
            write(root/'status.json',dict(stage='physics_complete',episodes=50))
        if a.action=='analyze':
            analyze(root)
        else:
            result=subprocess.run([sys.executable,__file__,'analyze','--output',str(root)])
            stage='analysis_error' if result.returncode else 'complete'
            location=root/'analysis/README.md'
            write(root/'status.json',dict(stage=stage,episodes=50,report=str(location)))
            subprocess.run([sys.executable,'learning/scripts/notify_sttw_job.py','--message',
                            f'PCRP 连续走廊分析 {stage}，结果位置 {location}',
                            '--receipt',str(root/(stage+'_notice.json'))])
            raise SystemExit(result.returncode)
