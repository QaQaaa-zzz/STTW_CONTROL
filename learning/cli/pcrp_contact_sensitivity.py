"""Prepare and execute the paired 1 cm ridge/contact-material sensitivity grid."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pcrp_replay_gate import REPO, read, sha, write

sys.path.insert(0, str(REPO/'learning/src'))

LAB = '/home/qy/miniconda3/envs/env_isaaclab/bin/python'
CENTERS = (3.9, 4.2, 4.5)
ALPHAS = (0., .5, 1.)
SETTINGS = {
    'nominal': {}, 'sdf384': {'sdf_resolution': 384},
    'stiffness80': {'stiffness_scale': .8}, 'stiffness120': {'stiffness_scale': 1.2},
    'damping80': {'damping_scale': .8}, 'damping120': {'damping_scale': 1.2},
    'friction072': {'friction': .72}, 'friction088': {'friction': .88},
    'physics10k': {'physics_dt': .0001},
}


def prepare(root, e2):
    root.mkdir(parents=True, exist_ok=False)
    history = read(e2.parent/'step0/protocol.json')
    bundle = Path(history['bundle'])
    ids = history['identities']
    for filename in ('actor.npz', 'config.json', 'route.npz'):
        if sha(bundle/filename) != ids[filename]:
            raise ValueError(f'Mismatched frozen bundle member {filename}')
    scenes = []
    lanes = []
    frozen_slots = None
    for name, change in SETTINGS.items():
        for shift in (0, 1):
            block = root/name/f'ds{shift:+d}cm'
            block.mkdir(parents=True)
            jobs = []
            for alpha in ALPHAS:
                src = e2/'matched'/f'ds{shift:+d}cm_a{alpha:g}'
                folder = block/f'a{alpha:g}'
                folder.mkdir()
                settings = read(src/'settings.json')
                source = read(src/'cases.json')
                original = read(src/'batch/contract.json')
                opts = settings['plant_options']
                selected = [i for i,c in enumerate(source) if c['seed'] in range(401,405)]
                if len(selected) != 12 or original['identity']['files'] != {k:ids[k] for k in ('actor.npz','config.json','route.npz')}:
                    raise ValueError(f'Unexpected E2 case inventory or frozen actor at {src}')
                slots = [opts['replay_clone_origins'][i] for i in selected]
                if frozen_slots is None:
                    frozen_slots = slots
                    for key, filename in [('local300','lower_actor'),('initial_state','initial_state'),
                                          ('controller','controller_config'),('asset','asset')]:
                        if sha(REPO/settings[filename]) != ids[key]:
                            raise ValueError(f'Mismatch in archived {key}')
                if frozen_slots != slots:
                    raise ValueError(f'Mismatched world translations at {src}')
                opts.update(replay_clone_origins=slots, sdf_resolution=change.get('sdf_resolution',256),
                            physics_dt=change.get('physics_dt',.0002), contact_material_friction=change.get('friction',.8))
                for row in opts['compliant_wheels'].values():
                    for key in ('stiffness','damping'):
                        row[key] *= change.get(key+'_scale',1.)
                cases = []
                for slot, old in enumerate(selected):
                    c = source[old]
                    terrain = Path(c['terrain_path'])
                    meta = original['terrain'][old]
                    center = CENTERS[slot//4]
                    if (c['alphas'] != [alpha] or c['seed'] != 401+slot%4 or sha(terrain/'terrain.npz')!=meta['sha256']
                            or not np.isclose(meta['center_s_m'], center+shift/100) or meta['height_m']!=.1):
                        raise ValueError(f'Mismatched geometry, seed or alpha in {src} slot {old}')
                    c['output'] = str(folder/Path(c['output']).name)
                    if 'friction' in change:
                        c['randomization']['friction_range'] = [change['friction']]*2
                    cases.append(c)
                    lanes.append(dict(setting=name, center_m=center, ds_cm=shift, group=c['seed'], alpha=alpha,
                                      slot=slot, origin=slots[slot], original_e2_source=str(src/Path(c['output']).name),
                                      output=c['output'],terrain_sha256=meta['sha256']))
                settings['evaluation_role']='Contact-model OAT on previously examined DEV groups; 0/+1 cm, three preferences'
                settings['randomization']=cases[0]['randomization']
                write(folder/'cases.json', cases)
                write(folder/'settings.json', settings)
                jobs.append(dict(config=str(folder/'settings.json'),terrain=cases[0]['terrain_path'],
                                 case_manifest=str(folder/'cases.json'),output=str(folder/'batch')))
            write(block/'jobs.json', jobs)
            scenes.append(str(block))
    write(root/'lanes.json',lanes)
    protocol=dict(name='PCRP contact-model OAT',data_role='exploratory paired DEV',archive=str(e2),bundle=str(bundle),
        identities=ids,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        scope='Frozen H75 + Local300, same initialization, fixed original route, 10 cm oblique ridge at 2.6 m/s, 10 s.',
        budget=dict(settings=9,worlds_per_rollout=12,geometry_jobs=18,rollouts=54,physical_episodes=len(lanes),training_updates=0),
        settings=SETTINGS,nominal={'sdf_resolution':256,'physics_dt_s':.0002,
            'wheel_materials':read(e2/'matched/ds+0cm_a0/settings.json')['plant_options']['compliant_wheels'],
            'ground_friction':.8,'vehicle_friction':'archived per physical group; average-combine with ground'},
        groups=[401,402,403,404],alphas=ALPHAS,centers_m=CENTERS,offsets_cm=[0,1],scenes=scenes,
        friction='Both terrain and vehicle material set to the named coefficient in each friction arm.',
        clocks='5 ms lower and 20 ms policy throughout. Physics refinement doubles substeps, preserves physical delay durations.',
        pairing='Fixed 12 physical slots and world origins across all jobs. Group, slot and other factors held for each difference.',
        f_s='C10 flip for ds=0 to +1 cm at each group, placement and alpha, sum / 36',
        f_alpha='C10 flip for each adjacent pair of 0,.5,1 at ds=0, sum / 24; ratio undefined for zero denominator',
        reversal='Strict C10 count advantage of alpha0 at 3.9 m AND of alpha1 at 4.5 m, each /4 at ds=0',
        interpretation='This three-alpha, two-offset diagnostic is distinct from the historical five-alpha, seven-offset grid.',
        stop='One sweep, 648 episodes maximum; no automatic run after a failed batch; all endpoints retained.')
    write(root/'protocol.json', protocol)


def notice(folder, name, message):
    receipt=folder/(name+'_notice.json')
    with (folder/'notifications.log').open('a') as stream:
        p=subprocess.run([sys.executable,'learning/scripts/notify_sttw_job.py','--message',message,
                          '--receipt',str(receipt)],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    if p.returncode:
        write(receipt, dict(status='failed', exit_code=p.returncode, log=str(folder/'notifications.log')))


def verify_job(job):
    batch=Path(job['output'])
    cfg=read(Path(job['config']))
    cases=read(Path(job['case_manifest']))
    opts=cfg['plant_options']
    c=read(batch/'contract.json')
    status=read(batch/'status.json')
    n=len(cases)
    dt=opts['physics_dt']
    if (c['origin']!=opts['replay_clone_origins'] or [c[k] for k in ('physics_dt','lower_dt','policy_dt')]!=[dt,.005,.02]
        or c['settings']['alphas']!=[row['alphas'][0] for row in cases] or status['stage']!='complete'
        or status['policy_steps']!=500 or status['physics_steps']!=n*round(10/dt) or status['push_events']!=0):
        raise ValueError(f'Slot, clock or budget mismatch at {batch}')
    data=c['lab_physics']['contact_oat_readback']
    np.testing.assert_allclose(data['ground_friction'],opts['contact_material_friction'])
    live=np.asarray(c['randomization_readback']['materials'])
    for i,case in enumerate(cases):
        np.testing.assert_allclose(live[i,:,:2],case['randomization']['friction_range'][0])
        for buf in c['delay_steps'].values():
            if buf[i]!=round(case['randomization']['latency_s'][0]/dt):
                raise ValueError(f'Delay mismatch at {batch}, slot {i}')
    for wheel,parameters in opts['compliant_wheels'].items():
        w=data['wheels'][wheel]
        if w['sdf_resolution']!=[opts['sdf_resolution']]:
            raise ValueError(f'SDF readback mismatch at {batch} {wheel}: {w}')
        for k in ('stiffness','damping'):
            np.testing.assert_allclose(w[k],parameters[k],rtol=1e-6)
        # The frozen six-body asset has one shape per articulation link.
        shape_idx=c['bodies'].index(wheel)
        if not np.all(np.isclose(live[:,shape_idx,2],-parameters['stiffness'])):
            raise ValueError(f'PhysX stiffness differs from authored wheel material at {batch}')


def run(root):
    spec=read(root/'protocol.json')
    started=time.monotonic()
    done=0
    for blockpath in spec['scenes']:
        block=Path(blockpath)
        jobs=read(block/'jobs.json')
        label='/'.join(block.parts[-2:])
        write(root/'status.json',dict(stage='running',scene=label,completed=done,budget=648))
        notice(block,'start',f'STTW 第一章接触模型敏感性 {label} 开始，3档×12个同槽工况。')
        cmd=[LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',spec['bundle'],
             '--seconds','10','--resident-jobs',str(block/'jobs.json')]
        env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
        with (block/'simulation.log').open('x') as stream:
            subprocess.run(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        for job in jobs:
            verify_job(job)
        done+=36
        notice(block,'complete',f'STTW 第一章接触模型 {label}，36条完成并核实物理参数。{block}')
    analyze(root)
    write(root/'status.json',dict(stage='complete',completed=done,wall_s=time.monotonic()-started))
    notice(root,'complete',f'STTW 第一章接触模型敏感性648条完成。{root}/analysis/README.md')


def read_episode(lane, bundle_config):
    """Retain actual steps, including the failure transition, on both controller clocks."""
    from pcrp_replay_gate import FLAGS
    from sttw_control.ridge_reload import first_front_reload
    from sttw_control.isaac_path_reporting import replay_rewards, static_case_metrics

    raw=Path(lane['output'])
    contract=read(raw/'contract.json')
    batch=contract['batch_source']
    s=contract['settings']
    surface=contract['terrain'][0]
    if (batch['lane_start']!=lane['slot'] or batch['lane_stop']!=lane['slot']+1 or
            contract['origin']!=[lane['origin']] or s['seed']!=lane['group'] or
            s['alphas']!=[lane['alpha']] or surface['sha256']!=lane['terrain_sha256'] or
            not np.isclose(surface['center_s_m'],lane['center_m']+.01*lane['ds_cm'],rtol=0,atol=1e-6)):
        raise ValueError(f'Case manifest differs from observed slot, group, alpha or terrain in {raw}')
    with np.load(raw/'trajectory.npz') as data:
        metric=static_case_metrics(data,contract)[0]
        rebuilt,components,error=replay_rewards(data,bundle_config)
        trace=data['lower_trace'].reshape(-1,11)
        live=np.flatnonzero(trace[:,6])
        steps=np.flatnonzero(data['valid'][:,0])
        reward=data['reward'][steps,0]
        # A pre-step terminal can have a reward but no live lower sample in that interval.
        last=steps[-1]*4+3
        flags={k:int(data[k][:last+1,0].any()) for k in FLAGS}
        sig=dict(time_s=live*.005+trace[live,7]*contract['physics_dt'],xy_m=trace[live,:2],
                 speed_m_s=trace[live,2],rear_wheel_m_s=data['measurement'][live,0,5]*.1,
                 path_error_m=trace[live,3],progress_m=data['progress'][live,0],
                 lower_step=live+1,policy_step=steps+1,reward=reward,
                 cumulative_reward=np.cumsum(reward,dtype=float))
        # Forces are latched after the final physics substep in each 5 ms
        # interval. In a partial terminal interval this force is after death.
        # Keep the true endpoint for the trajectory, exclude that later force.
        n_sub=round(contract['lower_dt']/contract['physics_dt'])
        complete=trace[live,7]==n_sub
        if contract['lower_dt']!=.005 or not np.array_equal(live,np.arange(len(live))) or not complete[:-1].all():
            raise ValueError(f'Expected contiguous 200 Hz post-control contact records at {raw}')
        valid_count=int(complete.sum())
        forces=data['contact_force'][:valid_count,0]
        names=contract['lab_physics']['runtime_inertial']['body_names']
        if names!=contract['bodies']:
            raise ValueError(f'Unexpected force channels at {raw}')
        wheel_forces=np.linalg.norm(forces[:,[names.index('frontwheel'),names.index('rearwheel')]],axis=2)
        if not np.isfinite(wheel_forces).all():
            raise ValueError(f'Invalid observed wheel-contact force at {raw}')
        center=surface['center_s_m']
        # Progress is arclength on the fixed route in the actual contract;
        # the route anchor is distinct from the world x-coordinate of the ridge.
        progress=sig['progress_m'][:valid_count]
        t=sig['time_s'][:valid_count]
        approach,search,crest=(next(iter(np.flatnonzero(progress>=center-delta)),None) for delta in (.40,.10,0))
        k,num,den=first_front_reload(progress,*wheel_forces.T,center)
        reload_stats=dict(actual_ridge_center_s_m=center,route_start_x_m=contract['identity']['world_anchor'][0],
                          route_start_y_m=contract['identity']['world_anchor'][1],force_samples=valid_count,
                          last_force_s=t[-1] if valid_count else None,
                          partial_terminal_force_excluded=int(not complete[-1]),
                          approach_s=None if approach is None else t[approach],
                          search_s=None if search is None else t[search],crest_s=None if crest is None else t[crest],
                          front_reload_identified=int(k is not None),front_reload_s=None if k is None else t[k],
                          front_reload_progress_m=None if k is None else progress[k],
                          rear_support_samples=num,reload_window_samples=den,dual_frac=num/den if den else None,
                          rear_support_class=('no_event' if not den else 'none' if not num else 'half_or_more'
                                              if 2*num>=den else 'intermediate'))
        # Signed components cover each valid policy transition. Keep all four lower
        # allocations so their sum reproduces the logged terminal penalty.
        lower_rows=(4*steps[:,None]+np.arange(4)).ravel()
        sig.update(reward_lower_step=lower_rows+1,reward_lower=rebuilt[lower_rows,0],
                   **{'component_'+k:v[lower_rows,0] for k,v in components.items()})
    return dict(**lane,**reload_stats,C10=int(not metric['failed']),final_hold=int(metric['final_hold']),
                goal_crossed=int(metric['goal_crossed']),observed_s=metric['observed_seconds'],
                path_rmse_m=metric['path_rmse_m'],speed_rmse_m_s=metric['speed_rmse_m_s'],
                peak_roll_rad=metric['peak_roll_rad'],reward_sum=float(reward.sum(dtype=float)),
                reward_closure_max=error,terminal=('+'.join(k for k in FLAGS if flags[k]) or
                ('unresolved' if metric['failed'] else 'full_window')),**flags),sig


def plot_cases(folder, lanes, config, route):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True,exist_ok=True)
    colors=('#0072b2','#b95500','#00825d')
    rows=[]
    gallery=[]
    for ds in (0,1):
        for center in CENTERS:
            for group in range(401,405):
                images=[]
                base={}
                for setting in SETTINGS:
                    fig,axes=plt.subplots(2,3,figsize=(11.8,6.5),layout='constrained')
                    xy,per,cum,velocity,path,wheel=axes.flat
                    xy.plot(*route.T,color='.38',ls=':',lw=1,label='original route')
                    arrays={}
                    for alpha,color in zip(ALPHAS,colors):
                        lane=lanes[setting,center,group,ds,alpha]
                        row,cur=read_episode(lane,config)
                        rows.append(row)
                        if setting=='nominal':
                            base[alpha]=cur
                        t=cur['time_s']
                        for ax,x,y in [(xy,cur['xy_m'][:,0],cur['xy_m'][:,1]),
                                       (per,cur['policy_step'],cur['reward']),
                                       (cum,cur['policy_step'],cur['cumulative_reward']),
                                       (velocity,t,cur['speed_m_s']),(path,t,cur['path_error_m']),
                                       (wheel,t,cur['rear_wheel_m_s']-cur['speed_m_s'])]:
                            ax.plot(x,y,c=color,lw=1.1,label=f'α={alpha:g}')
                            if not row['C10']:
                                ax.plot(x[-1],y[-1],marker='x',color=color,ms=5)
                        velocity.plot(t,cur['rear_wheel_m_s'],c=color,ls=':',lw=.8)
                        if setting!='nominal':
                            nominal=base[alpha]
                            for ax,x,y in [(xy,nominal['xy_m'][:,0],nominal['xy_m'][:,1]),
                                           (per,nominal['policy_step'],nominal['reward']),
                                           (cum,nominal['policy_step'],nominal['cumulative_reward'])]:
                                ax.plot(x,y,c=color,ls='--',alpha=.67,lw=.85)
                        arrays.update({f'a{alpha:g}_{key}':value for key,value in cur.items()})
                    xy.set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
                    xy.legend(fontsize=7)
                    per.set(xlabel='Policy step (20 ms)',ylabel='Logged reward',yscale='symlog')
                    cum.set(xlabel='Policy step (20 ms)',ylabel='Cumulative reward',yscale='symlog')
                    velocity.axhline(2.6,c='.3',ls='--',lw=.8)
                    velocity.set(xlabel='Time (s)',ylabel='Forward speed (m/s)',title='Body (solid), wheel proxy (dotted)')
                    path.set(xlabel='Time (s)',ylabel='Signed path error (m)')
                    wheel.set(xlabel='Time (s)',ylabel='Wheel proxy − body speed (m/s)')
                    for ax in axes.flat:
                        ax.grid(alpha=.19)
                    title=f'H75 · {setting} · ridge {center:g} m, offset {ds:+d} cm · group {group}'
                    fig.suptitle(title+'\n'+('Dashed XY and reward curves: same-slot nominal contact model. ' if setting!='nominal' else '')+
                                 '× marks early termination.',fontsize=11)
                    stem=f'{setting}_s{center:g}_ds{ds:+d}_g{group}'
                    np.savez_compressed(folder/(stem+'.npz'),**arrays)
                    fig.savefig(folder/(stem+'.png'),dpi=120)
                    plt.close(fig)
                    images.append(f'[{setting}](figures/{stem}.png)')
                gallery.append(f'| {center:g} | {ds:+d} | {group} | '+' · '.join(images)+' |')
    return rows,gallery


def pair_nominal_contacts(rows):
    """Orient each observed paired change from this sweep's nominal contact to the variant."""
    keys=('ds_cm','center_m','alpha','group')
    nominal={tuple(r[k] for k in keys):r for r in rows if r['setting']=='nominal'}
    pairs=[]
    by_cell={}
    by_setting={}
    for r in rows:
        if r['setting']=='nominal':
            continue
        base=nominal[tuple(r[k] for k in keys)]
        if any(r[k]!=base[k] for k in ('slot','origin','terrain_sha256')):
            raise ValueError(f"Unequal physical slot or mesh in {r['output']} and {base['output']}")
        pair={k:r[k] for k in ('setting',*keys,'slot','origin','terrain_sha256')}
        pair.update(nominal_output=base['output'],variant_output=r['output'],
                    nominal_terminal=base['terminal'],variant_terminal=r['terminal'],
                    nominal_end_s=base['observed_s'],variant_end_s=r['observed_s'])
        for label in ('C10','final_hold'):
            pair['nominal_'+label]=base[label]
            pair['variant_'+label]=r[label]
        pairs.append(pair)
        key=tuple(r[k] for k in ('setting','ds_cm','center_m','alpha'))
        by_cell.setdefault(key,[]).append(pair)
        by_setting.setdefault(r['setting'],[]).append(pair)

    def count(group):
        counts={'n':len(group)}
        for label in ('C10','final_hold'):
            for before,after in ((0,0),(0,1),(1,0),(1,1)):
                counts[f'{label}_{before}{after}']=sum(
                    (p['nominal_'+label],p['variant_'+label])==(before,after) for p in group)
            assert sum(counts[f'{label}_{cell}'] for cell in ('00','01','10','11'))==len(group)
        return counts

    cells=[]
    for (setting,ds,center,alpha),group in sorted(by_cell.items()):
        if sorted(p['group'] for p in group)!=[401,402,403,404]:
            raise ValueError(f'Expected each of four physical groups once in {(setting,ds,center,alpha)}')
        cells.append(dict(setting=setting,ds_cm=ds,center_m=center,alpha=alpha,**count(group)))
    return pairs,cells,[dict(setting=k,**count(v)) for k,v in by_setting.items()]


def analyze(root):
    from itertools import pairwise

    spec=read(root/'protocol.json')
    lanes={(r['setting'],r['center_m'],r['group'],r['ds_cm'],r['alpha']):r for r in read(root/'lanes.json')}
    if len(lanes)!=spec['budget']['physical_episodes'] or len(lanes)!=648:
        raise ValueError('The protocol requires 648 unique case rows.')
    for scene in spec['scenes']:
        for job in read(Path(scene)/'jobs.json'):
            verify_job(job)
    folder=root/'analysis'
    bundle=Path(spec['bundle'])
    with np.load(bundle/'route.npz') as trajectory:
        route=trajectory['xy'][trajectory['s']<=29]
    table,gallery=plot_cases(folder/'figures',lanes,read(bundle/'config.json'),route)
    by_case={(r['setting'],r['center_m'],r['group'],r['ds_cm'],r['alpha']):r['C10'] for r in table}
    summaries=[]
    cells=[]
    readme=['# Contact sensitivity on the paired development panel','',
            '[Actual and reference XY, rewards and speeds](#trajectory-gallery) · '
            '[648 episode records](episodes.csv) · [Full window and final hold by cell](cells.csv) · '
            '[Paired flips](paired_flips.csv) · [Nominal replay comparison](../nominal_replay_audit/)','',
            'Each row below uses groups 401–404 at 2.6 m/s on the 10 cm oblique ridge. '
            'A reversal requires a strict advantage for α=0 at 3.9 m and for α=1 at 4.5 m. '
            'The position comparison uses 0 versus +1 cm at all three positions and preferences (36 pairs). '
            'The preference comparison uses the two adjacent half-unit changes at the nominal centers (24 pairs). '
            'The seven-offset, five-preference historical grid has a different denominator and preference spacing.','',
            '| Setting | 3.9 m, α=0 / 1 (each of 4) | 4.5 m, α=0 / 1 (each of 4) | Reversal | Position flips / 36 | Preference flips / 24 | Rate ratio |',
            '|---|---:|---:|---|---:|---:|---:|']
    for setting in SETTINGS:
        def state(c,g,ds,a):
            return by_case[setting,c,g,ds,a]
        nf_s=sum(state(c,g,0,a)!=state(c,g,1,a) for c in CENTERS for g in range(401,405) for a in ALPHAS)
        nf_a=sum(state(c,g,0,a)!=state(c,g,0,b) for c in CENTERS for g in range(401,405) for a,b in pairwise(ALPHAS))
        p0,p1,q0,q1=(sum(state(c,g,0,a) for g in range(401,405)) for c in (3.9,4.5) for a in (0.,1.))
        ratio=(nf_s/36)/(nf_a/24) if nf_a else None
        reversal=p0>p1 and q0<q1
        summaries.append(dict(setting=setting,shifted_pairs=36,shifted_flips=nf_s,adjacent_pairs=24,adjacent_flips=nf_a,
                              frequency_ratio=ratio,c39_a0=p0,c39_a1=p1,c45_a0=q0,c45_a1=q1,reversal=reversal))
        rate='undefined' if ratio is None else f'{ratio:.3f}'
        readme.append(f'| {setting} | {p0} / {p1} | {q0} / {q1} | {reversal} | {nf_s} | {nf_a} | {rate} |')
        for ds in (0,1):
            for c in CENTERS:
                for a in ALPHAS:
                    subset=[r for r in table if (r['setting'],r['ds_cm'],r['center_m'],r['alpha'])==(setting,ds,c,a)]
                    cells.append(dict(setting=setting,ds_cm=ds,center_m=c,alpha=a,groups=len(subset),
                                      full=sum(r['C10'] for r in subset),hold=sum(r['final_hold'] for r in subset)))
    pairs,paired_cells,paired_settings=pair_nominal_contacts(table)
    if (len(pairs)!=576 or len(paired_cells)!=144 or len(paired_settings)!=8 or
            any(r['n']!=72 for r in paired_settings)):
        raise ValueError('Incomplete contact-versus-nominal comparison')
    readme+=['','## Contact models compared with the nominal model','',
             'Each setting is paired to the nominal run at the same ridge, displacement, preference, '
             'physical group, world origin and clone slot. Each cell contains four physical groups. '
             'The columns give nominal-to-variant outcomes in the order 0→0, 0→1 (gain), 1→0 (loss), 1→1. '
             'C10 denotes the full 10 s observation window. Final hold uses the final-second tracking criteria. '
             'The same 72 nominal runs serve as references for all eight settings.','',
             '[576 individual pairs and source trajectories](nominal_contact_pairs.csv) · '
             '[Four-group counts by displacement, ridge center and preference](nominal_contact_cells.csv) · '
             '[Setting totals](nominal_contact_settings.csv)','',
             '| Contact setting | Paired runs | C10 (00 / 01 / 10 / 11) | Final hold (00 / 01 / 10 / 11) |',
             '|---|---:|---:|---:|']
    for counts in paired_settings:
        summaries4=[' / '.join(str(counts[f'{label}_{cell}']) for cell in ('00','01','10','11'))
                    for label in ('C10','final_hold')]
        readme.append(f"| {counts['setting']} | {counts['n']} | {summaries4[0]} | {summaries4[1]} |")
    support=[]
    readme+=['','## Rear support at the first front reload','',
        'This post hoc description uses the archived 200 Hz event rule. Search starts when original-route '
        'progress first reaches the actual ridge center minus 0.10 m. Six consecutive front-wheel resultant '
        'force magnitudes below 1 N followed by a sample at or above 1 N define the first reload. '
        'The support fraction counts rear-wheel force magnitudes strictly above 1 N in that sample and the '
        'next seven, truncated at the last valid force sample. A partial terminal interval retains its actual '
        'trajectory endpoint and contributes no later force observation. Net contact force magnitude includes '
        'all vector components. A geometric crest crossing is recorded separately and can follow the reload.','',
        'The 16-run subset repeats the historical endpoint design on the new physical slots '
        '(3.9 and 4.5 m, unshifted, α=0 and 1, four groups). The full panel adds the middle placement, '
        'preference and shifted ridge. Each count here is descriptive. Early/any counts all early endpoints '
        'with a positive rear-support fraction, including intermediate fractions. '
        'Intermediate means greater than zero and less than one half.','',
        '[All episode counts, denominators and event times](episodes.csv) · '
        '[Counts by support class and full-window outcome](rear_support_by_setting.csv)','',
        '| Setting | Cohort | Reload / runs | Early / any support | Full / zero support | Intermediate | No event |',
        '|---|---|---:|---:|---:|---:|---:|']
    for setting in SETTINGS:
        all_rows=[r for r in table if r['setting']==setting]
        endpoint=[r for r in all_rows if r['ds_cm']==0 and r['center_m'] in (3.9,4.5) and r['alpha'] in (0.,1.)]
        for label,group,n in [('endpoint16',endpoint,16),('all72',all_rows,72)]:
            assert len(group)==n
            counts=dict(setting=setting,cohort=label,n=n,reload=sum(r['front_reload_identified'] for r in group))
            for code,key in [(0,'early'),(1,'full')]:
                for cls in ('none','intermediate','half_or_more','no_event'):
                    counts[f'{key}_{cls}']=sum(r['C10']==code and r['rear_support_class']==cls for r in group)
            counts['early_any']=counts['early_intermediate']+counts['early_half_or_more']
            counts['intermediate']=counts['early_intermediate']+counts['full_intermediate']
            counts['no_event']=counts['full_no_event']+counts['early_no_event']
            assert counts['reload']+counts['no_event']==n
            support.append(counts)
            readme.append(f"| {setting} | {label} | {counts['reload']}/{n} | {counts['early_any']} | "
                          f"{counts['full_none']} | {counts['intermediate']} | {counts['no_event']} |")
    for name,records in [('episodes',table),('cells',cells),('paired_flips',summaries),
                         ('rear_support_by_setting',support),('nominal_contact_pairs',pairs),('nominal_contact_cells',paired_cells),
                         ('nominal_contact_settings',paired_settings)]:
        with (folder/(name+'.csv')).open('w',newline='') as stream:
            out=csv.DictWriter(stream,fieldnames=list(records[0]))
            out.writeheader()
            out.writerows(records)
    readme+=['','## Trajectory gallery','',
             'Each image shows the three preferences, original XY route, 20 ms logged reward including termination, '
             'cumulative reward, signed path error, body speed and rear-wheel speed proxy. Dashed XY and reward traces '
             'use the same preference and physical slot under this sweep’s nominal contact model. '
             'The Zero lower-stack runs archived from other batch layouts have no matching twelve-slot trace in this design. '
             'Each image has a companion NPZ with 200 Hz reward allocations, signed components, actual step IDs and speeds. '
             'The episodes table links every original trajectory, contract and status directory.','',
             '| Center (m) | Offset (cm) | Physical group | Plots |','|---:|---:|---:|---|',*gallery]
    (folder/'README.md').write_text('\n'.join(readme)+'\n')
    subprocess.run([sys.executable,str(REPO/'docs/evidence/pcrp_matched_controls/revision/audit_contact_replay.py'),str(root)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','analyze'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--e2',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    if args.action=='prepare':
        prepare(root,args.e2.resolve())
    elif args.action=='run':
        try:
            run(root)
        except Exception as exc:
            state=read(root/'status.json')
            write(root/'status.json',dict(state,stage='error',error=str(exc)))
            notice(root,'error',f"STTW 第一章接触模型敏感性运行异常，见 {root}/status.json")
            raise
    else:
        analyze(root)


if __name__ == "__main__":
    main()
