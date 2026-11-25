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
    from sttw_control.isaac_path_reporting import replay_rewards, static_case_metrics

    raw=Path(lane['output'])
    contract=read(raw/'contract.json')
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
        # Signed components cover each valid policy transition. Keep all four lower
        # allocations so their sum reproduces the logged terminal penalty.
        lower_rows=(4*steps[:,None]+np.arange(4)).ravel()
        sig.update(reward_lower_step=lower_rows+1,reward_lower=rebuilt[lower_rows,0],
                   **{'component_'+k:v[lower_rows,0] for k,v in components.items()})
    return dict(**lane,C10=int(not metric['failed']),final_hold=int(metric['final_hold']),
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
    for name,records in [('episodes',table),('cells',cells),('paired_flips',summaries)]:
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
