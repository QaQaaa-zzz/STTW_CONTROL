"""Run matched frozen actors with native or delayed sampled feedback on two original layouts."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from pcrp_review_batches import LAB, load, write
from pcrp_contact_sensitivity import notice
from pcrp_corridor import digest

ARMS={'native':None, 'loc20_delay10':(20,.01), 'loc10_delay30':(10,.03)}
LABELS=['alpha0','alpha05','alpha1','Zero']


def prepare(root):
    source=root.parent/'pcrp_curvature_governor_20261121'
    old=load(source/'cases.json')
    template=load(source/'settings.json')
    bundle=Path('runs/h75_selected_local300_flat/bundle').resolve()
    identity=load(bundle/'identity.json')
    keys=('asset','controller_config','lower_actor','lower_sha256','plant_options','initial_state','selected_lower')
    common={k:template[k] for k in keys}
    common.update(auto_reset_horizon=False,episode_s=10,record_controller_terms=True,
                  omit_diagnostic_fields=['observation'],evaluation_role='Exploratory paired sampled-feedback DEV')
    groups=[c for c in old if Path(c['output']).name.startswith(('flat_original_','position_s4.2_')) and c['seed']<=404]
    assert len(groups)==8
    root.mkdir(parents=True,exist_ok=False)
    jobs=[]; identities={str(bundle/f):sha for f,sha in identity['files'].items()}
    for key in ('lower_actor','initial_state','asset','controller_config'):
        path=Path(common[key]).resolve();identities[str(path)]=digest(path)
    for level,setting in ARMS.items():
        out=root/level;out.mkdir()
        config=common.copy()
        if setting:
            hz,lag=setting
            config['sampled_feedback']=dict(location_hz=hz,availability_s=lag,seed=4053,
                noise_std=[.005,.02,.003,.02,.02,.3,.3,.01,.01,.002])
        cases=[]
        for c in groups:
            name=Path(c['output']).name
            case={k:c[k] for k in ('seed','randomization','user_speed_m_s','terrain_path')}
            case.update(output=str(out/name),alphas=[0.,.5,1.,0.],zero_middle=[False]*3+[True],case_labels=LABELS)
            cases.append(case)
            terrain=Path(c['terrain_path'])
            identities[str(terrain/'terrain.npz')]=load(terrain/'identity.json')['sha256']
        write(out/'settings.json',config);write(out/'cases.json',cases)
        for path in (out/'cases.json',out/'settings.json'): identities[str(path)]=digest(path)
        jobs.append(dict(config=str(out/'settings.json'),case_manifest=str(out/'cases.json'),
                         terrain=cases[0]['terrain_path'],output=str(out/'batch')))
    write(root/'jobs.json',jobs);identities[str(root/'jobs.json')]=digest(root/'jobs.json')
    write(root/'protocol.json',dict(code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        role='Closed-loop sensor degradation DEV; frozen H75 and Local300',bundle=str(bundle),
        source_cases=str(source),identities=identities,groups=[401,402,403,404],settings=list(ARMS),labels=LABELS,
        cases=['flat_original','position_s4.2'],speed_m_s=2.6,episode_s=10,dt_s=[.0002,.005,.02],
        budget=96,physical_slots=32,optimizer_updates=0,stop='Three arms once or first engineering error',
        implementation='H75, path pursuit, Local300 and ECBC/ESO receive delayed sampled signals. Signed rear encoder '
        'times radius supplies speed. Yaw is a noisy attitude-heading proxy sampled at 200 Hz. XY fixes are held '
        'between 20 or 10 Hz acquisitions. The first noisy reset fix is available at t=0 and initializes history. '
        'Sensor packets and XY updates are then delayed 10 or 30 ms. Fixed prerecorded noise generator seed per arm, '
        'identical draw sequence before sampling/delay. Config noise_std lists lean angle, gyro roll, steering '
        'angle/rate, gyro z, two wheel rates, x/y fix and yaw. Independent Gaussian perturbations are simulated '
        'estimator errors, without hardware calibration. Default group sensor zero offsets and physical delays are retained.',
        outcomes='Ground-truth route projection, forward speed, reward, terminal guards, working-roll and final-hold '
        'gates. Estimated invalid projection is recorded separately. Same-slot comparison for each method; different '
        'policy labels occupy distinct clone slots. Report full/hold/A/B/C, failure type, speed/path RMSE, '
        'truth and sensor traces, and zero-threshold warning on available samples at 50 Hz.',
        scope='Joint packet/estimator stress; original group servo gains and command delays matched. No extra actuator '
        'mismatch, map drift, dropouts or physical robot. Independent effects of noise, wheel slip and latency not identified.'))
    write(root/'status.json',dict(stage='prepared',episodes=96))


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(Path('docs/paper/english').resolve()))
    from analyze_matched_controls import check_pair,read_case
    from audit_warning_sensors import scores
    from sttw_control.isaac_path_reporting import replay_rewards
    out=root/'analysis';out.mkdir(exist_ok=True)
    scenarios=load(root/'native/cases.json')
    records=[]; links=[]
    for case in scenarios:
        name=Path(case['output']).name
        fig,axes=plt.subplots(len(ARMS),4,figsize=(15,10),layout='constrained')
        anchor=None
        for (level,setting),axs in zip(ARMS.items(),axes,strict=True):
            raw=root/level/name; contract,lanes=read_case(raw,LABELS)
            route=np.load(Path(contract['bundle'])/'route.npz')
            if anchor is None:anchor=contract
            else:
                check_pair(contract,anchor)
                assert contract['origin']==anchor['origin']
            xy,ev,rew,cum=axs
            progress_limit=max(x[3].max() for x in lanes)
            xy.plot(*route['xy'][route['s']<=progress_limit+.4].T,'k:',label='Original route')
            with np.load(raw/'trajectory.npz') as z:
                _,parts,closure=replay_rewards(z,load(Path(contract['bundle'])/'config.json'))
                np.savez_compressed(raw/'reward_components.npz',**parts)
                measured=z['policy_measurement']
                for lane,(metrics,tr,t,progress,r) in enumerate(lanes):
                    method=LABELS[lane];ids=np.flatnonzero(z['lower_trace'][:,:,lane,6].ravel())
                    imu=z['feedback_imu_tick'][ids,lane] if setting else ids+1
                    fix=z['feedback_location_tick'][ids,lane] if setting else ids+1
                    if setting:
                        q=round(setting[1]/.005);hz=round(200/setting[0])
                        np.testing.assert_array_equal(imu,np.maximum(0,ids+1-q))
                        np.testing.assert_array_equal(fix,imu-imu%hz)
                    inputs=measured[ids,lane][:,[0,1,2,5]].astype(float);inputs[:,-1]*=.1
                    clock=np.flatnonzero(((ids+1)%4==0)&(t<10.-1e-7)&(t<metrics['time_s']-1e-7))
                    margins=scores(inputs,clock); alerted=clock[margins[clock]<0]
                    first=t[alerted[0]] if len(alerted) else None
                    v_est=z['feedback_speed'][ids,lane] if setting else tr[:,2]
                    row=dict(level=level,condition=name,group=case['seed'],**metrics,
                             speed_est_rmse_m_s=float(np.sqrt(np.mean((v_est-tr[:,2])**2))),
                             sensor_pose_rmse_m=float(np.sqrt(np.mean(np.sum((z['feedback_pose'][ids,lane,:2]-tr[:,:2])**2,axis=1)))) if setting else 0.,
                             eligible=int(np.isfinite(margins).sum()),warning_s=first,
                             lead_ms=1000*(metrics['time_s']-first) if first is not None else None,reward_closure=closure)
                    records.append(row)
                    opt=dict(label=method,ls='--' if method=='Zero' else '-')
                    line,=xy.plot(*tr[:,:2].T,**opt)
                    xy.plot(*tr[-1,:2],marker='o' if metrics['full'] else 'x',color=line.get_color())
                    ev.plot(t,tr[:,2]-case['user_speed_m_s'],**opt);rew.plot(np.arange(1,len(r)+1),r,**opt)
                    cum.plot(np.arange(1,len(r)+1),np.cumsum(r),**opt)
                    if first is not None:ev.axvline(first,color=line.get_color(),lw=.65,ls=':')
            xy.set(ylabel='World y (m)',xlabel='World x (m)',aspect='equal',title=level)
            ev.set(xlabel='Time (s)',ylabel='Speed error (m/s)');rew.set(ylabel='Reward',xlabel='Policy step')
            cum.set(ylabel='Cumulative reward',xlabel='Policy step')
        axes[0,0].legend(fontsize=8);fig.suptitle(name+' | H75 + Local300 | markers show physical endpoints')
        fig.savefig(out/(name+'.png'),dpi=140);plt.close(fig)
        links.append(f'- [{name}]({name}.png)')
    with (out/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=['# Closed-loop sampled feedback','','[All 96 physical trajectories](episodes.csv)','',
             '| Terrain | Method | Observation setting | Full | Hold | A | B | C | Roll limits | Full-window alerts |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for terrain in ('flat_original','position_s4.2'):
        for label in LABELS:
            for level in ARMS:
                a=[r for r in records if r['condition'].startswith(terrain) and r['level']==level and r['method']==label]
                counts=[f'{sum(r[k] for r in a)}/{len(a)}' for k in ('full','hold','A','B','C')]
                full=[r for r in a if r['full']]
                summary.append('| '+' | '.join([terrain,label,level,*counts,f"{sum('roll_limit' in r['terminal'] for r in a)}/{len(a)}",f"{sum(r['warning_s'] is not None for r in full)}/{len(full)}"])+' |')
    (out/'README.md').write_text('\n'.join(summary+['',*links])+'\n')


def run(root):
    spec=load(root/'protocol.json')
    for f,h in spec['identities'].items():
        if digest(Path(f))!=h:raise ValueError(f'Frozen input changed: {f}')
    write(root/'status.json',dict(stage='running',episodes=spec['budget']))
    notice(root,'start','PCRP WP5 闭环传感退化开始，三种设置各32条同槽轨迹。')
    with (root/'simulation.log').open('x') as log:
        subprocess.run([LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',spec['bundle'],
                        '--seconds','10','--resident-jobs',str(root/'jobs.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
    for level in ARMS:
        if load(root/level/'batch/status.json')['policy_steps']!=500:raise ValueError(f'Incomplete batch {level}')
    write(root/'status.json',dict(stage='physics_complete',episodes=96))
    analyze(root)
    write(root/'status.json',dict(stage='complete',episodes=96))
    notice(root,'complete',f'PCRP WP5 三种闭环反馈的96条轨迹已完成，见 {root}/analysis/README.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','analyze'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve()
    if a.action=='prepare':prepare(root)
    elif a.action=='analyze':analyze(root)
    else:
        try:run(root)
        except Exception as exc:
            write(root/'status.json',dict(stage='error',error=str(exc)))
            notice(root,'error',f'PCRP WP5 闭环评测异常，见 {root}/status.json')
            raise
