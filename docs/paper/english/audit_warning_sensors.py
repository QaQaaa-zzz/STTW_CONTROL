"""Replay the fixed H75 domain and margin threshold with sensor proxies at 50 Hz."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_roll_warnings import EVIDENCE
from sttw_control.roll_warning_inputs import CTRL_CFG, PHI_C, euler_roll_rate
from sttw_control.margin_theory import recovery_margin_batch


SETTINGS = (('euler_body', 0), ('proxy_clean', 0), ('proxy_noise_0ms', 0),
            ('proxy_noise_10ms', 10), ('proxy_noise_20ms', 20), ('proxy_noise_30ms', 30))
STD = np.array([.005,.02,.003,.03])  # lean rad, roll rate rad/s, steer rad, ground speed m/s
NOISE_SEED = 4053


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def scores(inputs, sample_indices):
    values = inputs[sample_indices]
    eligible = (values[:,3] >= .5) & (abs(values[:,2]) <= .8) & (abs(values[:,0]) < PHI_C)
    margins = np.full(len(inputs),np.nan)
    ids = sample_indices[eligible]
    for i in range(0,len(ids),48):
        idx=ids[i:i+48]
        margins[idx]=recovery_margin_batch(inputs[idx,3],inputs[idx,:3],CTRL_CFG)
    return margins


def batch(item):
    raw, episodes=item
    path=Path(raw)
    contract=json.loads((path/'contract.json').read_text())
    dt=contract['physics_dt']
    if (dt,contract['lower_dt'],contract['policy_dt'])!=(.0002,.005,.02):
        raise ValueError(f'Unexpected archived clock at {path}')
    chassis=contract['lab_physics']['runtime_inertial']['body_names'].index('chasis')
    with np.load(path/'trajectory.npz') as z:
        m=z['measurement']
        omega=euler_roll_rate(z['body_state'][:,:,chassis])
        trace=z['lower_trace'].reshape(*m.shape[:2],11)
    rows=[]
    for episode, seq in episodes:
        lane=int(episode['lane'])
        live=np.flatnonzero(trace[:,lane,6])
        end=float(episode['end_s'])
        t=live*.005+trace[live,lane,7].astype(float)*dt
        if abs(t[-1]-end)>1e-6:
            raise ValueError(f'Endpoint time mismatch for {raw}, lane {lane}')
        if episode['endpoint']=='full':
            keep=t<=end+1e-7
        else:
            keep=t<end-1e-7
        idx=live[keep]
        t=t[keep]
        # First decision with an archived post-step observation is 20 ms. A partial
        # contact-terminal substep cannot advance the regular decision clock.
        if episode['endpoint']=='full':
            n_decisions=int(np.floor((end+1e-7)/.02))
        else:
            n_decisions=int(np.floor((end-1e-7)/.02))
        clock=np.arange(1,n_decisions+1)*.02
        source={d:np.searchsorted(t,clock-d/1000+1e-7,side='right')-1 for _,d in SETTINGS}
        ref=np.column_stack((m[idx,lane,0],omega[idx,lane],m[idx,lane,2],trace[idx,lane,2])).astype(float)
        proxy=np.column_stack((m[idx,lane,0],m[idx,lane,1],m[idx,lane,2],m[idx,lane,5]*.1)).astype(float)
        # The recorded rear encoder has the forward-positive sign convention.
        np.testing.assert_allclose(proxy[:,3],trace[idx,lane,10],rtol=1e-6,atol=1e-6)
        noisy=proxy+np.random.default_rng(np.random.SeedSequence([NOISE_SEED,seq])).normal(size=proxy.shape)*STD
        undelayed=np.unique(source[0][source[0]>=0])
        all_idx=np.unique(np.concatenate([i[i>=0] for i in source.values()]))
        margins=(scores(ref,undelayed),scores(proxy,undelayed),scores(noisy,all_idx))
        for j,(setting,delay) in enumerate(SETTINGS):
            indices=source[delay]
            avail=np.flatnonzero(indices>=0)
            margin=margins[min(j,2)][indices[avail]]
            crossing=np.flatnonzero(margin<0)
            hit=bool(len(crossing))
            first=avail[crossing[0]] if hit else None
            decision_s=float(clock[first]) if hit else None
            rows.append(dict(setting=setting,delay_ms=delay,condition=episode['condition'],group=int(episode['group']),
                alpha=float(episode['alpha']),raw=raw,lane=lane,endpoint=episode['endpoint'],terminal=episode['terminal'],
                end_s=end,noise_sequence=seq,decision_ticks=n_decisions,available_ticks=len(avail),
                eligible_ticks=int(np.isfinite(margin).sum()),alert=int(hit),first_source_s=float(t[indices[first]]) if hit else None,
                first_decision_s=decision_s,lead_ms=(end-decision_s)*1000 if hit else None,
                first_margin_rad=float(margin[crossing[0]]) if hit else None))
    return rows


def summarize(rows):
    return dict(n=len(rows),alerts=sum(r['alert'] for r in rows),
                median_lead_ms=median([r['lead_ms'] for r in rows if r['alert']]))


def median(x):
    return float(np.median(x)) if len(x) else None


def render(output,rows):
    outcomes=('roll','full','other')
    summary=[]
    by_stratum=[]
    paired=[]
    anchors={(r['setting'],r['raw'],r['lane']):r for r in rows if r['setting'] in ('euler_body','proxy_clean')}
    for setting,_ in SETTINGS:
        sub=[r for r in rows if r['setting']==setting]
        for endpoint in outcomes:
            members=[r for r in sub if r['endpoint']==endpoint]
            summary.append(dict(setting=setting,endpoint=endpoint,**summarize(members)))
            for baseline in ('euler_body','proxy_clean'):
                pairs=[(r,anchors[baseline,r['raw'],r['lane']]) for r in members]
                deltas=[r['lead_ms']-b['lead_ms'] for r,b in pairs if r['alert'] and b['alert']]
                paired.append(dict(setting=setting,reference=baseline,endpoint=endpoint,n=len(pairs),
                    gained=sum(r['alert']>b['alert'] for r,b in pairs),lost=sum(r['alert']<b['alert'] for r,b in pairs),
                    joint_alerts=len(deltas),median_paired_lead_change_ms=median(deltas)))
        for fields in [('condition',),('group',),('alpha',),('condition','group','alpha')]:
            groups=defaultdict(list)
            for r in sub:
                groups[tuple(r[k] for k in fields)].append(r)
            for key,group in groups.items():
                for outcome in outcomes:
                    by_stratum.append(dict(setting=setting,stratum='+'.join(fields),key='/'.join(map(str,key)),
                        endpoint=outcome,**summarize([r for r in group if r['endpoint']==outcome])))
    write_csv(output/'episodes.csv',rows)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'paired.csv',paired)
    write_csv(output/'strata.csv',by_stratum)
    colors=['#111111','#0879a7','#c44100','#007563','#7e5595','#a36d00']
    styles=['-','--',':','-.',(0,(6,2)),(0,(2,1,1,1))]
    labels=['Euler & body speed','IMU & rear wheel','Noisy, 0 ms','Noisy, 10 ms','Noisy, 20 ms','Noisy, 30 ms']
    fig,axes=plt.subplots(1,3,figsize=(8.8,3.15),layout='constrained')
    curves=[]
    for ax,outcome,title in zip(axes,outcomes,('Roll-limit terminations','Full-window completions','Competing terminations')):
        for (setting,_),label,color,style in zip(SETTINGS,labels,colors,styles):
            members=[r for r in rows if r['setting']==setting and r['endpoint']==outcome]
            leads=[r['lead_ms'] for r in members if r['alert']]
            # First-alert lead survival curve, counting unwarned trajectories in each denominator.
            grid=np.arange(0,501,5)
            y=[int(np.count_nonzero(np.array(leads)>=v-1e-4)) for v in grid]
            curves.extend(dict(setting=setting,endpoint=outcome,required_lead_ms=int(v),warned=n,total=len(members))
                          for v,n in zip(grid,y))
            ax.step(grid,np.array(y)*100/len(members),where='pre',label=label,c=color,ls=style,lw=1.45)
        ax.set(xlabel='Minimum warning lead (ms)',title=f'{title}\n(n = {len(members)})',xlim=(0,500))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Episodes warned (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.savefig(output/'warning_lead.pdf')
    fig.savefig(output/'warning_lead.png',dpi=180)
    plt.close(fig)
    write_csv(output/'curves.csv',curves)
    readme=['# Sensor degradation and warning lead on frozen H75 trajectories','',
            '[Warning lead curves](warning_lead.png) · [Per-episode evidence](episodes.csv) · '
            '[Paired changes](paired.csv) · [Outcome counts](summary.csv) · [Condition, preference and group breakdown](strata.csv)','',
            'All six rows replay the same 200 measured development trajectories. A warning occurs on the first '
            '50 Hz decision with nominal recovery margin below zero, speed at least 0.5 m/s, absolute steer at most '
            '0.8 rad and absolute lean below 0.7 rad. The archived post-step clock fixes every endpoint. Full cases '
            'are observed through 10 s and warnings on prematurely terminated cases must strictly precede the endpoint.','',
            'The reference uses recorded lean and steer, Euler roll derivative from chassis attitude and angular velocity, '
            'and body forward speed. The sensor proxy uses the recorded IMU-site signed roll gyro and the '
            'forward-positive measured rear axle rate multiplied by 0.1 m. Lean remains the simulated attitude channel. '
            'Gaussian standard deviations are 0.005 rad lean, 0.02 rad/s gyro, 0.003 rad steer and 0.03 m/s speed. '
            'One deterministic sequence per trajectory (NumPy seed sequence [4053, episode number]) is applied at '
            '200 Hz before imposing availability delays. Each 50 Hz decision uses the latest sample available at that time. '
            'No decision uses future samples. Missing early samples and invalid-domain inputs do not trigger alerts.','',
            '| Inputs | Roll warnings / 54 | Median lead (ms), warned roll cases | Full-window alerts / 118 | Other alerts / 28 |',
            '|---|---:|---:|---:|---:|']
    for (setting,_),label in zip(SETTINGS,labels):
        rr,ff,oo=(next(r for r in summary if (r['setting'],r['endpoint'])==(setting,o)) for o in outcomes)
        lead='undefined' if rr['median_lead_ms'] is None else f"{rr['median_lead_ms']:.1f}"
        readme.append(f"| {label} | {rr['alerts']} | {lead} | {ff['alerts']} | {oo['alerts']} |")
    readme+=['','This experiment measures measurement sensitivity of the offline detector on fixed actions and endpoints. '
             'The disturbance has one noise realization per trajectory. Sensor noise and availability delay do not alter the '
             'simulated motion. Localization drift, estimator dynamics and closed-loop vehicle response require separate experiments.']
    (output/'README.md').write_text('\n'.join(readme)+'\n')
    print('\n'.join(readme[-8:]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=EVIDENCE/'warning_comparison/episodes.csv')
    parser.add_argument('--output',type=Path,default=EVIDENCE/'sensor_replay')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    with args.input.open() as src:
        cases=[r for r in csv.DictReader(src) if r['cohort']=='H75']
    if Counter(r['endpoint'] for r in cases)!=dict(roll=54,full=118,other=28) or len({(r['raw'],r['lane']) for r in cases})!=200:
        raise ValueError('The registered H75 panel requires exactly 200 unique lanes with the original terminal labels')
    items=defaultdict(list)
    for seq,case in enumerate(cases):
        items[case['raw']].append((case,seq))
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,group in enumerate(pool.map(batch,items.items()),start=1):
            rows.extend(group)
            if n%10==0:
                print(f'Replayed {n}/{len(items)} physical cases',flush=True)
    render(args.output,rows)


if __name__=='__main__':
    main()
