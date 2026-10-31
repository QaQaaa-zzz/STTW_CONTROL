"""Merge cached alpha0 and new fixed-alpha capability evidence without simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import digest
from analyze_native_goal_confirmation import render_paired_load_diagnostics


def combine(candidate,baseline,output):
    roots=[candidate.resolve(),baseline.resolve()]
    plans=[json.loads((root/'plan.json').read_text()) for root in roots]
    for field in ('models','config','initial_source','replanning','horizon_s'):
        if plans[0][field]!=plans[1][field]:raise ValueError('cached identity differs: '+field)
    for root in roots:
        receipt=json.loads((root/'analysis/receipt.json').read_text())
        if receipt['plan_sha256']!=digest(root/'plan.json') or json.loads((root/'evaluation/status.json').read_text())['phase']!='completed':raise ValueError('complete authenticated analysis required')
    rows=json.loads((candidate/'analysis/results.json').read_text())
    keys={(r['bank_index'],r['route'],r['force_N'],r['onset_s'],r['duration_s']) for r in rows}
    cached=[r for r in json.loads((baseline/'analysis/results.json').read_text()) if r['alpha']==0. and (r['bank_index'],r['route'],r['force_N'],r['onset_s'],r['duration_s']) in keys]
    origins={r['case_id']:candidate for r in rows}
    if set(origins)&{r['case_id'] for r in cached}:raise ValueError('overlapping cached/new cases')
    origins.update({r['case_id']:baseline for r in cached});rows+=cached
    raw={};links=[]
    for row in rows:
        if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('trace identity changed')
        with np.load(row['raw_file']) as z:raw[row['case_id']]=dict(z)
    output.mkdir(exist_ok=True)
    for index,key in enumerate(sorted(keys)):
        group=sorted([r for r in rows if (r['bank_index'],r['route'],r['force_N'],r['onset_s'],r['duration_s'])==key],key=lambda r:r['alpha'])
        if len(group)!=5 or [r['alpha'] for r in group]!=[0.,.25,.5,.75,1.]:raise ValueError('exactly five unique alpha cases required')
        if plans[0]['protocol']['bank_state_fingerprints'][str(key[0])]!=plans[1]['protocol']['bank_state_fingerprints'][str(key[0])]:raise ValueError('cached complete bank state differs')
        paths=[]
        for row in group:
            with np.load(origins[row['case_id']]/'evaluation'/(row['path_key']+'_path.npz')) as z:paths.append(dict(z))
        for path in paths[1:]:
            for name in ('s','xy','heading','curvature','goal'):np.testing.assert_array_equal(path[name],paths[0][name])
        for row in group[1:]:
            for name in ('speed','original_goal_xy'):np.testing.assert_equal(row[name],group[0][name])
        path=paths[0];fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
        axes[0].plot(*path['xy'][path['s']<=path['goal']].T,'k:',label='Original reference')
        max_progress=max(float(raw[r['case_id']]['path_progress'][:r['active_lower_ticks']].max()) for r in group)
        after=(path['s']>path['goal'])&(path['s']<=max(float(path['goal']),max_progress)+1.)
        axes[0].plot(*path['xy'][after].T,color='gray',ls=':',label='Reference after goal')
        axes[0].scatter(*group[0]['original_goal_xy'],marker='*',color='black')
        for row,color,style in zip(group,['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00'],['-','--','-.',':','-']):
            d=raw[row['case_id']];n=row['active_lower_ticks'];t=d['time'][:n];label=f"alpha={row['alpha']:g}";xy=d['actual_xy'][:n]
            axes[0].plot(*xy.T,color=color,ls=style,label=label);axes[0].scatter(*xy[-1],color=color,marker='x' if row['endpoint']['physical_outcome']=='contact' else 's')
            axes[1].plot(t,d['actual_forward_speed'][:n],color=color,ls=style,label=label);axes[2].plot(t,np.rad2deg(d['phi'][:n]),color=color,ls=style,label=label)
        axes[0].set(xlabel='World X (m)',ylabel='World Y (m)');axes[0].set_aspect('equal')
        axes[1].set(xlabel='Task time (s)',ylabel='True speed (m/s)');axes[2].set(xlabel='Task time (s)',ylabel='Roll (deg)')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        for ax in axes[1:]:ax.axvspan(key[3],key[3]+key[4],color='gray',alpha=.2)
        fig.suptitle(f'{key[1]} | bank{key[0]} | {key[2]:+g}N | U170/lower200 | fixed from start, actual endpoints')
        stem=f'five_alpha_{index:02d}';fig.savefig(output/(stem+'_xy.png'),dpi=140);plt.close(fig);links.append(stem)
        components=json.loads((candidate/'evaluation/manifest.json').read_text())['reward_components']
        render_paired_load_diagnostics(output,group,raw,components,stem)
    summary={str(a):dict(n=sum(r['alpha']==a for r in rows),arrivals=sum(r['alpha']==a and r['endpoint']['goal_arrived'] for r in rows),survived=sum(r['alpha']==a and r['endpoint']['physical_outcome']=='survived' for r in rows)) for a in [0.,.25,.5,.75,1.]}
    (output/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows,new_cases=len(rows)-len(cached),reused_cases=len(cached)),indent=2)+'\n')
    (output/'README.md').write_text('# Fixed five-alpha capability at failed low-speed conditions\n\n'+'\n\n'.join(f'![All five actual/reference XY]({stem}_xy.png)\n\n[Reward/error traces]({stem}_traces.png) · [Reward components]({stem}_components.png)' for stem in links)+'\n\n'+json.dumps(summary,indent=2)+'\n\nDiagnostic development:16 new,4 authenticated cached alpha0 traces. All fixed from task start; not same-state late intervention. Rewards use each executed preference and must not be ranked across alpha; contact, original-goal and physical errors are separate. No new simulation from this report.\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('baseline',type=Path);p.add_argument('output',type=Path);a=p.parse_args();combine(a.candidate,a.baseline,a.output)
