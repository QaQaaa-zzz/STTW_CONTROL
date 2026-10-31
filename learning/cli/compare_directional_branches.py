"""Plot held-alpha suffixes and recorded online continuations from one source."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root, noq):
    plan=json.loads((root/'plan.json').read_text())
    rows=json.loads((root/'analysis/results.json').read_text())
    source=Path(plan['source']);protocol=plan['protocol'];g=protocol['source_groups'][0]
    online=json.loads((source/'evaluation/case_index.json').read_text())
    no_risk=json.loads((noq/'evaluation/case_index.json').read_text()) if noq else []
    assert len(rows)==5*len(protocol['forces_N']) and len(protocol['source_groups'])==1
    out=root/'analysis';summary=[];links=[]
    with np.load(source/'evaluation'/(g['path_key']+'_path.npz')) as z:path=dict(z)
    markers=['o','s','^','D','v'];colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']
    for force in sorted(protocol['forces_N']):
        group=sorted([r for r in rows if r['force_N']==force],key=lambda r:r['candidate_alpha'])
        streams=[]
        for k,r in enumerate(group):
            with np.load(r['raw_file']) as z:raw=dict(z)
            streams.append((f"Hold alpha={r['candidate_alpha']:g}",raw,colors[k],markers[k],'-'))
            summary.append(dict(alpha=r['candidate_alpha'],force_N=force,primary=r['primary']['label_status'],h10=r['horizons']['10'],active_ticks=int(raw['active_tick'].sum())))
        for name,cases,color,style in [('Full online',online,'black','--'),('No-risk online',no_risk,'#666666',':')]:
            if not cases:continue
            c=next(c for c in cases if c['route']==g['route'] and c['force_N']==force)
            with np.load(c['raw_file']) as z:raw={k:v[g['expected_task_tick']:] for k,v in dict(z).items()}
            streams.append((name,raw,color,None,style))
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        xy,speed,roll,reward=axes.flat
        xy.plot(*path['xy'][path['s']<=path['goal']].T,'k:',label='Original reference')
        goal=np.asarray(group[0]['original_goal_xy']);xy.add_patch(plt.Circle(goal,1.,fill=False,color='gray',ls=':'));xy.scatter(*goal,marker='*',color='k')
        fig2,rewaxes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
        for name,raw,color,marker,style in streams:
            n=int(raw['active_tick'].sum());t=raw['time'][:n];kw=dict(color=color,ls=style,label=name,lw=1.2,marker=marker,markevery=100,markersize=3)
            xy.plot(*raw['actual_xy'][:n].T,**kw)
            failure=bool(np.any(raw['contact_failure'][:n]|raw['policy_fault'][:n]|raw['nonfinite'][:n]|raw['lower_fault'][:n]))
            xy.scatter(*raw['actual_xy'][n-1],color=color,marker='x' if failure else 's',s=25)
            speed.plot(t,raw['actual_forward_speed'][:n],**kw);roll.plot(t,np.rad2deg(raw['phi'][:n]),**kw)
            reward.plot(t,raw['scored_tick_reward'][:n],**kw)
            rewaxes[0].plot(t,raw['scored_tick_reward'][:n],**kw);rewaxes[1].plot(t,np.cumsum(raw['scored_tick_reward'][:n]),**kw)
        xy.set(xlabel='World X (m)',ylabel='World Y (m)',title='Actual suffixes / original goal');xy.set_aspect('equal');xy.legend(fontsize=7)
        for ax,label in [(speed,'True speed (m/s)'),(roll,'Roll (deg)'),(reward,'Per-step reward'),(rewaxes[0],'Per-step reward'),(rewaxes[1],'Cumulative suffix reward')]:
            ax.set(xlabel='Task time (s)',ylabel=label);ax.axvspan(group[0]['onset_s'],group[0]['onset_s']+group[0]['duration_s'],color='gray',alpha=.2)
        for ax in [*axes.flat,*rewaxes]:ax.grid(alpha=.2)
        title=f"U170 / localU200 | bank{g['bank_index']} | same source task{g['source_task_s']:g}s | world-Y {force:+g}N\nFive held candidates and recorded online continuations; original H10 / 10Hz reconnection"
        fig.suptitle(title);fig2.suptitle(title+'\nRecorded user-alpha0 rewards; no smoothing; failure endpoints retained');rewaxes[0].legend(fontsize=8)
        name=f'direction_{force:+g}_xy_dynamics.png';fig.savefig(out/name,dpi=150);fig2.savefig(out/f'direction_{force:+g}_rewards.png',dpi=150);plt.close(fig);plt.close(fig2);links.append(name)
    (out/'directional_comparison.json').write_text(json.dumps(dict(rows=summary,scope='Held candidates from one complete source; online continuations may return alpha later. Recorded rewards, not independent component reconstruction.'),indent=2)+'\n')
    (out/'DIRECTIONAL.md').write_text('# Same-state directional candidate comparison\n\n'+'\n\n'.join(f'![All five alpha actual/reference trajectories]({name})' for name in links)+'\n\nOriginal 1m goal disk; x marks failure, squares mark observed endpoints. Each force direction contains all five held candidates plus the declared recorded online continuations. Reward plots use user-alpha0 logged per-step and cumulative suffix rewards, not cross-alpha rescoring.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--noq',type=Path)
    args=parser.parse_args();main(args.root,args.noq)
