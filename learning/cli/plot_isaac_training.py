"""Show every sampled alpha trajectory and reward; not a paired policy evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
history=json.loads((a.run/'history.json').read_text())
for row in history:
    update=row['update'];route=np.load(Path(row['scored_policy'])/'route.npz')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));axes[0].plot(*route['xy'].T,'k--',label='Original reference')
    for ep in row['episodes']:
        alpha=ep['alpha'];d=np.load(a.run/('u%03d_alpha%s.rewards.npz'%(update,alpha)))
        xy=d['xy'];label='alpha=%s (%s)'%(alpha,ep['termination'])
        line,=axes[0].plot(*xy.T,label=label);axes[0].plot(*xy[-1],marker='x',color=line.get_color())
        steps=np.arange(1,len(d['reward'])+1)
        axes[1].plot(steps,d['reward'],color=line.get_color(),label=str(alpha));axes[2].plot(steps,np.cumsum(d['reward']),color=line.get_color())
    axes[0].set(xlim=(-.5,10),ylim=(-1,10),xlabel='World X [m]',ylabel='World Y [m]',aspect='equal');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Policy step (50 Hz)',ylabel='Reward per policy transition');axes[2].set(xlabel='Policy step (50 Hz)',ylabel='Cumulative reward')
    fig.suptitle('Training samples from policy_%03d | %s | stochastic, no matched baseline\nEndpoints are actual; different batches are not a paired improvement test'%(update,['slope','rough_slope'][update]))
    fig.tight_layout();fig.savefig(a.run/('sampling_%03d.png'%update),dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(9,6));updates=[r['update'] for r in history]
for ax,(name,values) in zip(axes.flat,[('Policy loss',[r['loss_mean'][0] for r in history]),('Value loss',[r['loss_mean'][1] for r in history]),('Full-batch Gaussian KL',[r['kl'] for r in history]),('Training failure fraction',[sum(e['termination']!='budget_end' for e in r['episodes'])/len(r['episodes']) for r in history])]):
    ax.plot(updates,values,'o-');ax.set(xlabel='Update',ylabel=name,xticks=updates);ax.grid(alpha=.2)
fig.suptitle('Engineering training diagnostics | terrain changes between batches\nTraining failures are not a fixed DEV recovery evaluation');fig.tight_layout();fig.savefig(a.run/'training_diagnostics.png',dpi=150);plt.close(fig)
