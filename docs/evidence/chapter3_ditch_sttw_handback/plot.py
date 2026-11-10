"""Plot saved matched physical replays; no simulation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_ditch_random_contactfix450')
out=Path(__file__).parent
old=np.load(base/'eval50/update50/trajectory.npz')
new=np.load(base/'handoff50/checked/update50/trajectory.npz')
n=int(np.searchsorted(old['time'],1.46,side='right'))
assert n==73 and np.array_equal(old['root'][:n],new['root'][:n])
a=json.loads((base/'eval50/randomized_properties.json').read_text())
b=json.loads((base/'handoff50/checked/randomized_properties.json').read_text())
assert a==b
fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for d,label,color in [(old,'update50: jump only','#0072B2'),(new,'update50: handback to H75','#D55E00')]:
 r=d['root'][:,0];t=d['time'];rew=d['reward'][:,0]
 axs[0,0].plot(r[:,0],r[:,1],label=label,color=color);axs[0,0].plot(*r[-1,:2],'x',color=color)
 axs[0,1].plot(t,r[:,2],label=label,color=color)
 axs[1,0].plot(t,rew,label=label,color=color)
 axs[1,1].plot(t,np.cumsum(rew),label=label,color=color)
axs[0,0].axvspan(3.3,3.6,color='0.8');axs[0,0].axhline(0,color='k',ls=':',label='original route')
axs[0,0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY; crosses mark true stops');axs[0,0].set_aspect('equal',adjustable='box')
for ax,title,y in [(axs[0,1],'Measured root height','z (m)'),(axs[1,0],'Per-step reward (terminal penalty included)','Reward / step'),(axs[1,1],'Cumulative reward','Reward')]:
 ax.axvline(1.46,color='k',ls=':',label='handoff 1.46 s');ax.set(xlabel='Time (s)',ylabel=y,title=title)
for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Same update50 / map26 / seed20261109; 73-step identical prefix\nSingle DEV replay: both fail full continuation; handback stops on ground domain exit')
fig.savefig(out/'xy_and_rewards.png',dpi=160)
(out/'verification.json').write_text(json.dumps(dict(identical_prefix_steps=n,max_root_difference=0,randomized_properties_equal=True),indent=2))
