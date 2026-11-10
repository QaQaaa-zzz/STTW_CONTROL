from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_isaac_source_entry4');o=Path(__file__).parent
f,a=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for name,label in [('evaluation','policy reference3.3'),('source_reference','source reference3.6, wait hold'),('capture_handoff','source reference3.6, capture handoff')]:
 d=np.load(b/name/'heading125/trajectory.npz');r=d['root'][:,0];t=d['time'];rew=d['reward'][:,0]
 a[0,0].plot(r[:,0],r[:,1],label=label);a[0,0].plot(*r[-1,:2],'x')
 a[0,1].plot(t,r[:,2],label=label);a[1,0].plot(t,rew,label=label);a[1,1].plot(t,np.cumsum(rew),label=label)
a[0,0].axhline(0,c='k',ls=':',label='fixed y=0 reference');a[0,0].axvspan(3.3,3.6,color='0.8');a[0,0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY');a[0,0].set_aspect('equal',adjustable='box')
for ax,title,y in [(a[0,1],'Root height','z (m)'),(a[1,0],'Per-step reward','reward/step'),(a[1,1],'Cumulative reward','reward')]:
 ax.set(xlabel='Time (s)',ylabel=y,title=title);ax.axvline(1.46,c='k',ls=':')
for ax in a.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
f.suptitle('Isaac heading125 / restored entry4 / level30cm ditch / nominal DEV; no training')
f.savefig(o/'xy_and_rewards.png',dpi=150)
