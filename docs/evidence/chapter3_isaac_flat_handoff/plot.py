from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_isaac_flat_handoff/checked')
d=np.load(b/'update50/trajectory.npz');r=d['root'][:,0];t=d['time'];v=d['reward'][:,0]
f,a=plt.subplots(1,3,figsize=(13,4),layout='constrained')
a[0].plot(r[:,0],r[:,1],label='update50 finite servo');a[0].plot(*r[-1,:2],'rx',label='domain exit');a[0].axhline(0,c='k',ls=':',label='original route');a[0].axvspan(3.3,3.6,color='0.8');a[0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY');a[0].set_aspect('equal',adjustable='box');a[0].legend(fontsize=7)
a[1].plot(t,v);a[1].set(xlabel='Time (s)',ylabel='Reward/step',title='Actual emitted reward')
a[2].plot(t,np.cumsum(v));a[2].set(xlabel='Time (s)',ylabel='Reward',title='Cumulative reward')
for ax in a[1:]:ax.axvline(1.46,c='k',ls=':',label='handoff');ax.legend()
f.suptitle('Isaac / same-height flat ditch / single DEV / failed continuation')
f.savefig(Path(__file__).parent/'xy_and_reward.png',dpi=150)
