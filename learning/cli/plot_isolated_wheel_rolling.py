"""Plot saved isolated-wheel comparisons without additional simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
f,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');records=[]
for run in a.run:
 d=np.load(run/'trajectory.npz');t=d['time'];s=d['state'];r=json.loads((run/'results.json').read_text());records.append(r)
 for i in range(2):
  axs[i,0].plot(s[:,i,0],s[:,i,1]-s[0,i,1],label=run.name);axs[i,0].plot(s[-1,i,0],s[-1,i,1]-s[0,i,1],'x')
  axs[i,1].plot(t,s[:,i,7],label=run.name)
  q=s[:,i,3:7];axis=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:,2];tilt=np.degrees(np.arcsin(np.clip(axis[:,2],-1,1)));axs[i,2].plot(t,tilt,label=run.name)
for i,title in enumerate(('Initially rolling at 2 m/s','Initially spinning, zero translation')):
 axs[i,0].set(xlabel='World x (m)',ylabel='World y minus initial y (m)',aspect='equal',title=title);axs[i,1].set(xlabel='Time (s)',ylabel='World forward speed (m/s)');axs[i,2].set(xlabel='Time (s)',ylabel='Wheel axle tilt (deg)')
 for ax in axs[i]:ax.grid(alpha=.2)
axs[0,1].legend(fontsize=7);f.suptitle('Isolated original RB wheel: same mass, inertia, outer diameter, width and friction\nCollision representation changes; free wheel can wobble or fall — not whole-bike validation')
f.savefig(a.output/'xy_and_rolling.png',dpi=150);plt.close(f);(a.output/'summary.json').write_text(json.dumps(records,indent=2))
