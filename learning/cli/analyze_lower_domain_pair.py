"""Plot all frozen paired TEST cases without smoothing or outcome exclusions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');a=p.parse_args();root=Path(a.run)
results=json.loads((root/'results.json').read_text());names=list(results);out=root/'analysis';out.mkdir(exist_ok=True)
lines=['# Frozen lower-controller physical-domain TEST','', 'Local speed/steer command task; dashed XY reference is command integration, not a geometric path objective. No smoothing. Each trace ends at physical failure or 10 seconds. Initial-state excitation ends before this observation window.','']
summary={}
for name,rows in results.items():
 summary[name]=dict(cases=len(rows),survived=sum(not x['failed'] for x in rows),mean_case_speed_rmse=float(np.mean([x['speed_rmse'] for x in rows])),mean_case_steer_rmse_rad=float(np.mean([x['steer_rmse'] for x in rows])))
 lines.append(f'- {name}: `{json.dumps(summary[name])}`')
lines+=['','## All paired trajectories and rewards','']
for i in range(len(results[names[0]])):
 ds=[dict(np.load(root/name/f'case_{i:02d}.npz')) for name in names];n=min(len(d['time']) for d in ds)
 np.testing.assert_allclose(ds[0]['limited_command'][:n],ds[1]['limited_command'][:n],atol=1e-7,rtol=0)
 fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');ax=axs.ravel()
 for name,d,color,style in zip(names,ds,['#0072B2','#D55E00'],['-','--']):
  t=d['time']+.005;xy=d['actual_xy'];ax[0].plot(*xy.T,color=color,ls=style,label=name);ax[0].scatter(*xy[-1],color=color,marker='x' if d['physical_failure'][-1] else 's')
  ax[1].plot(t,d['reward'],color=color,ls=style,label=name);ax[2].plot(t,np.cumsum(d['reward']),color=color,ls=style)
  ax[3].plot(t,d['actual_forward_speed'],color=color,ls=style);ax[4].plot(t,np.rad2deg(d['actual_delta']),color=color,ls=style);ax[5].plot(t,np.rad2deg(d['phi']),color=color,ls=style)
 d=max(ds,key=lambda x:len(x['time']));t=d['time']+.005
 ax[0].plot(*d['reference_xy'].T,':',color='black',label='command integration');ax[0].set(xlabel='World X (m)',ylabel='World Y (m)');ax[0].set_aspect('equal',adjustable='datalim');ax[0].legend()
 ax[3].plot(t,d['limited_command'][:,0],':',color='black');ax[4].plot(t,np.rad2deg(d['limited_command'][:,1]),':',color='black')
 for a,label in zip(ax[1:],['Step reward','Cumulative reward','Forward speed (m/s)','Steer (deg)','Roll (deg)']):a.set(xlabel='Task time (s)',ylabel=label);a.grid(alpha=.2)
 fig.suptitle(f'TEST domain {results[names[0]][i]["domain"]}, state {i}; local300 vs frozen lower200; command seed 930502')
 f=f'case_{i:02d}.png';fig.savefig(out/f,dpi=150);plt.close(fig);lines.append(f'![Case {i}]({f})')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
