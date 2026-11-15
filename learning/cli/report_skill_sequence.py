"""All-case actual/reference/full-chain prediction plots for P4 diagnostics."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--routes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.run/'cases.json').read_text());z=np.load(a.run/'predictions.npz');contract=json.loads((a.run/'contract.json').read_text());world=json.loads((Path(contract['oracle'])/'map48/world.json').read_text())
bank=np.load(a.routes)
fig,axes=plt.subplots(6,3,figsize=(15,15),layout='constrained')
for i,(row,ax) in enumerate(zip(rows,axes.flat)):
 prefix=np.load(row['prefix']);actual=prefix['qpos'][:,:2]
 if row['suffix']:
  tail=np.load(row['suffix']);mask=prefix['trace'][:,0]<tail['trace'][0,0]-1e-6;actual=np.r_[prefix['qpos'][mask,:2],tail['qpos'][:,:2]];reference=tail['reference_xy']
 reference=bank['xy'][row['route']]
 prediction=np.concatenate([z['approach'][i,:int(z['entry_steps'][i])+1,:2],z['jump'][i,:int(z['jump_steps'][i]),:2],z['departure'][i,:int(z['departure_steps'][i])+1,:2]])
 ax.plot(reference[:,0],reference[:,1],':',color='#999999',label='Original reference');ax.plot(actual[:,0],actual[:,1],color='#0072B2',label='Actual');ax.plot(prediction[:,0],prediction[:,1],'--',color='#D55E00',label='C full prediction');ax.scatter(*actual[-1],color='#0072B2',marker='X',s=15)
 for rect in world['terrain']['forbidden_rectangles']:
  x0,x1,y0,y1=rect;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#777777',alpha=.35))
 ax.axvspan(10.3,10.6,color='#56B4E9',alpha=.15);ax.set_title(f'root{row["root"]} up{row["upstream"]} down{row["downstream"]}: actual{int(row["actual_success"])} / admit{int(row["admitted"])}',fontsize=8);ax.set_xlabel('World x(m)',fontsize=8);ax.set_ylabel('World y(m)',fontsize=8);ax.set_aspect('equal',adjustable='datalim');ax.tick_params(labelsize=7);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7);fig.suptitle('Known DEV map48: all18 full-chain candidates, real initial state only\nActual stops at failure/goal; predicted stages end at predicted events/goal or max horizon; travel +x')
fig.savefig(a.output/'xy.png',dpi=160);plt.close(fig)
for name in ['summary.json','contract.json']:shutil.copyfile(a.run/name,a.output/name)
with (a.output/'cases.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
s=json.loads((a.run/'summary.json').read_text());(a.output/'README.md').write_text('# P4 first C-only full-chain diagnostic\n\n[All18 actual/reference/predicted XY](xy.png).\n\n'+f'True admit{s["true_admit"]}, false admit{s["false_admit"]}, false reject{s["false_reject"]}, true reject{s["true_reject"]};18conditions total, four actually feasible.\n\n'+s['scope']+'\n\nOnly common first-frame physical/controller/RB histories enter the predictor. Ground states and estimated history feed RB; predicted RB event state/history feed departure. Real future trajectories/status are used only for offline comparison. Initial RB support assumes the declared flat approach zone; STTW locked hip/knee rates are zero in the ground model.\n\nThis is an engineering/DEV result, not deployment or independent TEST. No new physical episodes: actual outcomes reuse the full-snapshot Oracle. G/I/Oracle, same-candidate strong backtracking, calibration and actual selected-plan verification remain required.\n')
