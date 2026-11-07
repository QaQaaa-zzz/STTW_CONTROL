import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
rows=[];fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
for ax,p in zip(axs.flat,sorted(r.glob('*/batch_results.json'))):
 b=json.loads(p.read_text());route=None
 for c in b['cases']:
  d=Path(c['output']);cfg=json.loads((d/'config.json').read_text());res=json.loads((d/'results.json').read_text())['cases'][0]
  a=np.load(d/'trajectory.npz');t=a['trace'][:,0][a['valid'][:,0]];route=np.load(d/'ground_route.npz')['xy'];delta=np.diff(route,axis=0)
  u=((t[:,None,1:3]-route[None,:-1])*delta).sum(-1)/(delta*delta).sum(-1)
  error=np.linalg.norm(t[:,None,1:3]-route[None,:-1]-u.clip(0,1)[...,None]*delta,axis=-1).min(-1)
  v=c['candidate'];gain=v['rear_speed_feedback_gain'];reached=res['ground_handoff_reached'];corridor=bool((error<=1).all())
  row=dict(**v,origin=p.parent.name,reached=reached,corridor_intact=corridor,corridor_arrival=reached and corridor,max_path_error_m=float(error.max()),path_rmse_m=float(np.sqrt((error**2).mean())),end_speed=float(t[-1,4]),end_roll_deg=float(np.degrees(t[-1,5])),duration=float(t[-1,0]),output=str(d));rows.append(row)
  color={0.:'tab:blue',.5:'tab:orange',1.:'tab:green'}[v['alpha']]
  ax.plot(t[:,1],t[:,2],color=color,ls='-' if gain else '--',alpha=1 if v['speed_m_s']==2 else .45,lw=1.2)
  ax.scatter(t[-1,1],t[-1,2],c=color,marker='o' if reached else 'x',s=20)
 ax.plot(route[:,0],route[:,1],'k:',lw=1);ax.axvline(2,color='gray',lw=.6);ax.set(title=p.parent.name.split('_')[-1],xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xlim(-7,4);ax.set_ylim(-10,3)
fig.suptitle('Same rough approach: Kp0 dashed / Kp1 solid; alpha 0 blue, .5 orange, 1 green\n2m/s opaque / 1.8m/s faint; circle=handoff, cross=stopped; original reference black')
fig.savefig(r/'xy_comparison.png',dpi=160)
s={str(k):dict(n=sum(x['rear_speed_feedback_gain']==k for x in rows),arrivals=sum(x['reached'] for x in rows if x['rear_speed_feedback_gain']==k),corridor_arrivals=sum(x['corridor_arrival'] for x in rows if x['rear_speed_feedback_gain']==k)) for k in [0,1]}
(r/'comparison.json').write_text(json.dumps(dict(summary=s,cases=rows),indent=2));print(json.dumps(s))
