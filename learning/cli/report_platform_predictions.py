"""Plot the two frozen forecasts against measured platform departures and export paired errors."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from run_playground_task_reversal import write_rows


def report(o,d):
    d.mkdir(exist_ok=True,parents=True)
    def rows(f):
     with f.open() as s:return list(csv.DictReader(s))
    all_scores=[];errors=[]
    fig,axs=plt.subplots(8,4,figsize=(15,12),layout='constrained')
    for j,sk in enumerate(('source','rb_run1')):
     g=o/sk; f=dict(np.load(g/'forecast_xy.npz'))
     ground=rows(g/'oracle_task_reversal.csv')
     by={(r['lane'],r['departure']):r for r in ground if r['bin']=='near' and r['task']=='left'}
     for i,(lane,dep) in enumerate(zip(f['lane'],f['departure'])):
      r=by[str(lane),dep];ax=axs[int(r['root']),2*j+int(r['approach']=='right')]
      st=int(r['branch_tick'])*.005
      with np.load(r['trajectory']) as z:
       tr=z['trace'];base=tr[tr[:,0]>=st-.0005]
       xy=z['published_reference_xy']
      tt=st+np.arange(f['phase2'].shape[1])*float(f['dt'])
      good=tt<=base[-1,0]+1e-6
      obs=np.stack([np.interp(tt[good],base[:,0],base[:,k]) for k in (1,2)],1)
      ax.plot(*xy.T,':',color='.66',lw=.55);ax.plot(*base[:,1:3].T,color='black',lw=.9)
      for model,color in [('kinematic','#ce6727'),('phase2','#197495')]:
       forecast=f[model][i,good];ax.plot(*forecast.T,color=color,lw=.65)
       error=np.linalg.norm(forecast-obs,axis=1)
       end={'skill':sk,'root':r['root'],'approach':r['approach'],'departure':dep,'method':model,'observed_s':round(base[-1,0]-st,4)}
       for t in (.24,.5,1,2):
        k=round(t/float(f['dt']));end['XY_error_m_at_'+str(t)+'s']=error[k] if k<len(error) else ''
       errors.append(end)
      ax.set(xlim=(10,19),ylim=(-1.1,1.1),aspect='equal');ax.axvline(13.3,lw=.45,color='gray')
      if int(r['root'])==0:ax.set_title(sk+' / '+r['approach'])
      if j==0 and r['approach']=='left':ax.set_ylabel('root '+r['root']+' (m)')
      if int(r['root'])==7:ax.set_xlabel('world x (m)')
     for r in rows(g/'offline_false_accepts.csv'):
      src=by[r['lane'],r['departure']];all_scores.append(dict(skill=sk,root=src['root'],approach=src['approach'],**r))
    fig.suptitle('Actual XY (black), kinematic (orange), frozen dynamics (blue), requested route (dotted) | clipped to observed time')
    fig.savefig(d/'xy.png',dpi=120);plt.close(fig)
    write_rows(d/'comparison.csv',rows(o/'comparison.csv'));write_rows(d/'xy_errors.csv',errors)
    byroot=defaultdict(list)
    for r in all_scores:byroot[r['skill'],r['root'],r['method'],r['margin']].append(r)
    out=[]
    for (sk,root,method,margin),rr in byroot.items():
     out.append(dict(skill=sk,root=root,method=method,margin=margin,contracts=len(rr),actual_pass=sum(x['actual_pass']=='True' for x in rr),
      false_accept=sum(x['false_pass']=='True' for x in rr),false_reject=sum(x['false_reject']=='True' for x in rr)))
    write_rows(d/'per_root.csv',out)
    state={}
    for sk in ('source','rb_run1'):
     cap=o.parent/(sk+'_capture')
     schema=json.loads((cap/'model_state_schema.json').read_text())
     for r in json.loads((cap/'status.json').read_text())['results']:
      with np.load(cap/r['case_dir']/'trajectory.npz') as z:
       if r['handoff_s'] is None:continue
       k=round(float(r['handoff_s'])/.005)+1
       tr=z['trace'][k];state[sk,str(r['route'])]=[tr[1],tr[2],tr[6],tr[7]*np.cos(tr[6])+tr[8]*np.sin(tr[6]),tr[4],z['qpos'][k,schema['steer_qpos']],z['qvel'][k,schema['steer_dof']]]
    pairs=[]
    for sk in ('source','rb_run1'):
     for approach in ('left','right'):
      for dep in ('left','straight','right'):
       for group in ('near','mid','far'):
        for side in ('left','right'):
         rr=[r for r in all_scores if (r['skill'],r['approach'],r['departure'],r['bin'],r['task'],r['margin'])==(sk,approach,dep,group,side,'shared_dev')]
         dyn={r['lane']:r for r in rr if r['method']=='phase2'}
         for a,b in combinations((r for r in rr if r['method']=='kinematic'),2):
          if a['actual_pass']==b['actual_pass']:continue
          delta=np.abs(np.subtract(state[sk,a['lane']],state[sk,b['lane']]))
          pairs.append(dict(skill=sk,approach=approach,departure=dep,bin=group,task=side,
            root_a=a['root'],root_b=b['root'],actual_a=a['actual_pass'],actual_b=b['actual_pass'],
            kinematic_a=a['predicted_pass'],kinematic_b=b['predicted_pass'],dynamic_a=dyn[a['lane']]['predicted_pass'],dynamic_b=dyn[b['lane']]['predicted_pass'],
            delta_xy_m=np.hypot(*delta[:2]),delta_heading_deg=np.rad2deg(delta[2]),delta_speed_m_s=delta[3],
            delta_roll_deg=np.rad2deg(delta[4]),delta_steer_deg=np.rad2deg(delta[5]),delta_steer_rate_rad_s=delta[6],
            distance_score=np.linalg.norm(delta[:4]/[.1,.1,np.deg2rad(5),.2])))
    if pairs:
     write_rows(d/'opposite_outcomes.csv',sorted(pairs,key=lambda r:r['distance_score']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions',type=Path)
    p.add_argument('destination',type=Path)
    args=p.parse_args()
    report(args.predictions,args.destination)
