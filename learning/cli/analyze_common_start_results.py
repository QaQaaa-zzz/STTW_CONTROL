import argparse
import numpy as np,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();r=args.run;routes=np.load(r/'after_0/reference_routes.npz');rows=[]
for al in ['0','0.25','0.5','0.75','1']:
 ds={m:np.load(r/f'{m}_{al}/trajectory.npz') for m in ['before','after']}
 for lane,rid in enumerate(ds['before']['route_ids']):
  ts={m:d['lower_trace'][:,:,lane].reshape(-1,11) for m,d in ds.items()};mask=(ts['before'][:,6]>0)&(ts['after'][:,6]>0);v=float(routes['speed'][rid])
  for m,d in ds.items():
   t=ts[m];ids=np.flatnonzero(t[:,6]);failed=bool(d['failure'][:,lane].any());signals=[];when=None
   if failed:
    step=int(np.flatnonzero(d['failure'][:,lane])[0]);when=(step+1)*.02
    if d['domain_exit'][step,lane]:signals.append('domain')
    if d['policy_fault'][step,lane]:signals.append('policy_fault')
    if (t[4*step:4*step+4,4]>.7).any():signals.append('roll_limit')
    if not signals:signals.append('unknown')
   goal=routes['goal'][rid];gxy=np.array([np.interp(goal,routes['s'],routes['xy'][rid,:,j]) for j in range(2)]);heading=np.interp(goal,routes['s'],routes['heading'][rid]);ps=np.unique(ids//4);pose=d['pose'][ps,lane,:2];signed=(pose-gxy)@np.array([np.cos(heading),np.sin(heading)])
   rows.append(dict(model=m,alpha=float(al),lane=lane,route=int(rid),terrain=int(d['terrain'][lane]),failed=failed,signals=signals,failure_s=when,goal=bool(((signed>=0)&(d['progress'][ps,lane]>=goal)).any()),entered=bool(d['terrain_entered'][lane]),contacted=bool(d['terrain_contacted'][lane]),cleared=bool(d['terrain_cleared'][lane]),common_seconds=float(mask.sum()*.005),path_rmse=float(np.sqrt(np.mean(t[mask,3]**2))),speed_rmse=float(np.sqrt(np.mean((t[mask,2]-v)**2))),speed_bias=float(np.mean(t[mask,2]-v)),return_sum=float(d['reward'][ps,lane].sum())))
for m in ['before','after']:
 a=[x for x in rows if x['model']==m];print(m,{'complete':sum(not x['failed'] for x in a),'signals':dict(collections.Counter(v for x in a for v in x['signals'])),'goal':sum(x['goal'] for x in a),'contact':sum(x['contacted'] for x in a),'cleared':sum(x['cleared'] for x in a)});print({k:np.mean([x[k] for x in a]) for k in ['path_rmse','speed_rmse','speed_bias']})
for al in [0,.25,.5,.75,1]:
 print('alpha',al,[(m,sum(not x['failed'] for x in rows if x['model']==m and x['alpha']==al),*[round(np.mean([x[k] for x in rows if x['model']==m and x['alpha']==al]),4) for k in ['path_rmse','speed_rmse','speed_bias']]) for m in ['before','after']])
b={(x['lane'],x['alpha']):x for x in rows if x['model']=='before'};a={(x['lane'],x['alpha']):x for x in rows if x['model']=='after'}
print('rescued',[k for k in b if b[k]['failed'] and not a[k]['failed']]);print('lost',[k for k in b if not b[k]['failed'] and a[k]['failed']]);print('improved', {k:sum(a[i][k]<b[i][k] for i in b) for k in ['path_rmse','speed_rmse']});print('early precontact',[(m,sum(x['failed'] and not x['entered'] for x in rows if x['model']==m)) for m in ['before','after']])
for lane in range(13):
 print('case',lane,'terrain',b[lane,0]['terrain'],[(al,round(b[lane,al]['path_rmse'],2),round(a[lane,al]['path_rmse'],2),b[lane,al]['failure_s'],a[lane,al]['failure_s']) for al in [0,.5,1]])
(r/'plots/paired_outcomes.json').write_text(json.dumps(rows,indent=2))
