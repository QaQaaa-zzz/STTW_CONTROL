"""Plot every saved ground-suite trajectory without extending failure endpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
d=json.loads((a.run/'suite_declaration.json').read_text());status=json.loads((a.run/'status.json').read_text());cfg=d['source_config']['ground_suite'];z=np.load(a.run/'trajectory.npz');routes=np.load(cfg['reference_path'])
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#000000'];metrics=[]
pages=[]
for case_id,case in enumerate(cfg['cases']):
 if case_id%12==0:fig,axes=plt.subplots(3,4,figsize=(16,12),layout='constrained')
 ax=axes.flat[case_id%12];ref=routes['xy'][case_id][routes['s']<=case['goal_s']+.3];ax.plot(ref[:,0],ref[:,1],'k:',label='Original reference')
 detail,axs=plt.subplots(3,1,figsize=(9,8),layout='constrained')
 for method_id,r in enumerate([v for v in status['results'] if v['case']==case_id]):
  i=r['lane'];m=z['valid'][:,i];t=z['time'][m];pose=z['pose'][m,i];c='#000000' if r['zero_middle'] else colors[method_id%5];label=r['method']
  ax.plot(pose[:,0],pose[:,1],color=c,label=label);ax.scatter(*pose[-1,:2],color=c,marker='o' if r['success'] else 'x',s=25)
  for axis,key,scale in zip(axs,['speed','ey','eh'],[1,1,180/np.pi]):axis.plot(t,z[key][m,i]*scale,color=c,label=label)
  metrics.append(dict(**r,path_rmse_m=float(np.sqrt(np.mean(z['ey'][m,i]**2))),speed_rmse_m_s=float(np.sqrt(np.mean((z['speed'][m,i]-case['speed_m_s'])**2))),peak_roll_rad=float(abs(z['measurement'][m,i,0]).max())))
 ax.set(title=case['name'],xlabel='X (m)',ylabel='Y (m)',aspect='equal');ax.grid(alpha=.2)
 if case['shape']=='straight':ax.set_ylim(-.7,.7)
 axs[0].axhline(case['speed_m_s'],color='gray',ls=':');axs[0].set_ylabel('Body speed (m/s)');axs[1].set_ylabel('Signed path error (m)');axs[2].set(ylabel='Heading error (deg)',xlabel='Time (s)')
 for axis in axs:axis.grid(alpha=.2)
 axs[0].legend(ncol=3,fontsize=8);detail.suptitle(case['name']+' | frozen H75 / ECBC+ESO; nominal DEV');detail.savefig(a.run/(case['name']+'_tracking.png'),dpi=130);plt.close(detail)
 if case_id%12==11 or case_id==len(cfg['cases'])-1:
  axes.flat[0].legend(fontsize=6);fig.suptitle('Locked pendulum STTW | all declared methods | nominal DEV\nCircles: goal hold; crosses: real failure endpoints; original reference dotted')
  name='xy_all.png' if len(cfg['cases'])<=12 else f'xy_page{case_id//12+1:02d}.png';pages.append(name)
  fig.savefig(a.run/name,dpi=150);plt.close(fig)
(a.run/'metrics.json').write_text(json.dumps(metrics,indent=2))
lines=['# Locked-pendulum STTW turning diagnostic','','\n\n'.join(f'![All actual/reference XY]({page})' for page in pages),'','Nominal DEV, frozen H75, no training, one initialization per case. All failures retained. Each episode ends at its actual termination. These are inference-only behavior tests; task reward is not implemented and no reward improvement is claimed. RMSE below uses each episode\'s actual window and is not a paired-window ranking.','','|Case|Method|End|Seconds|Path RMSE m|Speed RMSE m/s|','|---|---|---|---:|---:|---:|']
for r in metrics:lines.append(f"|{r['case_name']}|{r['method']}|{r['termination']}|{r['duration_s']:.3f}|{r['path_rmse_m']:.3f}|{r['speed_rmse_m_s']:.3f}|")
lines+=['','## Tracking plots']+[f"- [{c['name']}]({c['name']}_tracking.png)" for c in cfg['cases']]
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(f"{status['success']}/{status['total']}; {a.run/'RESULTS.md'}")
