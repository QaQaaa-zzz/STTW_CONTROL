"""P1 paired factor contrasts and complete fixed-preference task counts."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args()
r=json.loads((a.evaluation/'gravel_task_metrics.json').read_text());e=json.loads((a.evaluation/'gravel_events.json').read_text())['lanes'];ph=json.loads((a.evaluation/'gravel_phase_metrics.json').read_text())
terrains=['S1','S2','S3','S4','S5','S6'];methods=['alpha0','alpha025','alpha05','alpha075','alpha1','Zero'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8B6500','#555555']
lines=['# P1 fixed embedded-pebble factor separation (DEV)','','Actual/reference XY and paired reward panels for all60 conditions: [RESULTS](RESULTS.md). Original trajectories stop at actual failures. One fixed map seed, five paired physical parameter groups, two speeds; no independent training seeds or holdout claim.','','|Terrain|Method|Survived/10|Task A/10|Task B/10|Task C/10|','|---|---|---:|---:|---:|---:|']
for t in terrains:
 for m in methods:
  g=[x for x in r if x['terrain']==t and x['method']==m]
  lines.append('|'+ '|'.join([t,m,str(sum(not x['failed'] for x in g))]+[str(sum(x['task_'+k] for x in g)) for k in 'ABC'])+'|')
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,k in zip(axes,'ABC'):
 for i,(m,c) in enumerate(zip(methods,colors)):
  ax.plot(range(6),[sum(x['task_'+k] for x in r if x['terrain']==t and x['method']==m) for t in terrains],'-o',color=c,label=m)
 ax.set(xticks=range(6),xticklabels=terrains,ylim=(-.4,10.4),xlabel='Terrain',ylabel='Qualified outcomes /10',title='Task '+k);ax.grid(alpha=.2)
axes[-1].legend(fontsize=8);fig.suptitle('P1 DEV | complete tasks, not survival | fixed H75 + orange Local300');fig.savefig(a.evaluation/'task_by_preference.png',dpi=150);plt.close(fig)
key=lambda x:(x['terrain'],x['speed_m_s'],x['parameter_seed'],x['method'])
merged={key(x):dict(x) for x in r}
for x in e:merged[key(x)].update(front_events_n=len(x['front_events']),front_low_load_s=x['front_low_load_s'],overlap_ratio=x['overlap_ratio'])
for x in ph:
 q=x['phase_metrics'].get('post_clear_1s');merged[key(x)]['post_clear_1s_roll_rms_rad']=None if q is None else q['roll_rms_rad'];merged[key(x)]['post_clear_1s_observed_s']=0 if q is None else q['observed_s']
fields=['front_events_n','front_low_load_s','overlap_ratio','post_clear_1s_roll_rms_rad'];labels=['Front raised-contact episodes','Front low load (s)','Front/rear raised-contact overlap','Post-clear1s roll RMS (rad)']
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for ax,field,label in zip(axes.flat,fields,labels):
 for j,(m,c) in enumerate(zip(methods,colors)):
  for i,t in enumerate(terrains):
   g=[v for v in merged.values() if v['terrain']==t and v['method']==m and v[field] is not None and (field!='post_clear_1s_roll_rms_rad' or v['post_clear_1s_observed_s']>=.995)]
   vals=[v[field] for v in g];x=i+(j-2.5)*.12
   ax.scatter(x+np.linspace(-.025,.025,len(vals)),vals,s=10,color=c,alpha=.5)
   if vals:ax.scatter(x,np.mean(vals),marker='_',s=100,color=c,label=m if i==0 else None)
 ax.set(xticks=range(6),xticklabels=terrains,xlabel='Terrain',ylabel=label);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('P1 DEV | dots: every observed condition; bars: descriptive means\nContact window0.8–5s; post-clear full1s only, incomplete windows remain in JSON');fig.savefig(a.evaluation/'contact_factor_metrics.png',dpi=150);plt.close(fig)
contrasts=[]
for left,right in [('S1','S2'),('S2','S3'),('S4','S2'),('S2','S5'),('S2','S6')]:
 for v in sorted(merged.values(),key=key):
  if v['terrain']!=left:continue
  other=merged[(right,v['speed_m_s'],v['parameter_seed'],v['method'])]
  delta={field:None if v[field] is None or other[field] is None else other[field]-v[field] for field in fields+['path_rmse_m','speed_rmse_m_s']}
  contrasts.append(dict(left=left,right=right,speed_m_s=v['speed_m_s'],parameter_seed=v['parameter_seed'],method=v['method'],right_minus_left=delta,tasks={k:dict(left=v['task_'+k],right=other['task_'+k]) for k in 'ABC'},failed_left=v['failed'],failed_right=other['failed']))
(a.evaluation/'paired_factor_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
lines+=['','![Task counts](task_by_preference.png)','','![Contact and motion summaries](contact_factor_metrics.png)','','Density: S1→S2→S3 has more actual contact episodes and fewer qualified outcomes overall. This is evidence for these nested layouts, not a universal density law. Height: S5(2cm) is outside current demonstrated task capability; no alpha solves it. Clustering: S6 vs S2 changes task rankings and contact patterns despite equal stone attributes, but is not uniformly harder. Task A on S6:alpha.25=6/10; taskB:alpha.75=5/10. S2 taskB:alpha.75=7/10. No universal best alpha.','','These are matched fixed-preference DEV results. They support further phase experiments, not proof that a phase selector improves deployment. Short failed observation windows are not compared as complete task successes; conditional motion means must be read with their observed denominators. Native episode counts include brief contact interruptions, not distinct stones. All raw paired differences and failures are in paired_factor_contrasts.json; incomplete recovery windows are retained in gravel_phase_metrics.json.','','Next: a separate extended-route S7 spacing experiment. Do not mix its rankings with the original-route P1; no PPO.']
(a.evaluation/'CONCLUSION.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(conditions=len(r),paired_contrasts=len(contrasts),figures=2)))
