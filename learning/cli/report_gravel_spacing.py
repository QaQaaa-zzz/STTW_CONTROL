"""Describe matched P2 phase evidence without assuming axle-spacing resonance."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();root=a.evaluation
rows=json.loads((root/'gravel_task_metrics.json').read_text());ev=json.loads((root/'gravel_events.json').read_text())['lanes'];cl=json.loads((root/'cluster_first_contacts.json').read_text());names=['gap0p2','gap0p4','gap0p8'];gaps=[.2,.4,.8];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8B6500'];alphas=[0,.25,.5,.75,1]
key=lambda x:(x['speed_m_s'],x['parameter_seed'],x['alpha'])
lookup={(x['terrain'],key(x)):x for x in cl};common=[key(x) for x in cl if x['terrain']==names[0] and all(lookup[(n,key(x))]['pre_cluster_state'][2] is not None for n in names)]
summary=[]
for n,gap in zip(names,gaps):
 group=[x for x in rows if x['terrain']==n];events=[x for x in ev if x['terrain']==n];paired=[lookup[(n,k)]['pre_cluster_state'][2] for k in common]
 summary.append(dict(terrain=n,spacing_m=gap,n=len(group),survived=sum(not x['failed'] for x in group),tasks={k:sum(x['task_'+k] for x in group) for k in 'ABC'},mean_contact_overlap=float(np.mean([x['overlap_ratio'] for x in events])),paired_third_cluster_n=len(paired),paired_third_cluster_abs_roll_rate_mean=float(np.mean([abs(x['roll_rate_rad_s']) for x in paired])),paired_third_cluster_abs_steer_rate_mean=float(np.mean([abs(x['steering_rate_rad_s']) for x in paired]))))
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for i,(alpha,color) in enumerate(zip(alphas,colors)):
 for ax,field,label in zip(axes,['overlap_ratio','roll_rate_rad_s','steering_rate_rad_s'],['Loaded raised-contact overlap','Pre-third-cluster |roll rate| (rad/s)','Pre-third-cluster |steer rate| (rad/s)']):
  means=[]
  for x,n in enumerate(names):
   if field=='overlap_ratio':vals=[e[field] for e in ev if e['terrain']==n and e['method']==['alpha0','alpha025','alpha05','alpha075','alpha1'][i]]
   else:vals=[abs(lookup[(n,k)]['pre_cluster_state'][2][field]) for k in common if k[2]==alpha]
   means.append(np.mean(vals));ax.scatter(np.full(len(vals),gaps[x])+(i-2)*.004,vals,s=12,color=color,alpha=.4)
  ax.plot(gaps,means,'-o',color=color,label=f'alpha{alpha:g}');ax.set(xlabel='Cluster spacing (m)',ylabel=label,xticks=gaps);ax.grid(alpha=.2)
axes[-1].legend(fontsize=8);fig.suptitle('P2 separate-route DEV | raw paired conditions and descriptive means\nThird-cluster state uses shared contacted-condition intersection, no failure padding');fig.savefig(root/'spacing_phase_response.png',dpi=150);plt.close(fig)
(root/'spacing_conclusion.json').write_text(json.dumps(dict(by_spacing=summary,shared_third_contact_conditions=[list(k) for k in common],role='Matched DEV conditional physical response, not a proved resonance law'),indent=2)+'\n')
lines=['# P2 separate-route spacing results (DEV)','','[All actual/reference XY and reward panels](RESULTS.md)','','|Spacing(m)|Survived/50|Task A/50|Task B/50|Task C/50|Mean loaded overlap|','|---|---:|---:|---:|---:|---:|']
for x in summary:lines.append('|'+ '|'.join([str(x['spacing_m']),str(x['survived'])]+[str(x['tasks'][k]) for k in 'ABC']+[f"{x['mean_contact_overlap']:.4f}"])+'|')
lines+=['','![Actual phase and preceding response](spacing_phase_response.png)','','0.4m has greater average loaded front/rear raised-contact overlap in this fixed layout;0.8m has no such overlap in the recorded data. Physical phase changes are observable. Third-cluster residual components are reported on the identical contacted-condition subset in all layouts; raw points and unavailable contacts are retained.','','There is no evidence here for a universal axle-spacing worst-case law: task counts are small, no spacing is uniformly worst across alpha/task, and these are five frozen physical parameter groups on one map. The extended R1.8 route itself produces larger path errors and is a distinct protocol. Do not mix its ranking with original-route P1.','','Keep the measured contact timing and motion results; do not add compensation or train new networks based on an assumed resonance. Next P3 isolates entry versus recovery preference on the original P1 route with a same-start, no-snapshot cohort.']
(root/'CONCLUSION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
