"""P3 interaction matrices and paired gains against unchanged preference."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();r=json.loads((a.evaluation/'recovery_task_metrics.json').read_text());audits=json.loads((a.evaluation/'prefix_audit.json').read_text());entries=[0,.5,1];targets=[0,.25,.5,.75,1]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');matrices={}
for ax,k in zip(axes,'ABC'):
 matrix=np.array([[sum(x['task_'+k] for x in r if x['entry_alpha']==entry and x['recovery_alpha']==target and x['prefix_accepted']) for target in targets] for entry in entries]);matrices[k]=matrix.tolist();ax.imshow(matrix,vmin=0,vmax=5,cmap='viridis',aspect='auto')
 for i in range(3):
  for j in range(5):ax.text(j,i,str(matrix[i,j])+'/5',ha='center',va='center',color='white' if matrix[i,j]<3 else 'black')
 ax.set(xticks=range(5),xticklabels=targets,yticks=range(3),yticklabels=entries,xlabel='Recovery alpha',ylabel='Entry alpha',title='Task '+k)
fig.suptitle('P3 S6,2.6m/s DEV | same-start prefixes verified |0.10s ramp after rear clearance');fig.savefig(a.evaluation/'entry_recovery_tasks.png',dpi=150);plt.close(fig)
gains=[]
for entry in entries:
 for seed in [101,102,103,104,105]:
  group=[x for x in r if x['entry_alpha']==entry and x['parameter_seed']==seed];fixed=next(x for x in group if x['recovery_alpha']==entry)
  for x in group:
   gains.append(dict(entry_alpha=entry,recovery_alpha=x['recovery_alpha'],parameter_seed=seed,prefix_accepted=x['prefix_accepted'],task_gains={k:int(x['task_'+k])-int(fixed['task_'+k]) for k in 'ABC'},path_rmse_delta_m=x['path_rmse_m']-fixed['path_rmse_m'],speed_rmse_delta_m_s=x['speed_rmse_m_s']-fixed['speed_rmse_m_s'],failed=x['failed'],fixed_failed=fixed['failed']))
(a.evaluation/'paired_recovery_gains.json').write_text(json.dumps(gains,indent=2)+'\n');lines=['# P3 same-start entry/recovery result (DEV)','','[All actual/reference XY and reward panels](RESULTS.md)','','![Complete interaction matrices](entry_recovery_tasks.png)','','All15 cohorts forked and passed predeclared prefix criteria. Maximum pre-fork position/speed/roll/observation/executed-command differences are all exactly zero in the saved arrays. No state snapshots were restored. All75 trajectories and8 failures retained.','','One additional Task A qualified condition appears:entry0.5/recovery0, parameter101, compared with unchanged0.5. No additional Task B or C qualified conditions. Higher recovery alpha after entry0 generally increases path error while only slightly improving speed error; after entry1, intermediate recovery can reduce path RMSE without increasing Task B qualified count. Physical error changes do not automatically establish task benefit.','','Do not expand P3 to300 on this evidence. Keep the negative and modest positive results. P4 can still test pre-entry task/terrain selection because P1 has task discrimination; recovery-stage selection is not adopted. The first P4 test freezes capability tables before independent map layouts, without PPO.']
(a.evaluation/'CONCLUSION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(prefix_accepted_groups=sum(x['prefix_accepted'] for x in audits),additional_task_contexts=dict(A=1,B=0,C=0),expand_P3=False)))
