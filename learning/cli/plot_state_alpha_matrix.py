"""All 25 schedules, two terrains, original task metrics; six cases per cell."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;rows=json.loads((r/'analysis_summary.json').read_text())['rows'];alphas=[0,.25,.5,.75,1]
fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
for i,terrain in enumerate(['dense_corridor','single_10cm']):
 for j,key in enumerate(['final_hold','A','B','C']):
  ax=axes[i,j];matrix=np.zeros((5,5),int)
  for y,pre in enumerate(alphas):
   for x,post in enumerate(alphas):
    ms=[m for m in rows if m['terrain']==terrain and m['label']==f'{pre:g}_to_{post:g}'];assert len(ms)==6
    matrix[y,x]=sum(m[key] if key=='final_hold' else m['tasks'][key] for m in ms)
  im=ax.imshow(matrix,vmin=0,vmax=6,cmap='YlGnBu')
  for y in range(5):
   for x in range(5):ax.text(x,y,str(matrix[y,x])+'/6',ha='center',va='center',color='white' if matrix[y,x]>=4 else '#222')
  ax.set(xticks=range(5),xticklabels=alphas,yticks=range(5),yticklabels=alphas,xlabel='Alpha after switch',ylabel=('Gravel' if i==0 else '10 cm ridge')+' / alpha before switch',title='Final hold' if key=='final_hold' else 'Task '+key)
fig.suptitle('Natural alpha schedules: same start, independent pre-switch trajectories',fontsize=15)
fig.savefig(r/'schedule_matrix.png',dpi=170)
