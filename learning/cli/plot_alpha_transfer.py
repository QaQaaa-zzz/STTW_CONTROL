"""Show every paired task-condition difference against the calibrated fixed alpha."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
a=p.parse_args()
d=json.loads((a.evaluation/'alpha_transfer.json').read_text())
alpha=[0.,.25,.5,.75,1.]
labels=[];values=[]
for case in d['cases']:
 labels.append(f"{case['terrain'].replace('oblique_', '').replace('_',' ')} / {case['speed_m_s']:g} m/s")
 v=[]
 for f in case['folds']:
  i=alpha.index(f['calibrated_fixed_alpha'])
  v.append([100*(f['held_cost']/f['calibrated_fixed_cost']-1),f['held_speed_rmse']-f['all_alpha_speed_rmse'][i],f['held_path_rmse']-f['all_alpha_path_rmse'][i]])
 values.append(v)
values=np.array(values)
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
 fig,axes=plt.subplots(1,3,figsize=(13,9),sharey=True,layout='constrained')
 titles=['A  Task cost change (%)','B  Speed RMSE change (m/s)','C  Path RMSE change (m)']
 colors=['#333333','#0072B2','#A44700']
 for k,ax in enumerate(axes):
  for i in range(len(labels)):
   y=i+np.linspace(-.17,.17,5)
   ax.scatter(values[i,:,k],y,s=25,facecolors='none',edgecolors=colors[k],linewidths=.9)
   ax.scatter(values[i,:,k].mean(),i,marker='D',s=30,c=colors[k],zorder=3)
  ax.axvline(0,color='.5',ls='--',lw=.9)
  for boundary in [4.5,9.5,14.5]:ax.axhline(boundary,color='.8',lw=.8)
  ax.set_title(titles[k],loc='left');ax.set_xlabel('Conditional choice minus calibrated fixed choice')
  ax.grid(axis='x',alpha=.15);ax.set_yticks(range(len(labels)),labels)
 axes[0].invert_yaxis()
 fig.suptitle('Original H75 + Local300 | offline alpha selection across parameter repeats\nFixed 10 s; 4 repeats select, 1 scores; all 20 conditions retained (DEV)\nOpen circles: five paired folds; diamonds: arithmetic mean; negative = lower error/cost',fontsize=12)
 fig.savefig(a.evaluation/'alpha_transfer_comparison.png',dpi=180,facecolor='white')
 plt.close(fig)
(a.evaluation/'alpha_transfer_figure.json').write_text(json.dumps(dict(source='alpha_transfer.json',script='learning/cli/plot_alpha_transfer.py',labels=labels,columns=['cost_delta_percent','speed_rmse_delta_m_s','path_rmse_delta_m'],values=values.tolist(),interpretation='Each row is five dependent leave-one-parameter-repeat-out comparisons. No uncertainty interval or independent-test claim. Open circles are offset only vertically. Diamonds are means of paired changes; mean relative cost changes per row are not the relative change of the overall mean cost. Negative values favor conditional choice for that metric. Fixed candidates are selected using the same four seeds across all conditions.',output='alpha_transfer_comparison.png',size_inches=[13,9],dpi=180,journal_requirements='not specified; review figure'),indent=2))
print(a.evaluation/'alpha_transfer_comparison.png')
