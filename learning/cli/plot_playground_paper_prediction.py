"""Three predeclared approach examples from saved trajectories; no simulation or smoothing."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--residual',default='residual_seed2');p.add_argument('--direct',default='direct_seed0');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
names=['M0',a.residual,a.direct];colors=['#009e73','#0072b2','#d55e00'];styles=['-.','--',':'];datasets={name:dict(np.load(a.root/name/'predictions.npz')) for name in names};records=json.loads((a.root/'M0/summary.json').read_text())['cases'];selected=[]
with plt.rc_context({'font.size':7,'axes.labelsize':7,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none'}):
 fig,axes=plt.subplots(3,1,figsize=(3.4,3.1),layout='constrained')
 for ax,route,label in zip(axes,[0,3,6],['Straight approach','Left approach','Right approach']):
  i=next(i for i,r in enumerate(records) if r['map_id']==0 and r['root']==6 and r['route']==route and abs(r['request']-1.8)<1e-4);r=records[i];selected.append(r);end=r['steps']+1
  source=Path(r['evaluation']);cfg=json.loads((source/'resolved_config.json').read_text());bank=np.load(cfg['ground_handoff']['reference_path']);ref=bank['xy'][route]
  with np.load(source/r['case']/'trajectory.npz') as d:actual=d['qpos'][round(r['anchor_s']/.005):round(r['actual_terminal_s']/.005)+1,:2].copy()
  use=(ref[:,0]>=actual[:,0].min())&(ref[:,0]<=actual[:,0].max());ax.plot(ref[use,0],ref[use,1],':',color='#777777',lw=.8,label='Reference');ax.plot(actual[:,0],actual[:,1],color='black',lw=1.1,label='Actual')
  for name,color,style,short in zip(names,colors,styles,['M0','M1','M2']):
   pred=datasets[name]['predicted'][i,:end];ax.plot(pred[:,0],pred[:,1],style,color=color,lw=1.,label=short)
  ax.plot(*actual[-1],'o',color='black',ms=2);ax.set(title=label,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='outside upper center',frameon=False)
 for suffix in ['png','svg']:fig.savefig(a.output/f'approach_prediction.{suffix}',dpi=240,facecolor='white')
 plt.close(fig)
(a.output/'approach_prediction_provenance.json').write_text(json.dumps(dict(root=str(a.root),models=names,cases=selected,selection='First declared DEV map,root6,1.8m/s,one straight/left/right approach; illustrative3/24cases, not outcome-ranked',processing='Actual5ms positions cropped at real phase0 endpoint; predictions at20ms, no smoothing. Equal XY scaling. No padded samples or post-jump actual path included.',destination='Local manuscript draft, publisher final requirements pending verification',accessibility='Color plus line style; scalar metrics and full24cases remain in report',alt_text='Actual and predicted XY trajectories for straight, left and right approaches on the same development map. M0 deviates more than the learned models in these selected examples; all24cases are reported separately.'),indent=2)+'\n')
