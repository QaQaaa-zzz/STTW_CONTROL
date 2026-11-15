"""All-case true-entry prediction XY panels and checkpoint learning curves."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.data/'manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='DEV'];allcases=[];summaries=[]
fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
for seed in range(3):
 folder=a.models/f'seed{seed}';log=json.loads((folder/'log.json').read_text());best=json.loads((folder/'best.json').read_text());summary=json.loads((folder/'full_segment/summary.json').read_text());summaries.append(dict(seed=seed,**summary))
 x=[r['update'] for r in log];axes[0].plot(x,[r['dev_score'] for r in log],label=f'Seed {seed}')
 axes[1].plot(x,[r['metrics']['50']['xyz_mean_m'] for r in log],label=f'Seed {seed}')
 axes[2].plot(x,[r['metrics']['50']['phase_error_fraction'] for r in log],label=f'Seed {seed}')
 pred=np.load(folder/'full_segment/predictions.npz');cases=json.loads((folder/'full_segment/cases.json').read_text());allcases.extend(dict(seed=seed,**r) for r in cases)
 for name in sorted({r['map'] for r in rows}):
  subset=[r for r in rows if r['map']==name];panel,axs=plt.subplots(6,6,figsize=(16,13),layout='constrained')
  for ax,r in zip(axs.flat,subset):
   key=r['map']+'_'+r['case_dir'];actual=pred[key+'_actual'];estimate=pred[key+'_predicted']
   ax.plot(actual[:,0],actual[:,1],color='#0072B2',label='Actual');ax.plot(estimate[:,0],estimate[:,1],'--',color='#D55E00',label='Free prediction');ax.scatter(actual[-1,0],actual[-1,1],s=10,marker='X',color='#0072B2');ax.scatter(estimate[-1,0],estimate[-1,1],s=10,marker='X',color='#D55E00');ax.axvspan(10.3,10.6,color='#56B4E9',alpha=.12);ax.set_title(r['case_dir'].replace('rb_run3_',''),fontsize=8);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World x (m)',fontsize=7);ax.set_ylabel('World y (m)',fontsize=7);ax.tick_params(labelsize=7)
  axs[0,0].legend(fontsize=6);panel.suptitle(f'{name}, seed {seed}, checkpoint {best["update"]}: all36 DEV true-entry RB sequences\nActual and free prediction; X=record end; travel +x; no future state feedback; not full-chain planning')
  panel.savefig(a.output/f'{name}_seed{seed}_xy.png',dpi=150);plt.close(panel)
for ax,label in zip(axes,['DEV selection loss (within this run)','1s window mean XYZ error (m)','1s window phase error fraction']):ax.set(xlabel='Optimizer update',ylabel=label);ax.grid(alpha=.2)
axes[0].legend();fig.savefig(a.output/'learning_curves.png',dpi=180);plt.close(fig)
(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
with (a.output/'cases.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=allcases[0]);writer.writeheader();writer.writerows(allcases)
(a.output/'README.md').write_text('# P3 supervised full-joint model: true-entry DEV assessment\n\n[Checkpoint curves](learning_curves.png). Full data and model source: `'+str(a.models)+'`. No policy updates.\n\n'+ '\n'.join(f'- [All36 {m}, seed{s} XY]({m}_seed{s}_xy.png)' for s in range(3) for m in sorted({r['map'] for r in rows}))+'\n\n|Seed|Checkpoint|Mean full-segment max XYZ (m)|Worst XYZ (m)|Missing handoff /72|\n|---|---|---|---|---|\n'+'\n'.join(f'|{r["seed"]}|{r["update"]}|{r["mean_max_xyz_m"]:.4f}|{r["worst_max_xyz_m"]:.4f}|{r["missing_predicted_handoff"]}/72|' for r in summaries)+'\n\nAll72 DEV sequences per seed retained. Same-domain generated terrain layouts; not novel terrain types or independent TEST. Full-record duration is used only to score prediction, not as model input. Controller/RB histories are learned estimates after initialization. This report does not establish planning utility or P3 completion.\n')
