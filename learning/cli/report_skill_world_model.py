"""All-case true-entry prediction XY panels and checkpoint learning curves."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path);p.add_argument('--run',type=Path,action='append',default=[]);p.add_argument('--compare-predictions',type=Path);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.data/'manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='DEV'];allcases=[];summaries=[];pages=[]
folders=a.run or sorted(a.models.glob('seed*'))
comparison=np.load(a.compare_predictions) if a.compare_predictions else None
fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
for folder in folders:
 seed=folder.name;training=folder/'training' if (folder/'training').exists() else folder
 log=json.loads((training/'log.json').read_text());best=json.loads((training/'best.json').read_text());summary=json.loads((folder/'full_segment/summary.json').read_text());summaries.append(dict(seed=seed,**summary))
 x=[r['update'] for r in log];axes[0].plot(x,[r['dev_score'] for r in log],label=f'Seed {seed}')
 axes[1].plot(x,[r['metrics']['50']['xyz_mean_m'] for r in log],label=f'Seed {seed}')
 axes[2].plot(x,[r['metrics']['50']['phase_error_fraction'] for r in log],label=f'Seed {seed}')
 pred=np.load(folder/'full_segment/predictions.npz');cases=json.loads((folder/'full_segment/cases.json').read_text());allcases.extend(dict(seed=seed,**r) for r in cases)
 for name in sorted({r['map'] for r in rows}):
  subset=[r for r in rows if r['map']==name];panel,axs=plt.subplots(6,6,figsize=(16,13),layout='constrained')
  if len(subset)>36:raise ValueError('Map contains more than36 cases; explicit pagination required')
  for ax,r in zip(axs.flat,subset):
   key=r['map']+'_'+r['case_dir'];actual=pred[key+'_actual'];estimate=pred[key+'_predicted']
   if comparison is not None:
    prior=comparison[key+'_predicted'];ax.plot(prior[:,0],prior[:,1],':',color='#555555',label='Before adaptation')
   ax.plot(actual[:,0],actual[:,1],color='#0072B2',label='Actual');ax.plot(estimate[:,0],estimate[:,1],'--',color='#D55E00',label='Free prediction');ax.scatter(actual[-1,0],actual[-1,1],s=10,marker='X',color='#0072B2');ax.scatter(estimate[-1,0],estimate[-1,1],s=10,marker='X',color='#D55E00');ax.axvspan(10.3,10.6,color='#56B4E9',alpha=.12);ax.set_title(r['case_dir'].replace('rb_run3_',''),fontsize=8);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World x (m)',fontsize=7);ax.set_ylabel('World y (m)',fontsize=7);ax.tick_params(labelsize=7)
  axs[0,0].legend(fontsize=6);panel.suptitle(f'{name}, seed {seed}, checkpoint {best["update"]}: {len(subset)} DEV true-entry RB sequences\nActual and free prediction; X=record end; travel +x; no future state feedback; not full-chain planning')
  filename=f'{name}_{seed}_xy.png';panel.savefig(a.output/filename,dpi=150);plt.close(panel);pages.append(dict(file=filename,run=seed,map=name,cases=len(subset)))
for ax,label in zip(axes,['DEV selection loss (within this run)','1s window mean XYZ error (m)','1s window phase error fraction']):ax.set(xlabel='Optimizer update',ylabel=label);ax.grid(alpha=.2)
axes[0].legend();fig.savefig(a.output/'learning_curves.png',dpi=180);plt.close(fig)
(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
with (a.output/'cases.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=allcases[0]);writer.writeheader();writer.writerows(allcases)
(a.output/'pages.json').write_text(json.dumps(pages,indent=2)+'\n')
(a.output/'README.md').write_text('# Full-segment conditional prediction DEV assessment\n\n'+
 '\n'.join(f'- [{r["run"]} {r["map"]}: {r["cases"]} actual/predicted XY]({r["file"]})' for r in pages)+
 '\n\n[Unsmoothed checkpoint curves](learning_curves.png). Selection-loss definitions can differ across runs; do not compare their absolute values across changed losses.\n\n'+
 '|Run|Checkpoint|Cases|Mean max XYZ m|Worst XYZ m|Missing handoff|\n|---|---:|---:|---:|---:|---:|\n'+
 '\n'.join(f'|{r["seed"]}|{r["update"]}|{r["cases"]}|{r["mean_max_xyz_m"]:.5f}|{r["worst_max_xyz_m"]:.5f}|{r["missing_predicted_handoff"]}/{r["cases"]}|' for r in summaries)+
 '\n\nKnown DEV, not independent TEST. Every declared case is shown; X marks record end. Actual entry initializes prediction; future history is predicted. Actual record duration only defines evaluation, never the model input. XY displays are prediction diagnostics, not new controller executions or planning benefit. Source coordinates remain in each run/full_segment/predictions.npz; source runs: '+str([str(f) for f in folders])+'; optional comparison: '+str(a.compare_predictions)+'\n')
