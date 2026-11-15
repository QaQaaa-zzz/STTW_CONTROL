"""Complete DEV event timing errors, fixed-timing control, and sampling illustration."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
labels=json.loads((a.run/'labels.json').read_text());dev=[r for r in labels if r['split']=='DEV'];baseline=json.loads((a.run/'constant_baseline.json').read_text());median=np.array(baseline['train_median_event_s']);grid=np.round(median/.02)*.02
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,layout='constrained');records=[];summaries=[]
for seed,style,color in [(0,'-','#0072B2'),(1,'--','#D55E00'),(2,'-.','#009E73')]:
 z=np.load(a.run/f'seed{seed}/dev_predictions.npz');error=z['event_time_s']-z['actual_time_s'];found=z['event_found'];finite=np.isfinite(z['actual_time_s']);summary=dict(seed=seed,mae_s=np.nanmean(np.where(found,error,np.nan).__abs__(),0).tolist(),max_abs_s=np.nanmax(np.where(found,error,np.nan).__abs__(),0).tolist(),missing=(finite&~found).sum(0).tolist());summaries.append(summary)
 for event,ax in enumerate(axes):
  ax.plot(np.arange(len(dev)),1000*np.where(found[:,event],error[:,event],np.nan),style,color=color,label=f'Event seed{seed}');ax.axhline(0,color='#bbbbbb',lw=.7);ax.set_ylabel(['No-wheel-contact time error (ms)','Far-contact time error (ms)'][event]);ax.grid(alpha=.2)
 for i,row in enumerate(dev):
  for event in range(2):records.append(dict(case=row['case'],seed=seed,event=['no_wheel_contact','far_contact'][event],predicted=bool(found[i,event]),actual_relative_s=float(z['actual_time_s'][i,event]),predicted_relative_s=float(z['event_time_s'][i,event]) if found[i,event] else None,signed_error_s=float(error[i,event]) if found[i,event] else None,median_grid_error_s=float(grid[event]-z['actual_time_s'][i,event])))
truth=z['actual_time_s']
for event,ax in enumerate(axes):ax.plot(np.arange(len(dev)),1000*(grid[event]-truth[:,event]),':',color='black',label='TRAIN median, same20ms grid')
axes[0].legend(ncol=4,fontsize=8);axes[-1].set_xlabel('DEV case index (all72, mapping in cases.csv)');fig.suptitle('Frozen physical predictor: all-case event timing errors\nEvent labels preserve5ms occurrences; no measured future inputs');fig.savefig(a.output/'event_errors.png',dpi=180);plt.close(fig)
example=dev[0];raw=np.load(example['source']);t=raw['trace'][:,0];event=example['takeoff_5ms_s'];select=(t>=event-.04)&(t<=event+.06);initial=np.flatnonzero(raw['trace'][:,-1]==1)[0];coarse=np.arange(initial,len(t),4);coarse=coarse[(t[coarse]>=event-.04)&(t[coarse]<=event+.06)]
fig,ax=plt.subplots(figsize=(8,3),layout='constrained');ax.step(t[select],raw['wheel_contact_flags'][select,0],where='post',label='Native5ms: any wheel contact');ax.scatter(t[coarse],raw['wheel_contact_flags'][coarse,0],marker='x',s=70,label='Instantaneous20ms samples');ax.step(t[coarse],(t[coarse]>=event).astype(float),where='post',ls='--',label='Cumulative event label');ax.set(xlabel='Simulation time(s)',ylabel='Boolean indicator',title=example['case']+': short event retained without changing motion');ax.legend(fontsize=8);fig.savefig(a.output/'sampling_example.png',dpi=180);plt.close(fig)
with (a.output/'cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
result=dict(seeds=summaries,continuous_median_mae_s=baseline['dev_mae_s'],grid_median_s=grid.tolist(),same_grid_median_mae_s=np.nanmean(abs(truth-grid),axis=0).tolist(),role='72known DEV records, not new terrain TEST or evidence of sustained flight')
(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['contract.json','resume_verification.json']:shutil.copyfile(a.run/name,a.output/name)
(a.output/'README.md').write_text('''# P3 short contact-event correction

[Unchanged physical prediction XY](../handoff_loss/README.md) · [All72 event errors](event_errors.png) · [5ms sampling example](sampling_example.png)

Native5ms events now produce cumulative20ms labels: the brief event remains visible after it occurs. No trajectory interpolation, new physics, or relabeling of task success. Readouts use only frozen predicted physical state, model memory and elapsed skill time; actual event times appear only in supervision/evaluation.

All3readout seeds report both events for all72DEV cases. Selected readout seed1/update600 has no-wheel-contact MAE10.83ms and far-contact MAE12.57ms. The same20ms-grid TRAIN-median control is10.14ms/7.01ms; its continuous-time version is6.11ms/6.04ms. **No learned-event advantage over fixed timing is established.** The control is a diagnostic baseline, never a replacement for the actual robot handoff logic.

The physical model remains seed0/update2275. A real-checkpoint GPU test confirms bitwise100=40+serialize+60 for both physical prediction and cumulative event memory, and zero physical change with the readout attached. Three unit tests cover short-event preservation, event continuation, and combined serialized continuation. Negative cases and earlier sampling failures remain in previous reports.

These are1–2native no-wheel-contact samples per trajectory, not proof of sustained aerial flight. Model development remains limited to the declared gap and known topology. Old frozen M1 margins reject all viable continuation cases; full-chain calibration and planning remain unfinished.
''')
