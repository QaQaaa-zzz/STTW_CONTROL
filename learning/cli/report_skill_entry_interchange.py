"""Offline true-entry intervention evidence; never a causal planner result."""
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
for name in ['baseline','interventions','routes','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
paths={'C':a.baseline,**{k:a.interventions/k for k in ['physical','history','both']}}
records={k:json.loads((v/'cases.json').read_text()) for k,v in paths.items()};arrays={k:np.load(v/'predictions.npz') for k,v in paths.items()};indices={k:{(r['root'],r['route']):i for i,r in enumerate(v)} for k,v in records.items()};bank=np.load(a.routes)
fig,axes=plt.subplots(4,3,figsize=(18,7),layout='constrained');errors=[]
for row,ax in zip(records['both'],axes.flat):
 key=(row['root'],row['route']);prefix=np.load(row['prefix']);tail=np.load(row['suffix']);actual=np.r_[prefix['qpos'][prefix['trace'][:,0]<tail['trace'][0,0]-1e-6,:2],tail['qpos'][:,:2]];ref=bank['xy'][row['route']]
 ax.plot(ref[:,0],ref[:,1],':',color='.6',label='Reference');ax.plot(actual[:,0],actual[:,1],color='#0072B2',label='Actual');ax.scatter(*actual[-1],marker='x',s=15,color='#0072B2');tick=int(np.flatnonzero(prefix['trace'][:,-1]==2)[0])
 for name,color,style in [('C','#D55E00','--'),('physical','#CC79A7','-.'),('history','#009E73',':'),('both','black','--')]:
  i=indices[name][key];z=arrays[name]
  for j,(stage,length) in enumerate([('approach',int(z['entry_steps'][i])+1),('jump',int(z['jump_steps'][i])),('departure',int(z['departure_steps'][i])+1)]):
   points=z[stage][i,:length,:2];ax.plot(points[:,0],points[:,1],style,color=color,lw=.9,label=name if j==0 else None)
  pred=z['jump'][i,int(z['jump_steps'][i])-1];r=records[name][i]
  errors.append(dict(method=name,root=row['root'],route=row['route'],actual_success=row['actual_success'],admitted=bool(r['admitted']),position_error_m=float(np.linalg.norm(pred[:3]-prefix['qpos'][tick,:3])),handoff_time_error_s=r['handoff_time_s']-float(prefix['trace'][tick,0])))
 ax.axvspan(10.3,10.6,alpha=.15,color='#56B4E9');ax.set_xlim(1.5,20.5);ax.set_ylim(-1.6,1.6);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.tick_params(labelsize=7);ax.set_xlabel('World x (m)',fontsize=7);ax.set_ylabel('World y (m)',fontsize=7);ax.set_title(f'root{row["root"]} up{row["upstream"]} down{row["downstream"]}',fontsize=8)
axes[0,0].legend(fontsize=6,ncol=3);fig.suptitle('OFFLINE TRUE-ENTRY INTERVENTIONS: known DEV map48,12 candidates reaching RB\nOld predictor trigger8.60m retained for this diagnosis; simulator8.55m. Travel +x; stages not artificially joined.',fontsize=10);fig.savefig(a.output/'xy.png',dpi=160);plt.close(fig)
with (a.output/'cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=errors[0],lineterminator='\n');w.writeheader();w.writerows(errors)
shutil.copyfile(a.interventions/'contract.json',a.output/'contract.json')
text='# Offline entry interchange diagnostic\n\n[All12 actual/reference/predicted XY](xy.png). Actual future entry is deliberately used only in these interventions; none is an online planning result. The six approach0 failures have no real RB entry and remain in the full18-case report, not silently dropped from its denominator.\n\n|Entry intervention|Feasible admitted /4|Infeasible admitted /8|\n|---|---:|---:|\n'
for name in paths:
 rs=[r for r in errors if r['method']==name];text+=f'|{name}|{sum(r["admitted"] and r["actual_success"] for r in rs)}|{sum(r["admitted"] and not r["actual_success"] for r in rs)}|\n'
text+='''\nPhysical-only replacement loses both baseline admissions; history-only retains two; jointly replacing physical state and history admits all four. Thus these observations do not isolate one scalar or one component as the sole cause. The pair matters; a history inferred from predicted motion can be inconsistent with independently substituted physical state. Predicted entry time is unchanged in every intervention. No training, controller updates, success-threshold change or new physical episodes.\n\nThis experiment used the pre-fix predictor trigger8.60m; source resolved execution uses8.55m. Retain the diagnosis as historical evidence; do not combine it with trigger-corrected scores. The trigger correction is evaluated separately before considering additional training.\n'''
(a.output/'README.md').write_text(text)
