"""All-candidate same-axis comparison, preserving independent-interface discontinuities."""
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
c=json.loads((a.run/'contract.json').read_text());cases=json.loads((a.run/'cases.json').read_text());summary=json.loads((a.run/'summary.json').read_text());bank=np.load(c['routes']);pred={k:np.load(Path(c[v])/'predictions.npz') for k,v in [('C','continuous'),('I','independent')]};source=json.loads((Path(c['continuous'])/'contract.json').read_text());world=json.loads((Path(source['oracle'])/'map48/world.json').read_text())
fig,axes=plt.subplots(6,3,figsize=(18,10),layout='constrained')
for i,(row,ax) in enumerate(zip(cases['C'],axes.flat)):
 z=np.load(row['prefix']);actual=z['qpos'][:,:2]
 if row['suffix']:
  tail=np.load(row['suffix']);actual=np.r_[z['qpos'][z['trace'][:,0]<tail['trace'][0,0]-1e-6,:2],tail['qpos'][:,:2]]
 xy=bank['xy'][row['route']];ax.plot(xy[:,0],xy[:,1],':',color='.5',label='G / original reference')
 ax.plot(actual[:,0],actual[:,1],color='#0072B2',label='Actual');ax.scatter(*actual[-1],color='#0072B2',marker='x',s=20)
 for method,color,style in [('C','#D55E00','--'),('I','#009E73','-.')]:
  z=pred[method]
  for j,(name,steps) in enumerate([('approach',int(z['entry_steps'][i])+1),('jump',int(z['jump_steps'][i])),('departure',int(z['departure_steps'][i])+1)]):
   points=z[name][i,:steps,:2];ax.plot(points[:,0],points[:,1],style,color=color,lw=1,label=method if j==0 else None)
 for x0,x1,y0,y1 in world['terrain']['forbidden_rectangles']:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='.5',alpha=.25))
 ax.axvspan(10.3,10.6,color='#56B4E9',alpha=.15);ax.set_xlim(1.5,20.5);ax.set_ylim(-1.6,1.6);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
 ax.set_title(f'root{row["root"]} up{row["upstream"]} down{row["downstream"]} | actual{int(row["actual_success"])} I{int(cases["I"][i]["admitted"])} C{int(row["admitted"])}',fontsize=8);ax.tick_params(labelsize=7);ax.set_xlabel('World x (m)',fontsize=7);ax.set_ylabel('World y (m)',fontsize=7)
axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle('Known DEV map48, all18 candidates; travel +x. Actual stops at failure/goal.\nIndependent stage resets shown as gaps; equal XY scales; raw envelopes, no new calibration.',fontsize=11);fig.savefig(a.output/'xy.png',dpi=160);plt.close(fig)
for file in ['summary.json','contract.json','decisions.json']:shutil.copyfile(a.run/file,a.output/file)
with (a.output/'cases.csv').open('w') as f:
 names=['method','root','upstream','downstream','actual_success','admitted','goal_time_s','clearance_m'];w=csv.DictWriter(f,fieldnames=names,lineterminator='\n');w.writeheader()
 for method,rows in cases.items():
  for r in rows:w.writerow(dict(method=method,**{k:r[k] for k in names[1:]}))
text='# P4 G/I/C/Oracle: first matched DEV diagnostic\n\n[All18 actual/reference/I/C XY](xy.png). No new physical episodes; real outcomes reuse P2 full-snapshot branches.\n\n|Method|True admit|False admit|False reject|True reject|Selected completed /2|No plan /2|Admitted failure /2|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for method,counts in summary['confusion'].items():
 rows=[r for r in summary['decisions'] if r['method']==method];text+='|'+method+'|'+ '|'.join(str(counts[k]) for k in ['true_admit','false_admit','false_reject','true_reject'])+'|'+ '|'.join(str(sum(r[k] for r in rows)) for k in ['actual_success','no_plan','admitted_failure'])+'|\n'
text+='''\nI correctly admits all four feasible candidates, C only two: no evidence that continuous transfer improves this DEV matrix. Both select one successful task out of two, through different choices. I selects the false-admitted straight approach on root1002 and collides before jumping; C has no plan on root1001 and succeeds on root1002. No-plan is reported as a decision outcome, not a safe stopping policy or a simulated collision. Oracle completes both roots using approach1/departure0.\n\nI uses one coherent observed TRAIN interface per upstream/speed/phase, selected by proximity to the TRAIN median xyz/full velocity; its physical state, quaternion and histories stay paired. It does not read actual DEV future states. C propagates predicted states/history throughout. G follows the reference at1.8m/s and fixed height with level posture/locked joints; it has no dynamic or aerial prediction. G rejects all candidates under the same raw body-envelope obstacle/world constraints.\n\nFull backtracking traverses all nine candidates per root using exactly the same cached predictor, gates and cost as exhaustive joint selection; all eight decisions coincide. This establishes equivalence for this finite implementation, not an advantage in planning quality or a new physical experiment. `summary.json` separates model-query wall time (including loading) and cached selection time; GPU timing is not a deployment latency claim.\n\nReal obstacle clearance and failure stage are preserved in decisions.json. Successful sources report0m for some samples in the existing native geometry-distance audit; do not interpret those values as a positive safety margin. Only two roots on one known DEV map;18 candidates share prefixes and are not18 independent task trials. No bootstrap/generalization claim.\n\nP4 remains incomplete: per-model and shared/cross-margin calibration, direct/residual structural comparison, larger frozen validation and stage-error/occupancy diagnosis remain. These raw-envelope diagnostics do not replace calibrated final results.\n'''
(a.output/'README.md').write_text(text)
# Offline event-state comparison only; no actual future is fed to selection.
errors=[]
for method in ['I','C']:
 z=pred[method]
 for i,row in enumerate(cases[method]):
  actual=np.load(row['prefix']);phase=actual['trace'][:,-1]
  indices=np.flatnonzero(phase==2)
  if len(indices):
   tick=int(indices[0]);truth=actual['qpos'][tick];estimate=z['jump'][i,int(z['jump_steps'][i])-1]
   event_time=row['handoff_time_s'];time_error=event_time-float(actual['trace'][tick,0]);where='RB_to_ground'
  else:
   tick=len(phase)-1;t=float(actual['trace'][tick,0]);estimate=np.array([np.interp(t,np.arange(len(z['approach'][i]))*.02,z['approach'][i,:,j]) for j in range(12)]);truth=actual['qpos'][tick];time_error=None;where='actual_approach_failure_time'
  yaw=lambda q:np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
  dyaw=float(yaw(estimate[3:7])-yaw(truth[3:7]));dyaw=float(np.arctan2(np.sin(dyaw),np.cos(dyaw)))
  errors.append(dict(method=method,root=row['root'],upstream=row['upstream'],downstream=row['downstream'],comparison=where,actual_time_s=float(actual['trace'][tick,0]),event_time_error_s=time_error,dx_m=float(estimate[0]-truth[0]),dy_m=float(estimate[1]-truth[1]),xyz_error_m=float(np.linalg.norm(estimate[:3]-truth[:3])),yaw_error_rad=dyaw))
with (a.output/'stage_errors.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=errors[0],lineterminator='\n');w.writeheader();w.writerows(errors)
with (a.output/'README.md').open('a') as f:f.write('\n[Stage-event state errors](stage_errors.csv) compare predicted versus actual RB-to-ground events (including event-time difference); approach failures are compared at the actual failure time. Event-state errors are not same-time trajectory errors. Downstream variants share the same preceding event and must not be counted as independent samples.\n')
