"""Compare saved jump actors through the same approach and frozen H75 continuation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
d=json.loads((a.run/'declaration.json').read_text());cases=d['cases'];results=[];rows=[];source=None;wall=steps=0;feedback_panel=bool(d.get('feedback_panel'))
fig,axs=plt.subplots(2,len(d['models']),figsize=(5*len(d['models']),8),layout='constrained')
rf,rax=plt.subplots(2,len(d['models']),figsize=(5*len(d['models']),7),layout='constrained')
for col,m in enumerate(d['models']):
 run=a.run/m['name'];cfg=json.loads((run/'config.json').read_text());s=json.loads((run/'evaluation/summary.json').read_text());archive=np.load(run/'evaluation/trajectories.npz');z={k:archive[k] for k in ['event_time','event_road_violation','event_root_xyzw','root_xyzw','observation','controller_history','time','valid','reward']};archive.close();bank=np.load(cfg['task_reference_path'])['xy'];wall+=s['elapsed_s'];steps+=s['physics_lane_steps'];decision_path=run/'evaluation/handoff_planning.json';decisions=json.loads(decision_path.read_text())['records'] if decision_path.exists() else [];labels=np.array([r['success'] for r in s['cases']]);valid_hand=np.isfinite(z['event_time'][:,3]) & ~z['event_road_violation'][:,3]
 if source is None:source=labels;source_z={k:z[k] for k in ['root_xyzw','observation','controller_history','time']};source_cases=s['cases']
 exact=entry_pairs=0;gains=losses=0
 for case,out in zip(cases,s['cases']):
  lane=case['lane'];v=z['valid'][:,lane];xy=z['root_xyzw'][v,lane,:2];color='tab:blue' if out['success'] else 'tab:red';t=z['time'][v];reward=z['reward'][v,lane]
  for ax in axs[:,col]:ax.plot(*xy.T,color=color,alpha=.25,lw=.65);ax.scatter(*xy[-1],marker='o' if out['success'] else 'x',color=color,s=10)
  rax[0,col].plot(t,reward,color=color,alpha=.2,lw=.5);rax[1,col].plot(t,np.cumsum(reward),color=color,alpha=.2,lw=.5)
  same=False;entry=out['entry_s'];other=source_cases[lane]['entry_s']
  if entry is not None and other is not None:
   entry_pairs+=1;k=np.argmin(abs(z['time']-entry));q=np.argmin(abs(source_z['time']-other));same=entry==other and all(np.array_equal(z[key][max(k-offset,0),lane],source_z[key][max(q-offset,0),lane]) for key,offset in [('root_xyzw',1),('observation',0),('controller_history',0)])
   exact+=same;gains+=same and labels[lane] and not source[lane];losses+=same and source[lane] and not labels[lane]
  rows.append(dict(case,model=m['name'],success=out['success'],end_code=out['end_code'],road_violation=out['road_violation'],entry_s=entry,handoff_s=out['handoff_s'],valid_handoff=bool(valid_hand[lane]),recorded_entry_exact=bool(same)))
 for ax in axs[:,col]:
  for route in [60,61,62]:ax.plot(*bank[route].T,'k--',lw=.8)
  ax.set(xlabel='world x (m)',ylabel='world y (m)',title=m['name']);ax.set_aspect('equal');ax.grid(alpha=.2)
 axs[0,col].set(xlim=(0,19),ylim=(-3,3));axs[1,col].set(xlim=(9,19),ylim=(-1.5,1.5))
 for ax in rax[:,col]:ax.set(xlabel='time (s)',title=m['name']);ax.grid(alpha=.2)
 rax[0,col].set_ylabel('Per-step task reward');rax[1,col].set_ylabel('Cumulative task reward')
 states=z['event_root_xyzw'][valid_hand,3]
 results.append(dict(model=m['name'],published_routes={str(r):sum(q['status']=='published' and q['selected_route']==r for q in decisions) for r in [60,61,62]},mean_query_ms=float(1000*np.mean([q['query_seconds'] for q in decisions])) if decisions else 0.,n=len(labels),goals=int(labels.sum()),gains=int((labels&~source).sum()),losses=int((source&~labels).sum()),handoffs=sum(r['handoff_s'] is not None for r in s['cases']),valid_handoffs=int(valid_hand.sum()),road_violations=sum(r['road_violation'] for r in s['cases']),recorded_entry_exact=exact,both_entry=entry_pairs,exact_entry_gains=int(gains),exact_entry_losses=int(losses),route_goals={str(r):int(labels[np.array([c['route']==r for c in cases])].sum()) for r in [60,61,62]},mean_abs_vy=float(np.abs(states[:,8]).mean()) if len(states) else None,mean_abs_world_wx=float(np.abs(states[:,10]).mean()) if len(states) else None))
fig.suptitle('All declared cases; blue goal/red failure; fixed reference dashed; physical endpoints');fig.savefig(a.run/'xy_all.png',dpi=160);rf.savefig(a.run/'reward_all.png',dpi=160);plt.close('all')
(a.run/'results.json').write_text(json.dumps(dict(groups=results,cases=rows,elapsed_s=wall,physics_lane_steps=steps,role=d['role']),indent=2))
lines=['# RB candidate handoff screening','','![All actual/reference XY](xy_all.png)','','Source and three saved RB actors; 30 previously inspected roots x3 continuation roads/model, 360 episodes. The three roads are distinct tasks, not independent repetitions. Same frozen H75 and physical criteria; 15 approach commands assigned before execution; no model-based upstream selection or retraining.','','|Actor|Goals/90|Gain/loss vs Source|Valid handoffs|Road violations|Straight/left/right goals (each30)|Exact entry pairs|','|---|---:|---:|---:|---:|---|---:|']
for r in results:lines.append(f"|{r['model']}|{r['goals']}|{r['gains']}/{r['losses']}|{r['valid_handoffs']}|{r['road_violations']}|{list(r['route_goals'].values())}|{r['recorded_entry_exact']}/{r['both_entry']}|")
lines+=['','![Per-step and cumulative rewards](reward_all.png)','',f'Actual {steps} physics lane steps, {wall:.2f}s summed simulation; zero training.','', 'Whole-task goals are primary; valid-handoff dynamics are conditional and selection-biased. Matched recorded entry descriptors do not serialize hidden contact solver state. Cross-batch outcomes may vary. This is development screening, not independent TEST or proof of full RecoveryBridge transfer. No candidate is automatically adopted and no checkpoints are discarded. Raw identity/config/trajectory records remain in each model subdirectory.']
if feedback_panel:
 lines[0]='# RB301 with measured-state downstream feedback'
 lines[4]='30 previously inspected roots x3 simulation slots per method,270 episodes. Source-fixed and RB301-fixed retain the straight road; RB301-feedback queries the frozen source-trained Frenet model at measured handoff. All15approach pairs are preassigned, H75/physics/10s horizon unchanged. No refitting, policy training, or calibrated RB capability claim.'
 lines += ['', 'Published routes and query cost: '+str([{k:r[k] for k in ['model','published_routes','mean_query_ms']} for r in results]), '', 'Route-goal columns above group by the initially assigned road (straight), not the final selected road. Slot replicates are grouped under the30roots, not counted as independent conditions.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(results,indent=2))
