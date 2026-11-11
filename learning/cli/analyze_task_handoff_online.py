"""Analyze the frozen measured-state downstream decision panel without dropping failures."""
import argparse,collections,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
declaration=json.loads((a.run/'declaration.json').read_text());methods=declaration['methods'];joint=methods[0]=='joint_feedback';rows=[];raw={};wall=steps=0
for rep in range(3):
 run=a.run/f'repeat{rep}';pr=json.loads((run/'protocol.json').read_text());cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'evaluation/summary.json').read_text());rec={r['lane']:r for r in json.loads((run/'evaluation/handoff_planning.json').read_text())['records']};z=np.load(run/'evaluation/trajectories.npz');raw[rep]=(z,cfg,pr);wall+=summary['elapsed_s'];steps+=summary['physics_lane_steps']
 for case,outcome in zip(pr['cases'],summary['cases']):
  lane=case['lane'];r=rec.get(lane);rows.append(dict(case,success=outcome['success'],end_code=outcome['end_code'],road_violation=outcome['road_violation'],handoff_s=outcome['handoff_s'],prior_road=bool(z['event_road_violation'][lane,3]) if outcome['handoff_s'] is not None else None,decision=r))
groups=[]
for method in methods:
 rr=[r for r in rows if r['method']==method];queries=[r['decision'] for r in rr if r['decision']];groups.append(dict(method=method,n=len(rr),success=sum(r['success'] for r in rr),road_violation=sum(r['road_violation'] for r in rr),handoffs=sum(r['handoff_s'] is not None for r in rr),valid_handoffs=sum(r['handoff_s'] is not None and not r['prior_road'] for r in rr),published=sum(q['status']=='published' for q in queries),mean_query_ms=float(1000*np.mean([q['query_seconds'] for q in queries])),delay_values_s=sorted({q['simulated_delay_s'] for q in queries}),selected_routes=dict(collections.Counter(str(q['selected_route']) for q in queries)),end_codes=dict(collections.Counter(str(r['end_code']) for r in rr))))
lookup={(r['root'],r['slot'],r['method']):r for r in rows};pairs=[];comparisons=[]
rng=np.random.default_rng(2026111220);boot=rng.integers(0,30,(5000,30))
for other in methods[1:]:
 diff=[];gain=loss=matched=both=0
 for root in range(30):
  diff.append(np.mean([int(lookup[root,slot,methods[0]]['success'])-int(lookup[root,slot,other]['success']) for slot in range(3)]))
  for slot in range(3):
   l,r=lookup[root,slot,methods[0]],lookup[root,slot,other];ld,rd=l['decision'],r['decision'];have=ld is not None and rd is not None;equal=bool(have and ld['query_s']==rd['query_s'] and np.array_equal(ld['query_root'],rd['query_root']) and np.array_equal(ld['query_observation'],rd['query_observation']));both+=have;matched+=equal;g=l['success'] and not r['success'];b=r['success'] and not l['success'];gain+=g;loss+=b
   pairs.append(dict(root=root,slot=slot,comparator=other,gain=bool(g),loss=bool(b),both_query=have,query_descriptor_exact=equal,same_selected=bool(have and ld['selected_route']==rd['selected_route'])))
 diff=np.array(diff);comparisons.append(dict(comparator=other,condition_mean_difference=float(diff.mean()),root_bootstrap95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),gains=int(gain),losses=int(loss),both_query=both,query_descriptor_exact=matched))
# Audit that each published decision reached the recorded controller route.
publication_checks=[]
for row in rows:
 q=row['decision']
 if q is None or q['status']!='published':continue
 z=raw[row['repeat']][0];lane=row['lane']
 # Route snapshots precede task.step; time labels are post-step. A switch
 # inside task.step first appears in the next snapshot, two ticks later.
 mask=z['valid'][:,lane] & (z['time']>=q['publication_s']+2*float(z['time'][1]-z['time'][0])-1e-6)
 actual=z['reference_route_id'][mask,lane]
 publication_checks.append(dict(root=row['root'],slot=row['slot'],method=row['method'],delay_correct=abs(q['publication_s']-q['query_s']-q['simulated_delay_s'])<1e-6,route_recorded=bool(len(actual) and np.all(actual==q['selected_route']))))
assert all(c['delay_correct'] and c['route_recorded'] for c in publication_checks)
route_outcomes=[]
for rid in (60,61,62):
 rr=[r for r in rows if r['method']==methods[0] and r['decision'] and r['decision']['selected_route']==rid]
 route_outcomes.append(dict(route=rid,selected=len(rr),goals=sum(r['success'] for r in rr)))
# Show every successful non-straight choice; this panel is explicitly descriptive.
positive=[r for r in rows if r['method']==methods[0] and r['success'] and r['decision']['selected_route']!=60]
if positive:
 pf,pa=plt.subplots(1,len(positive),figsize=(7*len(positive),4),squeeze=False,layout='constrained')
 for ax,row in zip(pa[0],positive):
  z,cfg,pr=raw[row['repeat']];lane=row['lane'];valid=z['valid'][:,lane];xy=z['root_xyzw'][valid,lane,:2];q=row['decision'];bank=np.load(cfg['task_reference_path'])['xy']
  for rid in (60,61,62):ax.plot(*bank[rid].T,'--',lw=.8,label=f'candidate {rid}')
  ax.plot(*xy.T,'k',lw=1.5,label='actual execution');ax.scatter(*q['query_root'][:2],marker='s',label='measured handoff');ax.scatter(*q['publication_root'][:2],marker='^',label='route publication');ax.scatter(*xy[-1],label='goal')
  ax.set(xlim=(9,19),ylim=(-1,1),xlabel='world x (m)',ylabel='world y (m)',title=f"Successful selected route {q['selected_route']}; root {row['root']}, slot {row['slot']}\nPost hoc illustration, not a matched gain claim");ax.set_aspect('equal');ax.legend(fontsize=7);ax.grid(alpha=.2)
 pf.savefig(a.run/'selected_success.png',dpi=170);plt.close(pf)
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');rf,rax=plt.subplots(2,3,figsize=(15,7),layout='constrained')
for rep,(z,cfg,pr) in raw.items():
 physical=z['root_xyzw'];valid=z['valid'];reward=z['reward'];bank=np.load(cfg['task_reference_path'])['xy']
 for row in [r for r in rows if r['repeat']==rep]:
  lane=row['lane'];col=methods.index(row['method']);v=valid[:,lane];xy=physical[v,lane,:2];color='tab:blue' if row['success'] else 'tab:red'
  for axis in axs[:,col]:axis.plot(xy[:,0],xy[:,1],color=color,alpha=.25,lw=.6);axis.scatter(*xy[-1],color=color,marker='o' if row['success'] else 'x',s=8)
  rax[0,col].plot(z['time'][v],reward[v,lane],color=color,alpha=.13,lw=.5);rax[1,col].plot(z['time'][v],np.cumsum(reward[v,lane]),color=color,alpha=.13,lw=.5)
for col,method in enumerate(methods):
 for axis in axs[:,col]:
  for route_id in pr['candidate_route_ids']:
   xy=bank[route_id];axis.plot(xy[:,0],xy[:,1],'k--',lw=.7,alpha=.4)
  axis.set(xlabel='world x(m)',ylabel='world y(m)',title=method);axis.set_aspect('equal');axis.grid(alpha=.2)
 axs[0,col].set(xlim=(0,19),ylim=(-4,4));axs[1,col].set(xlim=(9,19),ylim=(-2,2))
 rax[0,col].set(xlabel='time(s)',ylabel='Per-step task reward',title=method);rax[1,col].set(xlabel='time(s)',ylabel='Cumulative task reward')
fig.suptitle('All270 episodes; blue goal/red failure; immutable downstream references dashed; true endpoints')
fig.savefig(a.run/'xy_all.png',dpi=160);plt.close(fig);rf.savefig(a.run/'reward_all.png',dpi=160);plt.close(rf)
joint_decisions=[]
if joint:
 for rep in range(3):
  moving=json.loads((a.run/f'repeat{rep}/evaluation/moving_planning.json').read_text())['records']
  cases={r['lane']:r for r in rows if r['repeat']==rep}
  for rec in moving:
   row=cases[rec['lane']];prediction=rec['prediction'];selected=rec.get('selected')
   planned=prediction['selected_downstream_route'][selected] if prediction and selected is not None else 60
   actual=row['decision']['selected_route'] if row['decision'] else None
   joint_decisions.append(dict(root=row['root'],slot=row['slot'],method=row['method'],approach_candidate=selected,planned_downstream=planned,measured_choice=actual,revised=actual is not None and actual!=planned,success=row['success'],upstream_query_ms=rec['query_seconds']*1000,upstream_delay_s=rec.get('simulated_delay_s'),upstream_query_root=rec['query_root']))
result=dict(joint_decisions=joint_decisions,route_outcomes=route_outcomes,publication_checks=publication_checks,groups=groups,comparisons=comparisons,cases=rows,pairs=pairs,elapsed_s=wall,physics_lane_steps=steps,scope='Frozen30 new roots x3methods x3crossed slots;all upstream failures retained;query-state match only measured descriptors,not hidden solver state')
(a.run/'results.json').write_text(json.dumps(result,indent=2))
lines=['# Measured-state downstream route selection','','![All actual/reference XY](xy_all.png)','','![Per-step and cumulative rewards](reward_all.png)','','30 new independent roots; three methods and three crossed-slot repetitions, 90 episodes/method. All upstream failures remain in the denominator. Models and conditions frozen before execution.','','|Method|Goal/N|Handoffs|Valid handoffs|Published|Road violations|Mean query ms|','|---|---:|---:|---:|---:|---:|---:|']
for g in groups:lines.append(f"|{g['method']}|{g['success']}/{g['n']}|{g['handoffs']}|{g['valid_handoffs']}|{g['published']}|{g['road_violation']}|{g['mean_query_ms']:.3f}|")
lines+=['','|Comparator|Mean root difference|Root bootstrap95%|Gains/losses|Equal query descriptors/both queried|','|---|---:|---|---:|---:|']
for c in comparisons:lines.append(f"|{c['comparator']}|{c['condition_mean_difference']:.4f}|{c['root_bootstrap95']}|{c['gains']}/{c['losses']}|{c['query_descriptor_exact']}/{c['both_query']}|")
lines+=['','![Successful non-straight selection, descriptive](selected_success.png)','','Task-conditioned selected-route outcomes: '+str(route_outcomes)+'. These conditional counts are selection-biased, not route rankings. All '+str(len(publication_checks))+' publications match the logged route and declared delay.','',f'Actual {steps} physical lane steps; summed simulation wall{wall:.2f}s; 0 training.','', 'At first measured handoff the frozen model scores 3 known downstream roads. Straight H75 continues during rounded-up measured inference delay; publication changes the route index only and does not reset physical/controller state. Existing physical/corridor/goal predicates are retained. Candidate-route selection is downstream only; approach pairs are preassigned,not jointly optimized by this extension.','', 'No independent-map or certified recovery-margin claim. Differences in recorded query states are explicitly counted; even exact descriptors do not serialize hidden solver state. Causal gain attribution requires matched incoming states and repeated evidence. State-only scores share the same input across the3roads and ties choose the first (straight).','', 'Full query/publication records,model scores,actual contexts and endpoints remain in repeat0/1/2 and results.json. All failures and unpublished decisions are retained.']
if joint:
 lines+=['','|Method|Approach choices|Planned downstream|Handoff revisions|Successful revised|Mean upstream query ms|','|---|---|---|---:|---:|---:|']
 for method in methods:
  rr=[r for r in joint_decisions if r['method']==method]
  lines.append(f"|{method}|{dict(collections.Counter(r['approach_candidate'] for r in rr))}|{dict(collections.Counter(r['planned_downstream'] for r in rr))}|{sum(r['revised'] for r in rr)}|{sum(r['revised'] and r['success'] for r in rr)}|{np.mean([r['upstream_query_ms'] for r in rr]):.3f}|")
 lines=[line for line in lines if not line.startswith('At first measured handoff') and not line.startswith('No independent-map')]
 lines+=['', 'Joint panel: at x=3 m, frozen upstream predictions score 15 approach speed/preference pairs and three downstream routes. Joint-open retains its planned downstream route; joint-feedback queries again at measured handoff. Fixed uses 2 m/s, alpha .5 and straight continuation. The planned road runs during handoff inference delay. Route publication does not reset physical/controller state. This extension excludes upstream S/90-degree/bypass candidates; earlier broader panels remain separate.','', 'No certified margin, independent-map generalization, superiority or equivalence claim. Query descriptors differ and hidden solver state is not serialized. All failures remain in the primary denominator.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(groups=groups,comparisons=comparisons,elapsed_s=wall),indent=2))
