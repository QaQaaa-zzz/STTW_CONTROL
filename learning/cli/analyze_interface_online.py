"""Analyze all frozen conditions and crossed-slot repetitions; no exclusions for outcome."""
import argparse,collections,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
methods=['interface_world_model','interface_direct','distance_rule'];positions=[1.8,3.,5.];rows=[];raw={};total_steps=0;wall=0
for rep in range(3):
 run=a.run/f'repeat{rep}';summary=json.loads((run/'evaluation/summary.json').read_text());pr=json.loads((run/'protocol.json').read_text());cfg=json.loads((run/'config.json').read_text());records=json.loads((run/'evaluation/moving_planning.json').read_text())['records'];records={r['lane']:r for r in records};data=np.load(run/'evaluation/trajectories.npz');raw[rep]=(data,cfg,records);total_steps+=summary['physics_lane_steps'];wall+=summary['elapsed_s']
 for c,outcome in zip(pr['cases'],summary['cases']):
  r=records.get(c['lane']);chosen=r.get('selected') if r else None
  rows.append(dict(c,success=outcome['success'],road_violation=outcome['road_violation'],end_code=outcome['end_code'],handoff_s=outcome['handoff_s'],selected=chosen,selected_route=pr['route_names'][chosen] if chosen is not None else None,query_s=r['query_seconds'] if r else None,delay_s=r.get('simulated_delay_s') if r else None,query_root=r['query_root'] if r else None))
groups=[]
for method in methods:
 for x in [None]+positions:
  rr=[r for r in rows if r['method']==method and (x is None or r['x']==x)]
  groups.append(dict(method=method,query_x=x,n=len(rr),success=sum(r['success'] for r in rr),road_violation=sum(r['road_violation'] for r in rr),handoffs=sum(r['handoff_s'] is not None for r in rr),unpublished=sum(r['selected'] is None for r in rr),mean_query_ms=1000*np.mean([r['query_s'] for r in rr if r['query_s'] is not None]),selected_routes=dict(collections.Counter(r['selected_route'].split('_v')[0] if r['selected_route'] else 'unpublished' for r in rr)),end_codes=dict(collections.Counter(str(r['end_code']) for r in rr))))
lookup={(r['root'],r['slot'],r['method']):r for r in rows};pairs=[];root_scores=np.array([[np.mean([lookup[root,slot,m]['success'] for slot in range(3)]) for m in methods] for root in range(30)])
rng=np.random.default_rng(2026111208);bootstrap=np.stack([np.concatenate([rng.choice(np.arange(j,30,3),10,replace=True) for j in range(3)]) for _ in range(5000)])
comparisons=[]
for other in methods[1:]:
 mid=methods.index(other);diff=root_scores[:,0]-root_scores[:,mid];draw=diff[bootstrap].mean(1);gains=losses=equal_queries=0
 for root in range(30):
  for slot in range(3):
   left,right=lookup[root,slot,methods[0]],lookup[root,slot,other];gain=left['success'] and not right['success'];loss=right['success'] and not left['success'];gains+=gain;losses+=loss
   eq=left['query_root'] is not None and right['query_root'] is not None and np.array_equal(left['query_root'],right['query_root']);equal_queries+=eq
   distance=float(np.linalg.norm(np.array(left['query_root'])[:3]-np.array(right['query_root'])[:3])) if left['query_root'] is not None and right['query_root'] is not None else None
   pairs.append(dict(root=root,slot=slot,comparator=other,gain=bool(gain),loss=bool(loss),same_selected=left['selected']==right['selected'],query_root_exact=eq,query_position_distance_m=distance,interface_repeat=left['repeat'],comparator_repeat=right['repeat']))
 comparisons.append(dict(comparator=other,condition_mean_success_difference=float(diff.mean()),stratified_root_bootstrap_95percent=np.quantile(draw,[.025,.975]).tolist(),matched_slot_gains=int(gains),matched_slot_losses=int(losses),query_root_exact=equal_queries,total_slot_pairs=90))
fig,axs=plt.subplots(3,3,figsize=(16,12),layout='constrained');rfig,rax=plt.subplots(2,3,figsize=(15,7),layout='constrained')
reference_seen=set()
for row in rows:
 d,cfg,records=raw[row['repeat']];lane=row['lane'];valid=d['valid'][:,lane];xy=d['root_xyzw'][valid,lane,:2];col=methods.index(row['method']);ax=axs[positions.index(row['x']),col];color='tab:blue' if row['success'] else 'tab:red';ax.plot(xy[:,0],xy[:,1],color=color,alpha=.3,lw=.6);ax.scatter(*xy[-1],color=color,marker='o' if row['success'] else 'x',s=8)
 if row['selected'] is not None:
  road=np.load(cfg['task_reference_path'])['xy'][row['selected']];key=(row['x'],col,row['selected']//15)
  if key not in reference_seen:ax.plot(road[:,0],road[:,1],'k--',lw=.7,alpha=.5);reference_seen.add(key)
 for axis,values in [(rax[0,col],d['reward'][valid,lane]),(rax[1,col],np.cumsum(d['reward'][valid,lane]))]:axis.plot(d['time'][valid],values,color=color,alpha=.12,lw=.5)
for i,x in enumerate(positions):
 for j,m in enumerate(methods):
  ax=axs[i,j];ax.set_aspect('equal',adjustable='datalim');ax.set(title=f'{m},query{x}m',xlabel='world x(m)',ylabel='world y(m)');ax.grid(alpha=.2)
for j,m in enumerate(methods):
 rax[0,j].set(title=m,ylabel='Per-step reward',xlabel='time(s)');rax[1,j].set(ylabel='Cumulative reward',xlabel='time(s)')
fig.suptitle('All270 trajectories: blue goal,red failure; dashed selected original routes; true endpoints');fig.savefig(a.run/'xy_all.png',dpi=150);plt.close(fig);rfig.savefig(a.run/'reward_all.png',dpi=150);plt.close(rfig)
result=dict(groups=groups,comparisons=comparisons,cases=rows,pairs=pairs,physics_lane_steps=total_steps,elapsed_s=wall,independent_conditions=30,repeats=3,bootstrap='5000 root-resamples stratified by10roots per query position; all conditions retained; descriptive uncertainty,not significance claim')
(a.run/'results.json').write_text(json.dumps(result,indent=2))
lines=['# Frozen two-interface online execution','','![All actual/reference XY](xy_all.png)','','![All per-step and cumulative rewards](reward_all.png)','','30 newly sampled independent roots,10 per query position; each method evaluated3times with crossed environment slots.90episodes per method are not90independent initial conditions. Model weights,conditions and distance rule frozen before physics.','','|Query(m)|Method|Goal/N|Road violation|Handoffs|Mean query(ms)|','|---|---|---:|---:|---:|---:|']
for g in groups:lines.append(f"|{g['query_x']}|{g['method']}|{g['success']}/{g['n']}|{g['road_violation']}|{g['handoffs']}|{g['mean_query_ms']:.3f}|")
lines+=['','|Comparator|Mean condition difference|Root-bootstrap95% interval|Matched-slot gains/losses|Exactly equal query roots|','|---|---:|---|---:|---:|']
for c in comparisons:lines.append(f"|{c['comparator']}|{c['condition_mean_success_difference']:.4f}|{c['stratified_root_bootstrap_95percent']}|{c['matched_slot_gains']}/{c['matched_slot_losses']}|{c['query_root_exact']}/90|")
lines+=['',f"Actual{total_steps} physical lane steps; summed simulator wall time{wall:.2f}s; no policy training.",'', 'Same known map and fixed cold handoff context. This evaluates one moving query and publication recheck,not repeated feedback replanning or changed downstream widths/curvatures. Missing publications and upstream failures stay in the main denominator. Each method uses each environment slot once per condition; query-state equality is audited but does not serialize hidden solver state. If query states differ, paired outcome gains are descriptive rather than a pure causal estimator. Bootstrap units are roots,not samples or repeats.','', 'Raw trajectories/configs and selected forecasts remain under repeat0/1/2. All per-case outcomes,pairs and route choices are in results.json. No outcomes were used to retune the frozen model.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(groups=[g for g in groups if g['query_x'] is None],comparisons=comparisons,elapsed_s=wall),indent=2))
