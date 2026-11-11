"""Paired observed post-handoff windows; no inferred recoverability boundary."""
import argparse,json,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();pr=json.loads((a.run/'protocol.json').read_text());s=json.loads((a.run/'evaluation/summary.json').read_text());d=np.load(a.run/'evaluation/trajectories.npz');rows=[]
for c,r in zip(pr['cases'],s['cases']):
 lane=c['lane'];h=r['handoff_s'];row=dict(pair=c['root'],source_root=c['source_root'],source_lane=c['source_lane'],method=c['method'],candidate=c['candidate'],handoff_s=h,success=r['success'],end_code=r['end_code'],road_violation=r['road_violation'],two_seconds_observed=False,two_seconds_road_clean=None,goal_before_two_seconds=False,road_at_handoff=None,advance_x_first_two_seconds_m=None)
 if h is not None:
  t=d['time'];valid=d['valid'][:,lane];ids=np.flatnonzero(valid&(t>h+1e-6)&(t<=h+2.+1e-6));end=float(t[np.flatnonzero(valid)[-1]])
  row['two_seconds_observed']=end>=h+2.-1e-5
  row['goal_before_two_seconds']=bool(r['success'] and not row['two_seconds_observed'])
  row['road_at_handoff']=bool(d['event_road_violation'][lane,3]);row['two_seconds_road_clean']=bool(row['two_seconds_observed'] and not d['road_violation_history'][ids,lane].any())
  if len(ids):row['advance_x_first_two_seconds_m']=float(d['root_xyzw'][ids[-1],lane,0]-d['event_root_xyzw'][lane,3,0])
 rows.append(row)
groups=[]
for method in ['cold','measured']:
 rs=[r for r in rows if r['method']==method];groups.append(dict(method=method,n=len(rs),handoffs=sum(r['handoff_s'] is not None for r in rs),success=sum(r['success'] for r in rs),two_seconds_observed=sum(r['two_seconds_observed'] for r in rs),two_seconds_road_clean=sum(r['two_seconds_road_clean'] is True for r in rs),goal_before_two_seconds=sum(r['goal_before_two_seconds'] for r in rs),termination_codes=dict(collections.Counter(str(r['end_code']) for r in rs))))
pairs=[]
for i in range(24):
 c,m=[next(r for r in rows if r['pair']==i and r['method']==method) for method in ['cold','measured']];pair=dict(pair=i,cold=c['success'],measured=m['success'],both_handoff=c['handoff_s'] is not None and m['handoff_s'] is not None)
 if pair['both_handoff']:
  lc,lm=[next(q['lane'] for q in pr['cases'] if q['root']==i and q['method']==method) for method in ['cold','measured']]
  pair['handoff_root_max_abs_difference']=float(np.max(np.abs(d['event_root_xyzw'][lc,3]-d['event_root_xyzw'][lm,3])))
  pair['handoff_time_difference_s']=m['handoff_s']-c['handoff_s']
  pair['handoff_position_distance_m']=float(np.linalg.norm(d['event_root_xyzw'][lc,3,:3]-d['event_root_xyzw'][lm,3,:3]))
 pairs.append(pair)
result=dict(groups=groups,pairs=pairs,cases=rows,elapsed_s=s['elapsed_s'],physics_lane_steps=s['physics_lane_steps'],scope='Descriptive DEV initialization ablation.24 selected historical handoff branches from4 original roots, not24 independent roots. Full prefix rerun; no certified capture envelope or warmed H75 history.')
(a.run/'initialization_results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Landing observer initialization diagnostic','','![Actual/reference XY](plots/map_and_actual_xy.png)','','|Initialization|N|Handoffs|Complete task|Observed full2s after handoff|Full2s without prior/current road violation|Goal before2s|','|---|---:|---:|---:|---:|---:|---:|']
for g in groups:lines.append('|'+ '|'.join(str(g[k]) for k in ['method','n','handoffs','success','two_seconds_observed','two_seconds_road_clean','goal_before_two_seconds'])+'|')
lines+=['',f"Measured versus cold: gains {sum(r['measured'] and not r['cold'] for r in pairs)}, losses {sum(r['cold'] and not r['measured'] for r in pairs)}. Wall time {s['elapsed_s']:.2f}s; {s['physics_lane_steps']} physical lane steps; no training.",'','Pre-intervention matching is not established: handoff counts/states differ across cloned lanes. Completion differences are descriptive and must not be attributed causally to observer initialization. The default remains cold.', '', 'The measured condition initializes ESO steering/lean/lean-rate and the lean-rate filter from the current measurement. Unknown disturbance remains zero. H75 policy history is reset identically in both conditions. This is not temporal prewarming and does not test history initialization.','',result['scope'],'','Full2s means the trajectory has not ended before that observation point; it does not alone establish capture or eventual task success. Road-clean uses the latched road flag, including earlier violations. Goals reached earlier are separately reported, not counted as full-window observations. All sources, failures and individual records remain in JSON.','','[All paired XY, reward and speed panels](plots/README.md). Raw data: `'+str(a.run/'evaluation/trajectories.npz')+'`.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,indent=2))
