from pathlib import Path
import json,csv,numpy as np
root=Path(__file__).parent;rows=json.loads((root/'evaluation_candidates.json').read_text());summary=[];cases=[]
for candidate in rows:
 for mapid in ['map06','map07','map48']:
  p=root/'evaluation'/candidate['name']/mapid;pred=np.load(p/'predictions.npz');records=json.loads((p/'cases.json').read_text());metrics=[]
  for i,row in enumerate(records):
   z=np.load(row['prefix']);idx=np.flatnonzero(z['trace'][:,-1]==2);has_actual=bool(len(idx));position=dt=dy=None
   if row['events_found'] and has_actual:
    actual=z['qpos'][idx[0]];q=pred['handoff_qpos'][i];position=float(np.linalg.norm(q[:3]-actual[:3]));dy=float(abs(q[1]-actual[1]));dt=float(abs(row['handoff_time_s']-z['trace'][idx[0],0]))
    metrics.append((position,dy,dt))
   cases.append(dict(candidate=candidate['name'],map=mapid,root=row['root'],route=row['route'],speed=row['speed'],actual_success=row['actual_success'],admitted=bool(row['admitted']),actual_handoff=has_actual,predicted_handoff=bool(row['events_found']),position_error_m=position,lateral_error_m=dy,event_error_s=dt))
  summary.append(dict(**candidate,map=mapid,**json.loads((p/'summary.json').read_text()),predicted_handoffs=sum(bool(r['events_found']) for r in records),matched_handoffs=len(metrics),position_mae_m=float(np.mean([x[0] for x in metrics])) if metrics else None,lateral_mae_m=float(np.mean([x[1] for x in metrics])) if metrics else None,event_mae_s=float(np.mean([x[2] for x in metrics])) if metrics else None))
ranking=[]
for c in rows:
 v=[x for x in summary if x['name']==c['name'] and x['map']!='map48'];n=sum(x['matched_handoffs'] for x in v)
 ranking.append(dict(**c,dev_admitted=sum(x['true_admit'] for x in v),dev_handoffs=n,dev_position_mae_m=sum((x['position_mae_m'] or 0)*x['matched_handoffs'] for x in v)/n if n else None))
selected=[]
for arm in ['ordered']:
 c=min([x for x in ranking if x['arm']==arm],key=lambda x:(-x['dev_admitted'],x['dev_position_mae_m'] if x['dev_position_mae_m'] is not None else float('inf')));selected.append(c)
for name,data in [('candidate_summary',summary),('ranking',ranking),('selected_comparison',selected)]:
 (root/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
with (root/'candidate_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=cases[0]);w.writeheader();w.writerows(cases)
print(json.dumps(ranking,indent=2));print('Selected',json.dumps(selected))
