"""Recompute TRAIN-route baseline and inspect the two frozen rule executions."""
from pathlib import Path
import json
import numpy as np
b=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs');out=Path('docs/evidence/chapter3_handoff_audit')
train=b/'chapter3_offroad_distance_training/segments'
p=json.loads((train/'protocol.json').read_text())['cases'];s=json.loads((train/'evaluation/summary.json').read_text())['cases'];rates={}
for route in sorted({r['candidate'].split('_v')[0] for r in p}):
 values=[r['success'] for r,q in zip(s,p) if q['role']=='TRAIN' and q['candidate'].split('_v')[0]==route];rates[route]=dict(n=len(values),rate=sum(values)/len(values))
test=json.loads(Path('docs/evidence/chapter3_event_frozen_test/event.json').read_text());cases=test['cases'];bs=np.mean([(rates[c['candidate'].split('_v')[0]]['rate']-c['success'])**2 for c in cases]);res=dict(train_route_rates=rates,route_lookup_brier=float(bs),event_brier=test['autoregressive']['brier'],constant_brier=test['autoregressive']['constant_brier'],scope='Same frozen240 TEST predictions; route rates computed from1199 publication collection TRAIN roots only. No tuning.')
old=b/'chapter3_offroad_distance_online';new=b/'chapter3_shared_head_online';differences=[]
ro=json.loads((old/'results.json').read_text())['cases'];rn=json.loads((new/'results.json').read_text())['cases'];po={r['lane']:r for r in json.loads((old/'evaluation/moving_planning.json').read_text())['records']};pn={r['lane']:r for r in json.loads((new/'evaluation/moving_planning.json').read_text())['records']}
for c in ro:
 if c['method']!='forced':continue
 d=next(r for r in rn if (r['method'],r['root'],r['x'])==(c['method'],c['root'],c['x']));u,v=po[c['lane']],pn[d['lane']]
 differences.append(dict(root=c['root'],x=c['x'],old_success=c['success'],new_success=d['success'],old_choice=c['selected'],new_choice=d['selected'],old_code=c['end_code'],new_code=d['end_code'],old_delay=u['simulated_delay_s'],new_delay=v['simulated_delay_s'],query_root_max_abs_difference=float(np.max(np.abs(np.array(u['query_root'])-v['query_root']))),published_root_max_abs_difference=float(np.max(np.abs(np.array(u['published_root'])-v['published_root'])))))
res['rule_repeats']=differences;(out/'baseline_audit.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
