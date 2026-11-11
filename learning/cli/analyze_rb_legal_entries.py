"""Separate whole-task skill substitution from matched legal-entry evidence."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data={};groups=[]
for skill in ['source','rb400']:
 run=a.base/skill;cases=json.loads((run/'protocol.json').read_text())['cases'];summary=json.loads((run/'evaluation/summary.json').read_text())['cases'];z=np.load(run/'evaluation/trajectories.npz');root=z['root_xyzw'];ob=z['observation'];history=z['controller_history'];road=z['road_violation_history'];entries={}
 for case,out in zip(cases,summary):
  t=out['entry_s'];lane=out['lane'];candidate=case['candidate']
  if t is None or not candidate.startswith(('straight_jump','s_bend_jump')):continue
  k=round(t/.02)
  if road[:k,lane].any():continue
  entries[(case['root'],candidate)]=dict(root=root[k-1,lane],obs=ob[k,lane],history=history[k,lane],time=t,success=out['success'],handoff=out['handoff_s'] is not None)
 data[skill]=entries;groups.append(dict(skill=skill,total=len(summary),whole_goals=sum(c['success'] for c in summary),legal_straight_S_entries=len(entries),entry_goals=sum(c['success'] for c in entries.values()),handoffs=sum(c['handoff'] for c in entries.values())))
pairs=[]
for key in sorted(set(data['source'])&set(data['rb400'])):
 s,r=data['source'][key],data['rb400'][key];exact=s['time']==r['time'] and all(np.array_equal(s[k],r[k]) for k in ['root','obs','history'])
 pairs.append(dict(root=key[0],candidate=key[1],exact_recorded_entry=bool(exact),source_goal=s['success'],rb_goal=r['success']))
result=dict(groups=groups,common_legal=len(pairs),exact_entries=sum(x['exact_recorded_entry'] for x in pairs),matched_gains=sum(x['exact_recorded_entry'] and not x['source_goal'] and x['rb_goal'] for x in pairs),matched_losses=sum(x['exact_recorded_entry'] and x['source_goal'] and not x['rb_goal'] for x in pairs),pairs=pairs,scope='Historical whole-task and conditional legal-entry counts reported together;exact records do not serialize hidden PhysX state;no new RB+replanning execution claimed')
a.output.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='pairs'})
