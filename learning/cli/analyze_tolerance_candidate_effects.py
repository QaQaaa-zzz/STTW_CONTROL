import argparse,json,collections
from pathlib import Path
parser=argparse.ArgumentParser(description='Describe paired candidate outcomes without fitting or physics.');parser.add_argument('analysis',type=Path);parser.add_argument('--compare-with',type=Path);args=parser.parse_args();root=args.analysis

rows=json.loads((root/'results.json').read_text());groups=collections.defaultdict(list)
for r in rows:groups[(r['source_index'],r['preparation_s'],r['force_N'])].append(r)
fixed={};lines=['# Paired candidate effects on the new base','','All outcomes use the frozen source-through-pulse-end+3s no-contact definition. Each branch is one deterministic simulation; these counts are not calibrated probabilities. All nominal DEV source states and both signs/preparation times are retained.','','| Source / task time / route / bank | Preparation (s) | Force Y (N) | alpha 0 / .25 / .5 / .75 / 1 |','|---|---:|---:|---|']
discordant=0;validgroups=0
for key,rr in sorted(groups.items()):
 rr.sort(key=lambda r:r['candidate_alpha']);labels=[r['primary']['label_status'] for r in rr]
 assert len(rr)==5
 if set(labels)<= {'fall','no_fall'}:
  validgroups+=1;discordant+=len(set(labels))==2
 lines.append(f"| {key[0]} / {rr[0]['source_task_s']:g}s / {rr[0]['route']} / {rr[0]['bank_index']} | {key[1]} | {key[2]:g} | {' / '.join(labels)} |")
for a in [0,.25,.5,.75,1]:
 rr=[r for r in rows if r['candidate_alpha']==a];fixed[str(a)]=dict(total=len(rr),no_fall=sum(r['primary']['label_status']=='no_fall' for r in rr),fall=sum(r['primary']['label_status']=='fall' for r in rr),invalid=sum(r['primary']['label_status'] not in ('fall','no_fall') for r in rr))
prep={}
for v in [.2,.6]:
 rr=[r for r in rows if r['preparation_s']==v];prep[str(v)]=dict(total=len(rr),no_fall=sum(r['primary']['label_status']=='no_fall' for r in rr),invalid=sum(r['primary']['label_status'] not in ('fall','no_fall') for r in rr))
roles={}
for role in ['training','development']:
 rr=[r for r in rows if r['fitting_role']==role];roles[role]=dict(total=len(rr),labels=dict(collections.Counter(r['primary']['label_status'] for r in rr)))
summary=dict(cases=len(rows),fixed_alpha=fixed,preparation=prep,roles=roles,fully_valid_groups=validgroups,groups_with_alpha_dependent_outcomes=discordant,scope='Descriptive same-source branching; does not prove causal online scheduler benefit. Preparation also changes disturbance onset and vehicle state.')
(root/'candidate_effects.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'CANDIDATE_EFFECTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if args.compare_with is not None:
 other=json.loads((args.compare_with/'results.json').read_text())
 def pairing_key(r):
  return (r['source_group_identity'],r['candidate_alpha'],r['preparation_s'],1 if r['force_N']>0 else -1)
 left={pairing_key(r):r for r in other};right={pairing_key(r):r for r in rows}
 if len(left)!=len(other) or len(right)!=len(rows) or left.keys()!=right.keys():raise ValueError('Distinct, identical source/alpha/preparation/sign coverage required')
 pairs=[]
 for key,r in right.items():
  old=left[key]
  for field in ('duration_s','primary_window_start_s','primary_window_end_s','fitting_role','original_goal_xy'):
   if r[field]!=old[field]:raise ValueError(f'Paired label contract differs: {field}')
  pairs.append(dict(source_group_identity=key[0],candidate_alpha=key[1],preparation_s=key[2],sign=key[3],previous_force_N=old['force_N'],force_N=r['force_N'],previous_label=old['primary']['label_status'],label=r['primary']['label_status']))
 transitions=collections.Counter((p['previous_label'],p['label']) for p in pairs)
 comparison=dict(previous_analysis=str(args.compare_with.resolve()),current_analysis=str(root.resolve()),paired_branches=len(pairs),transitions=[dict(previous_label=k[0],label=k[1],count=v) for k,v in sorted(transitions.items())],pairs=pairs,scope='Descriptive paired branches. Shared ancestors are correlated; this comparison does not establish a monotonic safe-force threshold.')
 (root/'paired_force_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
 print(json.dumps({k:v for k,v in comparison.items() if k!='pairs'},indent=2))
