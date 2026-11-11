"""Freeze a 24-source paired initialization diagnostic; no outcome-based selection."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.source/'config.json').read_text());pr=json.loads((a.source/'protocol.json').read_text());summary=json.loads((a.source/'evaluation/summary.json').read_text())['cases']
groups=collections.defaultdict(list)
for c,r in zip(pr['cases'],summary):
 if r['handoff_s'] is not None:groups[(c['root'],c['candidate'].split('_v')[0])].append(c['lane'])
for key in groups:groups[key].sort(key=lambda i:pr['cases'][i]['candidate_id'])
selected=[]
while len(selected)<24:
 for key in sorted(groups):
  if groups[key] and len(selected)<24:selected.append(groups[key].pop(0))
 assert any(groups.values()) or len(selected)==24,'Insufficient handoff sources'
lanes=[lane for lane in selected for _ in range(2)];modes=['cold','measured']*24
for k in ['reference_route_ids','ground_only_lanes']:cfg['ground_handoff'][k]=[cfg['ground_handoff'][k][i] for i in lanes]
for k in ['methods','forced_candidates']:cfg['moving_event_planner'][k]=[cfg['moving_event_planner'][k][i] for i in lanes]
cfg['ground_initial_conditions']=[cfg['ground_initial_conditions'][i] for i in lanes];cfg['num_envs']=48;cfg['ground_handoff']['observer_initialization_lanes']=modes;cfg['data_role']='DEV retrospective handoff-conditioned initialization diagnostic; 24 sources from4 roots;48 fresh prefix executions;0training';cfg['report_directory']=str(a.output)
cases=[]
for lane,(source_lane,mode) in enumerate(zip(lanes,modes)):
 c=copy.deepcopy(pr['cases'][source_lane]);c.update(lane=lane,source_lane=source_lane,source_root=c['root'],root=lane//2,method=mode,role='DEV_INITIALIZATION',logical_case=lane);cases.append(c)
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(a.output/'protocol.json').write_text(json.dumps(dict(cases=cases,candidates=['cold','measured'],route_names=pr['route_names'],source_roots=pr['roots'],scope='Pair index root denotes source branch, not independent initial condition. Four underlying original roots. Historical handoff selection; not population success estimate.'),indent=2)+'\n')
(a.output/'declaration.json').write_text(json.dumps(dict(source=str(a.source),selected_source_lanes=selected,budget='48 environments x10s maximum;192000 physical lane steps;0PPO;25min wall limit',selection='Round-robin source-root/route groups, ascending candidate ID among recorded handoffs; no success/failure selection',change='At actual handoff only: measured steer/lean/lean-rate ESO state and lean-rate filter initial value; unknown disturbance remains zero; H75 history reset unchanged',labels='First2s posthandoff descriptive survival/domain/corridor diagnostics; no calibrated recovery-domain label',invariants='Frozen original source jump/H75, same tasks, contacts, thresholds, initial root conditions, full prefix rerun, no root-only snapshot restore',concurrency='User explicitly authorized this small evaluation alongside existing H75 task'),indent=2)+'\n');print(a.output)
