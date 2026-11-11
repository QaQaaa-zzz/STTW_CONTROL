"""Freeze30 fresh independent roots and3 crossed-slot method repetitions."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((a.source/'config.json').read_text());protocol=json.loads((a.source/'protocol.json').read_text());rng=np.random.default_rng(2026111207)
roots=[dict(y_m=float(rng.uniform(-.03,.03)),yaw_rad=float(rng.uniform(-.035,.035)),roll_rad=float(rng.uniform(-.02,.02)),speed_m_s=float(rng.uniform(1.9,2.1))) for _ in range(30)]
methods=['interface_world_model','interface_direct','distance_rule'];positions=[1.8,3.,5.];fixed={1.8:48,3.:22,5.:4}
for rep in range(3):
 out=a.output/f'repeat{rep}';out.mkdir();c=copy.deepcopy(cfg);cases=[]
 for root in range(30):
  for slot in range(3):
   method=methods[(slot+rep)%3];x=positions[root%3];cases.append(dict(lane=len(cases),root=root,slot=slot,repeat=rep,method=method,x=x,candidate=method,logical_case=root*3+methods.index(method),role='FROZEN_NEW_ROOTS'))
 c['num_envs']=90;c['ground_initial_conditions']=[roots[q['root']] for q in cases];h=c['ground_handoff'];h['reference_route_ids']=[7]*90;h['ground_only_lanes']=[False]*90;h['observer_initialization_lanes']=['cold']*90
 m=c['moving_event_planner'];m['interface_model']=str(a.model);m['methods']=['forced' if q['method']=='distance_rule' else q['method'] for q in cases];m['decision_x_m']=[q['x'] for q in cases];m['forced_candidates']=[fixed[q['x']] for q in cases]
 c['report_directory']=str(out);c['data_role']='Frozen30 fresh roots x3methods,one of3query positions per root; three crossed-slot repetitions; fixed known map. No tuning on these outcomes.'
 (out/'config.json').write_text(json.dumps(c,indent=2));(out/'protocol.json').write_text(json.dumps(dict(roots=roots,cases=cases,candidates=methods,route_names=protocol['route_names'],scope=c['data_role']),indent=2))
declaration=dict(question='Can the trained two-interface scorer execute complete known-map tasks?',n_independent_conditions=30,methods=methods,repetitions=3,total_episodes=270,query_positions=positions,allocation='10 independent roots per query position; same root copied to3slots; Latin method rotation over3runs so each method uses each slot once per root',primary='Full-task completion and original constraint violations; all30conditions retained. Method-disagreement subgroup secondary only. Report paired condition-average differences;3repeats not90independent roots.',limitations='Same known map and controller context; no new downstream-width/curvature generalization or continuous replanning claim. Comparison includes actual inference latency.',budget='3batched90lane Isaac runs x10s;1,080,000physical lane steps maximum;each run5min wall cap;0training',resources='Small evaluation allowed parallel by explicit user instruction; no stopping other GPU job',models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.glob('*.pt')},downstream_directory=json.loads((a.model/'model.json').read_text())['downstream_directory'],rule=fixed,rule_source='Previously frozen TRAIN distance rule from original online comparison',seed=2026111207,stop='Do not retune frozen model or seeds after results. Runtime errors may be fixed with failed launches retained.')
(a.output/'declaration.json').write_text(json.dumps(declaration,indent=2));print(a.output)
