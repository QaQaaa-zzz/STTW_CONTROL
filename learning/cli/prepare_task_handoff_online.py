"""Freeze new roots for delayed handoff route selection; no outcome tuning."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
source=a.collection/'straight';cfg=json.loads((source/'config.json').read_text());pr=json.loads((source/'protocol.json').read_text());rng=np.random.default_rng(2026111219)
roots=[dict(y_m=float(rng.uniform(-.03,.03)),yaw_rad=float(rng.uniform(-.035,.035)),roll_rad=float(rng.uniform(-.02,.02)),speed_m_s=float(rng.uniform(1.9,2.1))) for _ in range(30)]
methods=['task_context','state_only','fixed_straight'];candidate_ids=[json.loads((a.collection/v/'protocol.json').read_text())['downstream_route_id'] for v in ['straight','left','right']]
for rep in range(3):
 out=a.output/f'repeat{rep}';out.mkdir();c=copy.deepcopy(cfg);cases=[]
 for root in range(30):
  for slot in range(3):
   cases.append(dict(lane=len(cases),root=root,slot=slot,repeat=rep,method=methods[(slot+rep)%3],approach_candidate=root%15,role='FROZEN_NEW_ROOTS'))
 c['num_envs']=90;c['ground_initial_conditions']=[roots[r['root']] for r in cases];h=c['ground_handoff'];h['reference_route_ids']=[7]*90;h['ground_only_lanes']=[False]*90;h['observer_initialization_lanes']=['cold']*90;h['handoff_route_ids']=[candidate_ids[0]]*90
 h['handoff_route_planner']=dict(model=str(a.model),candidate_route_ids=candidate_ids,methods=[r['method'] for r in cases])
 m=c['moving_event_planner'];m['methods']=['forced']*90;m['forced_candidates']=[r['approach_candidate'] for r in cases];m['decision_x_m']=1.8
 c['report_directory']=str(out);c['data_role']='Frozen30 new roots;15declared straight approach speed/alpha pairs balanced;3methods x3 crossed slots;downstream selection only;0training'
 (out/'config.json').write_text(json.dumps(c,indent=2));(out/'protocol.json').write_text(json.dumps(dict(roots=roots,cases=cases,route_names=pr['route_names'],candidate_route_ids=candidate_ids,methods=methods,scope=c['data_role']),indent=2))
contract=dict(question='Can frozen task-conditioned prediction select and execute downstream routes at measured handoff?',seed=2026111219,independent_roots=30,repetitions=3,methods=methods,total_episodes=270,upstream='Original source jump,H75;all15straight approach speed/alpha combinations assigned twice across30independent roots;not selected by new model',downstream='Actual handoff query;3known routes;default straight H75 runs during rounded-up measured query delay;no reset at publication',primary='Complete-task success over all30roots x3replicates including nohandoff/prior violations;conditional counts secondary;crossedslots;root-level bootstrap',budget='3x90lanes x10s;1080000physical lane steps;each300s wall cap;0fitting/PPO',stop='No seed/model/threshold retuning after outcomes;keep errors and failures',models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.glob('*.pt')},scope='Fixed map/corridor and new root panel;posthandoff decision only,not joint upstream task-conditioned optimization or safety certification')
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2));print(a.output)
