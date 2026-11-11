"""Freeze a joint approach/downstream selection panel without refitting skills or models."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
b=a.base;old=json.loads((b/'chapter3_task_handoff_online/repeat0/config.json').read_text())
interface=json.loads((b/'chapter3_interface_online/repeat0/config.json').read_text())['moving_event_planner']['interface_model']
rng=np.random.default_rng(2026111223)
roots=[dict(y_m=float(rng.uniform(-.03,.03)),yaw_rad=float(rng.uniform(-.035,.035)),roll_rad=float(rng.uniform(-.02,.02)),speed_m_s=float(rng.uniform(1.9,2.1))) for _ in range(30)]
methods=['joint_feedback','joint_open','fixed_straight'];task=dict(model=str(b/'chapter3_task_condition_model'),routes=str(b/'chapter3_task_condition_collection/issued.npz'),route_ids=[60,61,62],half_width_m=.6)
for rep in range(3):
 out=a.output/f'repeat{rep}';out.mkdir();c=copy.deepcopy(old)
 cases=[dict(lane=root*3+slot,root=root,slot=slot,repeat=rep,method=methods[(slot+rep)%3],role='FROZEN_NEW_ROOTS') for root in range(30) for slot in range(3)]
 c['ground_initial_conditions']=[roots[x['root']] for x in cases]
 m=c['moving_event_planner'];m['interface_model']=interface;m['task_condition']=task;m['candidate_parameters']=m['candidate_parameters'][:15]
 m['methods']=['forced' if x['method']=='fixed_straight' else 'interface_task' for x in cases];m['forced_candidates']=[7]*90;m['decision_x_m']=3.
 h=c['ground_handoff'];h['handoff_route_planner']['methods']=[dict(joint_feedback='task_context',joint_open='planned',fixed_straight='fixed_straight')[x['method']] for x in cases]
 c['report_directory']=str(out);c['data_role']='Frozen new30roots; joint15approaches x3downstream routes;open plan vs measured feedback vs fixed2m/s alpha.5 straight;no refitting'
 (out/'config.json').write_text(json.dumps(c,indent=2));(out/'protocol.json').write_text(json.dumps(dict(roots=roots,cases=cases,candidate_route_ids=[60,61,62],methods=methods,scope=c['data_role']),indent=2))
paths=list(Path(interface).glob('upstream_*.pt'))+list((b/'chapter3_task_condition_model').glob('task_context_*.pt'))
(a.output/'declaration.json').write_text(json.dumps(dict(seed=2026111223,methods=methods,independent_roots=30,repetitions=3,total_episodes=270,candidates='15straight approaches x3downstream roads;S/90degree upstream and bypass outside this extension',query='x3m, then first measured handoff for feedback; old controls execute during rounded inference latency',budget='3x90env x10s;1080000physical lane steps maximum;300s wall cap per batch;0training',primary='All episodes including upstream failures;root-bootstrap differences;query-state mismatch reported',models={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},stop='Keep model,thresholds,seeds fixed after launch;no outcome retuning'),indent=2))
print(a.output)
