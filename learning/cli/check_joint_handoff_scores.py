"""Reproduce joint scores from recorded pre-step observations/controller history."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.interface_world_model import InterfaceWorldModel
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);checks=[]
for rep in range(3):
 run=a.run/f'repeat{rep}';cfg=json.loads((run/'config.json').read_text());m=cfg['moving_event_planner'];model=InterfaceWorldModel(m['interface_model'],m['task_condition']);z=np.load(run/'evaluation/trajectories.npz');obs=z['observation'];history=z['controller_history'];params=np.asarray(m['candidate_parameters'],np.float32);n=len(params);dt=float(z['time'][0])
 for q in json.loads((run/'evaluation/moving_planning.json').read_text())['records']:
  if q['method']!='interface_task':continue
  step=q['query_step'];lane=q['lane'];root=np.asarray(q['query_root'],np.float32)[None]
  pred=model.score(np.repeat(root,n,0),np.repeat(obs[step,lane:lane+1],n,0),np.repeat(history[step,lane:lane+1],n,0),np.full(n,step*dt),params,particles=32,task_conditioned=True)
  difference=float(np.max(np.abs(np.asarray(pred['score'])-q['scores'])))
  checks.append(dict(repeat=rep,lane=lane,max_score_difference=difference,route_match=pred['selected_downstream_route']==q['prediction']['selected_downstream_route']))
result=dict(queries=len(checks),maximum_score_difference=max(x['max_score_difference'] for x in checks),all_routes_match=all(x['route_match'] for x in checks),checks=checks,scope='Frozen weight and recorded-input replay; not physics determinism or prediction accuracy')
(a.run/'score_replay.json').write_text(json.dumps(result,indent=2));assert result['maximum_score_difference']<1e-6 and result['all_routes_match'];print({k:v for k,v in result.items() if k!='checks'})
