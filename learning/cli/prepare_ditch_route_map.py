"""Prepare one known map with jump and continuous-ground route/skill candidates.

The central trench stays open. A named lateral land strip forms the detour.
This only generates assets/configuration; it does not certify any skill.
"""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--roots',type=int,default=4);p.add_argument('--repeats',type=int,default=2)
p.add_argument('--detour-y',type=float,default=2.2)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=json.loads((a.base_collection/'config.json').read_text());protocol=json.loads((a.base_collection/'protocol.json').read_text())
if (base['ditch_terrain']['front_x_m']!=10.3 or base['ditch_terrain']['width_m']!=.3
    or not 1.4<a.detour_y<3.4 or not 1<=a.roots<=len(protocol['roots']) or a.repeats<1):
 raise ValueError('Requested roots/repeats or detour center outside the declared map contract')
source=np.load(base['ground_handoff']['reference_path']);road=np.load(base['task_reference_path']);s=source['s'];x=np.linspace(-5,25,12001)
rise=.5-.5*np.cos(np.pi*np.clip((x-2)/6,0,1));fall=.5+.5*np.cos(np.pi*np.clip((x-11)/5,0,1));y=a.detour_y*rise*fall
arc=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))];xy=np.column_stack((np.interp(s,arc,x),np.interp(s,arc,y)))
heading=np.arctan2(np.gradient(xy[:,1],s),np.gradient(xy[:,0],s));curvature=np.gradient(heading,s)
chosen=[protocol['candidates'].index(k) for k in ('smooth_left','smooth_right')]
routes=dict(s=s,xy=np.stack([source['xy'][i] for i in chosen]+[xy]),heading=np.stack([source['heading'][i] for i in chosen]+[heading]),curvature=np.stack([source['curvature'][i] for i in chosen]+[curvature]))
np.savez_compressed(a.output/'issued.npz',**routes);np.savez_compressed(a.output/'roads.npz',s=s,xy=np.stack([road['xy'][i] for i in chosen]+[xy]))
roots=protocol['roots'][:a.roots];names=['jump_left','jump_right','ground_detour'];assignment=np.repeat(np.arange(3),a.repeats).tolist();per_root=len(assignment)
base.update(num_envs=a.roots*per_root,ground_initial_conditions=[r for r in roots for _ in assignment],task_reference_path=str(a.output/'roads.npz'),data_role='Fixed map jump/ground-detour capability DEV; complete common starts; no training')
base['ditch_terrain']['land_bridge_y_interval_m']=[1.4,4.]
base['ground_handoff'].update(reference_path=str(a.output/'issued.npz'),reference_route_ids=assignment*a.roots,ground_only_lanes=[i==2 for i in assignment]*a.roots)
cases=[dict(lane=i*per_root+j,root=i,candidate=names[method],role='DEV') for i in range(a.roots) for j,method in enumerate(assignment)]
(a.output/'config.json').write_text(json.dumps(base,indent=2))
(a.output/'protocol.json').write_text(json.dumps(dict(roots=roots,cases=cases,candidates=[f'{name}_r{j}' for name in names for j in range(a.repeats)],method_assignment=assignment,budget=f'{len(cases)} lanes x10s maximum, 0training',map_change='Continuous side land strip y1.4..4 across x10.3..10.6; center jump trench remains open',selection='Manually enumerated route/skill candidates; no model or LLM choice'),indent=2))
contract=dict(map=base['ditch_terrain'],regions=dict(start={'x_m':1.5},crossing={'x_interval_m':[10.3,10.6]},goal=base['ground_handoff']['goal']),routes=[dict(name=name,path_index=i,skills=['STTW','heading125','STTW'] if i<2 else ['STTW'],handoff='first_far_contact' if i<2 else None) for i,name in enumerate(names)],constraints=dict(original_road_half_width_m=.6,physical_limits='unchanged task_spec',ground_detour_has_no_jump_handoff=True),capability_status='unmeasured; generation does not imply feasible')
(a.output/'map_contract.json').write_text(json.dumps(contract,indent=2));print(a.output/'config.json')
