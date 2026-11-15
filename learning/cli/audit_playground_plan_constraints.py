"""Reconstruct finite-library clearance witnesses from saved predictions; no simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.playground_constraint_log import clearance_witness

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--maps',nargs='+',type=int,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
manifest=json.loads((a.root/'manifest.json').read_text());source=Path(manifest['source'])
schema=json.loads((source/'model_state_schema.json').read_text());model=mujoco.MjModel.from_binary_path(str(source/'terrain.mjb'))
envelope=BodyEnvelope(model,device='cpu',dimensions=3)
names=[model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i]>0 and (model.geom_contype[i] or model.geom_conaffinity[i])]
items=[]
for row in json.loads((source/'status.json').read_text())['results']:
    with np.load(source/row['case_dir']/'trajectory.npz') as d:
        pose,_=states(d['qpos'][::4],d['qvel'][::4],schema);trace=d['trace'][::4];anchor=int(np.flatnonzero(trace[:,-1]==2)[0])
        items.append(dict(row=row,pose=pose,phase=trace[:,-1],anchor=anchor,speed=float(trace[anchor,18])))
plans=json.loads((a.root/'plans.json').read_text());calibration=json.loads((a.root/'calibration.json').read_text());dynamics=json.loads((a.root/'dynamics.json').read_text());records=[]
for plan in plans:
    if plan['map_id'] not in a.maps:continue
    mid,name=plan['map_id'],plan['model'];world=json.loads((a.root/f'map{mid:02d}_layout{plan["layout"]:02d}'/'world.json').read_text())
    cal=next(r for r in calibration if r['map_id']==mid and r['model']==name);edges=np.array(cal['edges']);margins=np.array(cal['axis_margin_m'])
    boxes=np.array([[r[0],r[2],0.,r[1],r[3],.5] for r in world['terrain']['forbidden_rectangles']])
    predictions=np.load(a.root/f'predictions_map{mid:02d}_{name}.npz')['pose'];details=[]
    for i,item in enumerate(items):
        dyn=next(r for r in dynamics if r['map_id']==mid and r['model']==name and r['source_case']==item['row']['case_dir'])
        end=round(dyn['predicted_goal_time_s']/.02)+1 if dyn['predicted_goal_time_s'] is not None else predictions.shape[1]
        poses=np.concatenate((item['pose'][:item['anchor']],predictions[i,:end]));region=np.clip(np.searchsorted(edges,poses[:,0],side='right')-1,0,len(margins)-1)
        with torch.no_grad():lo,hi=[v.numpy() for v in envelope(torch.as_tensor(poses,dtype=torch.float32))]
        witness=clearance_witness(lo,hi,margins[region],world['bounds_xy_m'],boxes,names)
        step=witness['step'];witness.update(time_s=step*.02,phase=int(item['phase'][step]) if step<item['anchor'] else 2,source_case=item['row']['case_dir'],margin_source=str(a.root/'calibration.json'),margin_rule=cal['rule'],prediction_kind='measured_TRAIN_prefix' if step<item['anchor'] else 'predicted_departure')
        details.append(dict(route=item['row']['route'],speed=item['speed'],witness=witness,domain_checks=dyn['terminal_gate_checks'],terminal_gate_valid=dyn['terminal_gate_valid'],predicted_goal_time_s=dyn['predicted_goal_time_s']))
    for candidate,(route,speed) in enumerate(( (r,v) for r in range(9) for v in (1.8,1.9))):
        matched=[d for d in details if d['route']==route and abs(d['speed']-speed)<1e-4]
        limiting=min(matched,key=lambda d:d['witness']['expanded_clearance_m'])['witness']
        np.testing.assert_allclose(limiting['expanded_clearance_m'],plan['all_clearance_m'][candidate],atol=2e-6,rtol=0)
        admitted=plan['eligible'][candidate] and limiting['expanded_clearance_m']>=0
        reasons=[]
        if limiting['expanded_clearance_m']<0:reasons.append('clearance')
        if not plan['eligible'][candidate]:reasons.append('source_eligibility_or_predicted_domain_or_goal')
        records.append(dict(map_id=mid,model=name,candidate=candidate,route=route,speed=speed,admitted=bool(admitted),selected=bool(plan['feasible'] and plan['candidate']==candidate),reasons=reasons,limiting=limiting,source_diagnostics=matched))
(a.output/'candidate_constraints.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(candidates=len(records),admitted=sum(r['admitted'] for r in records),rejected=sum(not r['admitted'] for r in records),clearance_parity='all within 2e-6 m',scope='sampled AABB clearance; not continuous collision proof; dynamic flags have no extremum timestamp yet')))
