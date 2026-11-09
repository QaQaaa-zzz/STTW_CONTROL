"""Fit frozen first-pass endpoint planners from declared short-station TRAIN rows."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sttw_control.jump_connection import decision_features,fields
from sttw_control.approach_planner import fit_model

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--completion-protocol',choices=['original_strict','contact_filtered_20ms'],default='original_strict');a=p.parse_args()
m=json.loads((a.run/'manifest.json').read_text());candidates=[dict(speed_m_s=v,alpha=alpha) for v in (1.8,2.) for alpha in (0.,.5,1.)]
rows=[];identities=[];cases=[];skipped=[]
for case in m['cases']:
    spec=json.loads(Path(case['config']).read_text());batch=Path(spec['output'])/'batch_results.json'
    if batch.exists():
        for lane in json.loads(batch.read_text())['cases']:
            cases.append(dict(case,config=str(Path(lane['output'])/'config.json'),**lane['candidate']))
    else:cases.append(case)
for case in cases:
    spec=json.loads(Path(case['config']).read_text());out=Path(spec['output']);event=out/'connection_decision.json'
    if not event.exists():
        skipped.append(dict(root=case['origin_id'],output=str(out),reason='decision_not_reached'));continue
    protocol=json.loads((out/'protocol.json').read_text());decision=json.loads(event.read_text())
    state,_=decision_features(decision,out,protocol,spec)
    summary=json.loads((out/'chain_summary.json').read_text());d=np.load(out/'trajectory.npz');trace=d['trace'][d['valid'][:,0],0]
    checks=summary['checks'] if a.completion_protocol=='original_strict' else summary['contact_filtered_protocol']['checks']
    handoff=out/'handoff.json';entry=None;prefix=False;suffix=False
    if handoff.exists():
        h=json.loads(handoff.read_text());root=np.array(h['root'][0]);qx,qy,qz,qw=root[3:7]
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));sample=trace[np.argmin(abs(trace[:,0]-h['time']))]
        entry=[float(root[1]),float(yaw),float(np.cos(yaw)*root[7]+np.sin(yaw)*root[8]),float(-np.sin(yaw)*root[7]+np.cos(yaw)*root[8]),float(sample[5]),float(sample[6])]
        pre=trace[(trace[:,0]>=decision['time'])&(trace[:,0]<=h['time'])]
        post=trace[trace[:,0]>=h['time']]
        prefix=bool(summary['decision_prefix']['corridor_intact_at_decision'] and (abs(pre[:,2])<=1).all())
        suffix=bool(all(checks[k] for k in ('full_window','stable_landing','ground_return','continued_time','continued_forward','final_goal')) and (abs(post[:,2])<=1).all())
    ground=json.loads((out/'ground_controller.json').read_text());plant=json.loads(Path(spec['plant_config']).read_text())
    identities.append(dict(ground=ground['upper_sha256'],jump=protocol['policies'][0]['artifact_sha256'],
        model_contract=hashlib.sha256(Path(plant['asset']).with_name('contract.json').read_bytes()).hexdigest(),physics_dt=spec['physics_dt'],zero_lower=spec['zero_lower']))
    if a.completion_protocol!='original_strict':
        identities[-1]['execution_contract']=dict(completion_protocol=a.completion_protocol,landing_return=spec['landing_return'],seconds=spec['seconds'],completion={k:v for k,v in spec['completion'].items() if k!='waypoints'})
    action=candidates.index(dict(speed_m_s=case['speed_m_s'],alpha=case['alpha']))
    rows.append(dict(root=case['origin_id'],state=state,action=action,entry=entry,prefix_qualified=prefix,suffix_complete=suffix,full_complete=summary['complete'] if a.completion_protocol=='original_strict' else summary['contact_filtered_protocol']['complete'],original_strict_complete=summary['complete'],completion_protocol=a.completion_protocol,output=str(out)))
if not rows:raise ValueError('No observed decisions in declared TRAIN data')
assert all(identity==identities[0] for identity in identities)
names=[x[0] for x in fields];base=['cross_track','heading_error','speed','remaining_path_to_release']
for name,extra,entry_indices in [('geometry',[],[0,1,2]),('dynamics',['lateral_velocity_heading_frame','roll','roll_rate'],list(range(6)))]:
    model=fit_model(rows,[names.index(k) for k in base+extra],entry_indices,candidates,identities[0])
    model.update(training_roots=m['roots'],source_manifest=str(a.run/'manifest.json'),role='TRAIN fit; fixed regularization; evaluate on validation_tuning, not independent TEST',entry_names=['lateral_m','heading_rad','body_forward_m_s','body_lateral_m_s','roll_rad','roll_rate_rad_s'])
    (a.run/('planner_'+name+'.json')).write_text(json.dumps(model,indent=2)+'\n')
(a.run/'planner_coverage.json').write_text(json.dumps(dict(declared=len(cases),observed_decisions=len(rows),skipped=skipped,completion_protocol=a.completion_protocol),indent=2)+'\n')
(a.run/'planner_training_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),prefix_qualified=sum(r['prefix_qualified'] for r in rows),suffix_successes=sum(r['suffix_complete'] for r in rows))))
