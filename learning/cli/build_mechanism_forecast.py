"""Map DEV operating states, then evaluate fixed physical recovery/domain margins.

No binary outcome is a predictor input. Each DEV physical group supplies one
scenario; new alpha values interpolate the critical state, not success labels.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_roll_contacts import roll_state
from sttw_control.margin_theory import recovery_margin_batch,tracking_domain_margins


def collect(run,family,controller):
    jobs=json.loads((run/'jobs.json').read_text());metrics=json.loads((run/'analysis_summary.json').read_text())['rows'];rows=[]
    for job in jobs:
        height=float(json.loads((Path(job['terrain'])/'identity.json').read_text())['height_m'])
        if family=='gravel' and height==0:continue
        folder=run/job['name'];c=json.loads((folder/'contract.json').read_text());z=np.load(folder/'trajectory.npz');route=np.load(Path(c['bundle'])/'route.npz');cfg=json.loads((Path(c['bundle'])/'config.json').read_text())['follower'];n=len(c['settings']['alphas']);trace=z['lower_trace'].reshape(-1,n,11);time=(np.arange(len(trace))+1)*c['lower_dt']
        for lane,label in enumerate(c['settings']['case_labels']):
            if c['settings']['zero_middle'][lane] or 'duplicate' in label:continue
            valid=trace[:,lane,6].astype(bool);phi,omega=roll_state(z['body_state'][:,lane,c['bodies'].index('chasis')]);speed=trace[:,lane,2];delta=z['measurement'][:,lane,2]
            sampled=np.flatnonzero(valid)[::4];ids=sampled[(time[sampled]>=1.1)&(speed[sampled]>=.5)&(abs(phi[sampled])<.7)&(abs(delta[sampled])<=.8)]
            states=np.stack([phi[ids],omega[ids],delta[ids]],-1);margin=np.concatenate([recovery_margin_batch(speed[ids[k:k+128]],states[k:k+128],controller) for k in range(0,len(ids),128)]);worst=int(np.argmin(margin));critical=ids[worst]
            ix=np.flatnonzero(valid);pose=z['pose'][ix,lane];progress=z['progress'][ix,lane];v=speed[ix]
            look=np.clip(cfg['lookahead_time_s']*np.maximum(v,cfg['lookahead_min_speed_m_s']),cfg['lookahead_min_m'],cfg['lookahead_max_m']);target=np.stack([np.interp(progress+look,route['s'],route['xy'][:,k]) for k in range(2)],-1);difference=target-pose[:,:2];forward=np.cos(pose[:,2])*difference[:,0]+np.sin(pose[:,2])*difference[:,1];heading=np.interp(progress,route['s'],route['heading'])-pose[:,2];heading=(heading+np.pi)%(2*np.pi)-np.pi
            dm,hm=tracking_domain_margins(forward,heading,cfg['preview_body_x_min_m']);didx=int(np.argmin(np.minimum(dm,hm/(np.pi/2))))
            m=next(r for r in metrics if r['case']==job['name'] and r['label']==label)
            rows.append(dict(family=family,height_m=height,requested_speed_m_s=float(c['settings']['user_speed_m_s']),alpha=float(c['settings']['alphas'][lane]),group=job['physical_seed'],case=job['name'],source=str(folder),critical_time_s=float(time[critical]),critical_front_force_N=float(np.linalg.norm(z['contact_force'][critical,lane,c['bodies'].index('frontwheel')])),critical_rear_force_N=float(np.linalg.norm(z['contact_force'][critical,lane,c['bodies'].index('rearwheel')])),critical_state=[float(speed[critical]),*map(float,states[worst])],min_recovery_margin_rad=float(margin[worst]),domain_state=[float(forward[didx]),float(heading[didx])],min_domain_margin_m=float(dm.min()),min_heading_margin_rad=float(hm.min()),path_peak_m=float(m['path_peak_m']),speed_rmse_m_s=float(m['speed_rmse_m_s']),observed_cause=m['cause'],observed_tasks=m['tasks']))
        print(family,job['name'],len(rows),flush=True)
    return rows


def predict(rows,family,height,speed,alpha,controller):
    members=[]
    for group in [101,102]:
        source=sorted([r for r in rows if r['family']==family and r['height_m']==height and r['requested_speed_m_s']==speed and r['group']==group],key=lambda r:r['alpha']);axis=[r['alpha'] for r in source]
        interpolate=lambda key:np.array([np.interp(alpha,axis,[r[key][j] for r in source]) for j in range(len(source[0][key]))])
        state=interpolate('critical_state');domain=interpolate('domain_state');roll=float(recovery_margin_batch(np.array([state[0]]),state[None,1:],controller)[0]);dm,hm=tracking_domain_margins(*domain)
        path=float(np.interp(alpha,axis,[r['path_peak_m'] for r in source]));ev=float(np.interp(alpha,axis,[r['speed_rmse_m_s'] for r in source]));score=min(roll/.7,float(dm),float(hm)/(np.pi/2))
        front=float(np.interp(alpha,axis,[r['critical_front_force_N'] for r in source]));rear=float(np.interp(alpha,axis,[r['critical_rear_force_N'] for r in source]))
        members.append(dict(group=group,critical_front_force_N=front,critical_rear_force_N=rear,critical_state=state.tolist(),domain_state=domain.tolist(),recovery_margin_rad=roll,domain_margin_m=float(dm),heading_margin_rad=float(hm),bottleneck_score=score,path_peak_m=path,speed_rmse_m_s=ev,physical_feasible=score>=0))
    return dict(alpha=alpha,members=members,physical_feasibility_vote=float(np.mean([m['physical_feasible'] for m in members])),mean_bottleneck_score=float(np.mean([m['bottleneck_score'] for m in members])))


TASK_THRESHOLDS={'A':(.3,.2),'B':(.6,.1),'C':(.4,.15)}


def task_selection(predictions,thresholds=TASK_THRESHOLDS):
    """Select within physical and error constraints; hold remains a measured outcome."""
    selected={}
    for task,(path_tol,speed_tol) in thresholds.items():
        for prediction in predictions:
            members=prediction['members']
            eligible=[m['physical_feasible'] and m['path_peak_m']<=path_tol and m['speed_rmse_m_s']<=speed_tol for m in members]
            error=sum((m['path_peak_m']/path_tol)**2+(m['speed_rmse_m_s']/speed_tol)**2 for m in members)/len(members)
            prediction.setdefault('task_constraint_forecast',{})[task]=dict(feasible_vote=sum(eligible)/len(eligible),normalized_squared_error=error)
        def rank(p):
            t=p['task_constraint_forecast'][task]
            return (t['feasible_vote'],p['physical_feasibility_vote'],p['mean_bottleneck_score'] if t['feasible_vote'] else -t['normalized_squared_error'],-t['normalized_squared_error'],-p['alpha'])
        chosen=max(predictions,key=rank)
        selected[task]=dict(alpha=chosen['alpha'],feasible_vote=chosen['task_constraint_forecast'][task]['feasible_vote'],mode='task_constrained' if chosen['task_constraint_forecast'][task]['feasible_vote'] else 'least_error_physical_fallback')
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--ridge',type=Path,required=True);p.add_argument('--gravel',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-states',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=json.loads(a.controller.read_text())
    rows=json.loads((a.output/'operating_states.json').read_text()) if a.reuse_states else collect(a.ridge,'ridge',c)+collect(a.gravel,'gravel',c)
    (a.output/'operating_states.json').write_text(json.dumps(rows,indent=2))
    alphas=sorted(set(np.linspace(0,1,11).round(2).tolist()+[.25,.75]));contexts=sorted(set((r['family'],r['height_m'],r['requested_speed_m_s']) for r in rows));forecasts=[]
    for family,h,v in contexts:
        predictions=[predict(rows,family,h,v,alpha,c) for alpha in alphas]
        selected=max(predictions,key=lambda x:(x['physical_feasibility_vote'],x['mean_bottleneck_score'],-x['alpha']))['alpha']
        forecasts.append(dict(family=family,height_m=h,speed_m_s=v,predictions=predictions,selected_alpha_physical=selected,selected_alpha_tasks=task_selection(predictions)))
    validation=[]
    for r in rows:
        if r['group']!=103:continue
        pr=predict(rows,r['family'],r['height_m'],r['requested_speed_m_s'],r['alpha'],c)
        validation.append(dict(case=r['case'],alpha=r['alpha'],predicted_vote=pr['physical_feasibility_vote'],actual_full_window=r['observed_cause']=='full_window',observed_cause=r['observed_cause']))
    artifact=dict(status='S2 candidate before S3 freeze',inputs=['terrain family','height','requested speed','candidate alpha'],mapping='Two DEV scenario maps of critical operating state and worst geometric domain state; linear alpha interpolation; no height/speed extrapolation',fit_groups=[101,102],assessment_group=103,physical_rules='Exact rate/travel nominal recovery envelope and original follower domain inequalities',probability_semantics='Two-scenario physical-feasibility vote; not calibrated task-completion probability',task_thresholds=TASK_THRESHOLDS,task_selection_semantics='Physical feasibility and path-peak/speed-RMSE constraints; final hold is not forecast and remains an independent measured task requirement',controller=c,contexts=forecasts,validation=validation)
    (a.output/'mechanism_forecast.json').write_text(json.dumps(artifact,indent=2));print('rows',len(rows),'contexts',len(contexts))

if __name__=='__main__':main()
