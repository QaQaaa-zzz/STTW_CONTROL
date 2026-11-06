"""Geometric decomposition at saved poses; not a counterfactual dynamics replay."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',nargs='+',type=Path,required=True)
p.add_argument('--wheel-radii',nargs=3,type=float,required=True,help='Nominal ellipsoid radii in the wheel link frame, metres')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rows=[]
for run in a.runs:
 contract=json.loads((run/'contract.json').read_text());bodies=contract['bodies']
 with np.load(run/'trajectory.npz') as raw:
  state=raw['body_state'];alphas=contract['settings']['alphas']
  for lane,alpha in enumerate(alphas):
   if contract['settings']['zero_middle'][lane]:continue
   tag=f'alpha{alpha:g}'
   with np.load(run/'trail_geometry'/f'{tag}.npz') as z:
    wheel=state[:,lane,bodies.index('frontwheel')]
    rot=Rotation.from_quat(wheel[:,3:7]);up=np.tile([0.,0.,1.],(len(wheel),1))
    q_up=rot.apply(rot.inv().apply(up)*np.square(a.wheel_radii))
    q=wheel[:,:3]-q_up/np.sqrt((q_up*up).sum(1))[:,None]
    axis=z['axis'];o=z['axis_origin'];point=z['point']
    forward=Rotation.from_quat(state[:,lane,0,3:7]).apply([1.,0.,0.])
    direction=np.cross(rot.apply([0.,0.,1.]),up);direction/=np.linalg.norm(direction,axis=1)[:,None]
    direction*=np.where((direction*forward).sum(1)>=0,1.,-1.)[:,None]
    axis_gain=(axis*direction).sum(1)/axis[:,2]
    reference=((o-q)*direction).sum(1)+(q[:,2]-o[:,2])*axis_gain
    height=(point[:,2]-q[:,2])*axis_gain
    forward_shift=-((point-q)*direction).sum(1)
    plane=z['resultant_normal']-z['horizontal']
    reconstructed=reference+height+forward_shift+plane
    valid=z['valid']&z['raised'];negative=valid&(z['resultant_normal']<0)
    normal=z['force'].astype(float);length=np.linalg.norm(normal,axis=1);normal=np.divide(normal,length[:,None],out=np.full_like(normal,np.nan),where=length[:,None]>1e-3)
    qn_offset=rot.apply(rot.inv().apply(normal)*np.square(a.wheel_radii))
    qn=wheel[:,:3]-qn_offset/np.sqrt((qn_offset*normal).sum(1))[:,None]
    dn=np.cross(rot.apply([0.,0.,1.]),normal);dn/=np.linalg.norm(dn,axis=1)[:,None]
    dn*=np.where((dn*forward).sum(1)>=0,1.,-1.)[:,None]
    predicted=((o-qn)*dn).sum(1)+((qn-o)*normal).sum(1)*(axis*dn).sum(1)/(axis*normal).sum(1)
    predicted_negative=valid&(predicted<0)
    residual=predicted-z['resultant_normal']
    prediction=dict(raised_samples=int(valid.sum()),true_positive=int((negative&predicted_negative).sum()),false_negative=int((negative&~predicted_negative).sum()),false_positive=int((~negative&predicted_negative).sum()),true_negative=int((valid&~negative&~predicted_negative).sum()),trail_rmse_m=float(np.sqrt(np.mean(residual[valid]**2))),contact_point_rmse_m=float(np.sqrt(np.mean(np.sum((point[valid]-qn[valid])**2,axis=1)))))
    ids=np.flatnonzero(valid);i=ids[np.argmin(z['resultant_normal'][ids])]
    components=dict(predicted_ellipsoid_trail_m=predicted,reference_flat_support_m=reference,contact_height_term_m=height,contact_forward_term_m=forward_shift,normal_plane_term_m=plane,actual_trail_m=z['resultant_normal'])
    rows.append(dict(support_prediction=prediction,run=str(run),alpha=alpha,speed_m_s=contract['settings'].get('user_speed_m_s',2.3),raised_negative_samples=int(negative.sum()),negative_with_positive_flat_support=int((negative&(reference>0)).sum()),negative_also_horizontal=int((negative&(z['horizontal']<0)).sum()),closure_max_m=float(np.max(np.abs(reconstructed[valid]-z['resultant_normal'][valid]))),at_minimum=dict(time_s=float(z['time'][i]),**{k:float(v[i]) for k,v in components.items()}),negative_samples=[dict(time_s=float(z['time'][j]),**{k:float(v[j]) for k,v in components.items()}) for j in np.flatnonzero(negative)]))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'trail_decomposition.json').write_text(json.dumps(dict(wheel_radii_m=a.wheel_radii,definition='q is the lowest support point of a nominal ellipsoid at the recorded wheel pose; p is the measured mean contact point; a is the steering axis; d is horizontal wheel-forward. T_actual=T_horizontal(q)+(p_z-q_z)*(a dot d)/a_z-(p-q) dot d+(T_normal(p)-T_horizontal(p)).',scope='Algebraic geometry at recorded states, not simulated contact replacement, causal isolation, or exact SDF tire shape. The reference ellipsoid matches nominal asset radii; contact compliance and mesh approximation remain in p-q.',rows=rows),indent=2))
lines=['# Contact-trail geometric decomposition','','At each saved pose, compare the measured contact point with the lowest point of the nominal wheel ellipsoid on a horizontal plane. This is a geometric reference, not a dynamics intervention. The actual SDF/compliant contact and its measured normal are unchanged.','', '|Speed|Alpha|Negative samples|Positive reference among negatives|Reference at min mm|Height term mm|Forward term mm|Plane term mm|Actual min mm|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 q=r['at_minimum'];v=[q[k]*1000 for k in ['reference_flat_support_m','contact_height_term_m','contact_forward_term_m','normal_plane_term_m','actual_trail_m']]
 lines.append(f"|{r['speed_m_s']:g}|{r['alpha']:g}|{r['raised_negative_samples']}|{r['negative_with_positive_flat_support']}|"+'|'.join(f'{x:.3f}' for x in v)+'|')
lines+=['','All negative samples and formula definitions are retained in trail_decomposition.json. The terms sum to the measured-plane trail proxy. Normal-plane replacement alone cannot explain a sign reversal that is also present under horizontal projection; the reference-point change includes terrain contact migration and compliant-contact/model differences. It does not establish that negative trail causes failure.']
lines += ['', '## Ellipsoid support prediction without the measured contact point', '', 'Use q(n)=center-Qn/sqrt(n^T Qn), Q=R diag(radii^2) R^T, where n is the measured normal-resultant direction. Intersect the measured steering axis with the plane through q(n), then compute trail. The prediction uses pose, normal and nominal shape, but not the measured point p. It is conditional on recorded force direction, not an independent terrain/dynamics prediction.', '', '|Speed|Alpha|Raised samples|True negative-trail detections|Missed negatives|Predicted negative but observed positive|Trail RMSE mm|Contact point RMSE mm|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 q=r['support_prediction'];lines.append(f"|{r['speed_m_s']:g}|{r['alpha']:g}|{q['raised_samples']}|{q['true_positive']}|{q['false_negative']}|{q['false_positive']}|{q['trail_rmse_m']*1000:.3f}|{q['contact_point_rmse_m']*1000:.3f}|")
(a.output/'TRAIL_DECOMPOSITION.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
