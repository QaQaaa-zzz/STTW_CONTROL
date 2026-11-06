"""Saved-data verification: measured-motion closure versus steering-model prediction."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid,solve_ivp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
index=json.loads((a.evaluation/'index.json').read_text());rows=[];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500'];alphas=[0,.25,.5,.75,1]
integ=lambda y,t:cumulative_trapezoid(y,t,axis=0,initial=0)
rmse=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
for case in index:
 run=Path(case['run']);d=np.load(run/'trajectory.npz');contract=json.loads((run/'contract.json').read_text());route=np.load(Path(contract['bundle'])/'route.npz');cc=json.loads(Path(contract['settings']['controller_config']).read_text());tr=d['lower_trace'].reshape(-1,6,11);shared_count=int(tr[:,:,6].astype(bool).sum(0).min());count=int(tr[:,:5,6].astype(bool).sum(0).min());t=np.arange(count)*.005
 tree=cKDTree(route['xy']);segments=np.diff(route['xy'],axis=0);span=np.diff(route['s']);base=route['xy'][:-1]
 def project(xy):
  # Original route is a non-self-intersecting single bend; adjacent segments
  # around eight nearest vertices retain the continuous segment projection.
  _,near=tree.query(xy,k=8);ix=np.clip(np.concatenate([near,near-1],axis=1),0,len(segments)-1)
  v=segments[ix];q=base[ix];u=np.clip(((xy[:,None]-q)*v).sum(-1)/(v*v).sum(-1),0,1);pts=q+u[...,None]*v
  best=((xy[:,None]-pts)**2).sum(-1).argmin(1);rr=np.arange(len(xy));i=ix[rr,best];ss=route['s'][i]+u[rr,best]*span[i];delta=xy-pts[rr,best];hh=np.interp(ss,route['s'],route['heading']);cross=-np.sin(hh)*delta[:,0]+np.cos(hh)*delta[:,1]
  return ss,cross,hh
 fig,axs=plt.subplots(5,3,figsize=(16,17),layout='constrained');saved={}
 for lane,alpha in enumerate(alphas):
  xy=d['pose'][:count,lane,:2];m=d['measurement'][:count,lane];vel=d['body_state'][:count,lane,contract['bodies'].index('chasis'),7:9];v=np.linalg.norm(vel,axis=1);course=np.unwrap(np.arctan2(vel[:,1],vel[:,0]));omega=np.gradient(course,.005,edge_order=2);kappa=omega/v
  ss,ey,hh=project(xy);theta=course-hh;theta=(theta+np.pi)%(2*np.pi)-np.pi
  k2=np.cos(cc['caster'])*np.tan(m[:,2])/cc['wheelbase'];k3=k2/np.cos(m[:,0]);phi_qs=np.arctan(v*v*kappa/9.81)
  predictions={'world_velocity_closure':xy[0]+integ(vel,t)}
  for name,k in [('measured_curvature_closure',kappa),('steer_2d',k2),('steer_roll_3d',k3)]:
   psi=course[0]+integ(v*k,t);predictions[name]=xy[0]+integ(v[:,None]*np.stack([np.cos(psi),np.sin(psi)],axis=1),t)
  def ode(tt,z):
   speed=float(np.interp(tt,t,v));kv=float(np.interp(tt,t,kappa));kr=float(np.interp(z[0],route['s'],route['curvature']));ds=speed*np.cos(z[2])/(1-kr*z[1]);return [ds,speed*np.sin(z[2]),speed*kv-kr*ds]
  sol=solve_ivp(ode,(t[0],t[-1]),[ss[0],ey[0],theta[0]],t_eval=t,max_step=.005,rtol=1e-6,atol=1e-8)
  ell=integ(v,t);kr_linear=np.interp(ss[0]+ell,route['s'],route['curvature']);linear_theta=theta[0]+integ(kappa-kr_linear,ell);linear_ey=ey[0]+integ(linear_theta,ell)
  plateau=(ss>=3.499)&(ss<=5.819);steady=plateau
  metrics={};ax=axs[lane];ax[0].plot(*xy.T,color='black',label='Recorded XY');ax[1].plot(t,ey,color='black',label='Recorded geometric error')
  for name,pred in predictions.items():
   _,ep,_=project(pred);metrics[name]=dict(paired_path_rmse_m=rmse(ep[:shared_count]),xy_error_rmse_m=rmse(np.linalg.norm(pred-xy,axis=1)),ey_reconstruction_rmse_m=rmse(ep-ey),path_rmse_m=rmse(ep));saved[f'a{lane}_{name}_xy']=pred
   ax[0].plot(*pred.T,label=name,lw=.9);ax[1].plot(t,ep,label=name,lw=.9)
  if sol.success:
   metrics['nonlinear_frenet_closure']=dict(paired_path_rmse_m=rmse(sol.y[1,:shared_count]),ey_reconstruction_rmse_m=rmse(sol.y[1]-ey),path_rmse_m=rmse(sol.y[1]));ax[1].plot(t,sol.y[1],'--',label='Nonlinear Frenet');saved[f'a{lane}_frenet']=sol.y
  else:metrics['nonlinear_frenet_closure']={'error':sol.message}
  metrics['linear_curvature_integral']=dict(paired_path_rmse_m=rmse(linear_ey[:shared_count]),ey_reconstruction_rmse_m=rmse(linear_ey-ey),path_rmse_m=rmse(linear_ey));ax[1].plot(t,linear_ey,':',label='Linear double integral')
  ax[2].plot(t,kappa,label='Motion curvature');ax[2].plot(t,k2,label='Steer 2D');ax[2].plot(t,k3,label='Steer + roll 3D');ax[2].plot(t,np.interp(ss,route['s'],route['curvature']),':',label='Reference curvature')
  ax[0].set_aspect('equal');ax[0].set(xlabel='World x (m)',ylabel='World y (m)');ax[1].set(xlabel='Time (s)',ylabel='Signed path error (m)');ax[2].set(xlabel='Time (s)',ylabel='Curvature (1/m)')
  for aa in ax:aa.grid(alpha=.2);aa.set_title(f'alpha {alpha:g}');aa.legend(fontsize=6)
  u=tr[:count,lane,10];delta_ref=tr[:count,lane,9];A=np.cos(cc['caster'])/(cc['wheelbase']*cc['gravity']);B=cc['cg_forward']*cc['trail']*np.cos(cc['caster'])/(cc['wheelbase']*cc['cg_height']);phi_ref=(A*np.maximum(u,cc['minimum_speed'])**2-B)*delta_ref
  row=dict(terrain=case['terrain'],speed=case['user_speed_m_s'],alpha=alpha,n=count,window_s=count*.005,paired_window_s=shared_count*.005,paired_geometric_path_rmse_m=rmse(ey[:shared_count]),paired_speed_rmse_m_s=rmse(tr[:shared_count,lane,2]-case['user_speed_m_s']),logged_path_rmse_m=rmse(tr[:count,lane,3]),geometric_path_rmse_m=rmse(ey),speed_rmse_m_s=rmse(tr[:count,lane,2]-case['user_speed_m_s']),peak_path_m=float(abs(ey).max()),metrics=metrics,ecbc_A=A,ecbc_B=B,missing_contact_point_data='wheel_contact_points' not in d.files)
  if steady.any():row['constant_curvature_segment']=dict(samples=int(steady.sum()),speed=float(v[steady].mean()),curvature=float(kappa[steady].mean()),roll_deg=float(np.rad2deg(m[steady,0].mean())),qs_roll_deg=float(np.rad2deg(phi_qs[steady].mean())),qs_roll_error_rmse_deg=rmse(np.rad2deg(phi_qs[steady]-m[steady,0])),k2_error_rmse=rmse(k2[steady]-kappa[steady]),k3_error_rmse=rmse(k3[steady]-kappa[steady]),wheel_speed=float(u[steady].mean()),steer_ref=float(delta_ref[steady].mean()),ecbc_roll_ref_deg=float(np.rad2deg(phi_ref[steady].mean())),lateral_acceleration_mean=float((v*v*kappa)[steady].mean()),lateral_acceleration_max=float((v*v*kappa)[steady].max()))
  rows.append(row);saved.update({f'a{lane}_actual_xy':xy,f'a{lane}_actual_ey':ey,f'a{lane}_linear_ey':linear_ey,f'a{lane}_curvature':kappa,f'a{lane}_course':course,f'a{lane}_planar_speed':v,f'a{lane}_qs_roll':phi_qs,f'a{lane}_ecbc_roll_ref':phi_ref})
 stem=case['terrain']+'_v'+str(case['user_speed_m_s']).replace('.','p');fig.suptitle(stem+f' | five alphas, full H75 common 0–{count*.005:.2f}s | measured-motion closure versus uncalibrated steering models\nSame recorded initial XY/course; no fitted coefficients; no new simulation. Linear approximation may drift.');fig.savefig(a.output/(stem+'.png'),dpi=125);plt.close(fig);np.savez_compressed(a.output/(stem+'.npz'),time=t,**saved)
(a.output/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'conditions':len(index),'alpha_cases':len(rows),'output':str(a.output)}))

fig,axs=plt.subplots(2,5,figsize=(20,8),layout='constrained')
for case,ax in zip(index,axs.flat):
 rr=[r for r in rows if r['terrain']==case['terrain'] and r['speed']==case['user_speed_m_s']]
 yy=[r['paired_speed_rmse_m_s'] for r in rr]
 ax.plot([r['paired_geometric_path_rmse_m'] for r in rr],yy,'ko-',label='Recorded')
 for method,style in [('nonlinear_frenet_closure','--'),('linear_curvature_integral',':')]:
  ax.plot([r['metrics'][method]['paired_path_rmse_m'] for r in rr],yy,style,marker='.',label=method)
 ax.scatter([r['paired_geometric_path_rmse_m'] for r in rr],yy,c=colors,zorder=5,s=25)
 ax.set(title=f"{case['terrain']} | {case['user_speed_m_s']} m/s\nshared 0–{rr[0]['paired_window_s']:.2f}s",xlabel='Path RMSE (m)',ylabel='Speed RMSE (m/s)');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=7);fig.suptitle('Performance curve reconstructed from measured motion | DEV, one initial state\nSame per-condition time window; no fitting to error points. This is kinematic closure, not independent dynamics prediction.')
fig.savefig(a.output/'reconstructed_tradeoffs.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,5,figsize=(20,8),layout='constrained')
for case,ax in zip(index,axs.flat):
 run=Path(case['run']);contract=json.loads((run/'contract.json').read_text());route=np.load(Path(contract['bundle'])/'route.npz')
 stem=case['terrain']+'_v'+str(case['user_speed_m_s']).replace('.','p');saved=np.load(a.output/(stem+'.npz'))
 ax.plot(*route['xy'].T,':',color='grey',label='Original reference')
 for lane,alpha in enumerate(alphas):
  xy=saved[f'a{lane}_actual_xy'];ax.plot(*xy.T,color=colors[lane],label=f'alpha {alpha:g}');ax.scatter(*xy[-1],color=colors[lane],marker='x',s=20)
 allxy=np.concatenate([saved[f'a{i}_actual_xy'] for i in range(5)]);lo=allxy.min(0)-.5;hi=allxy.max(0)+.5;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
 ax.set(title=f"{case['terrain']} | {case['user_speed_m_s']} m/s",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=7);fig.suptitle('Original H75 + Local300 | recorded XY, all five alphas | DEV, one initial state per condition\nFull recorded 10 s; crosses are observation endpoints. Offline reanalysis; no new physics.')
fig.savefig(a.output/'recorded_xy_overview.png',dpi=150);plt.close(fig)
