#!/usr/bin/env python3
"""Rebuild stage-B metrics and figures from saved trajectories, no physics."""
import argparse,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.path_frequency import signal_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((run/'manifest.json').read_text());cfg=manifest['config'];results={}
 xyfig,xyaxes=plt.subplots(1,2,figsize=(14,6))
 for index,case in enumerate(cfg['stage_B_baseline']['cases']):
  name=case['name'];d=dict(np.load(run/f'{name}.npz'));path=dict(np.load(run/f'{name}_path.npz'));t=d['time'].astype(float)+.005;n=len(t);h=case['horizon_s'];v=case['v_user'];b=cfg['stage_B_baseline']
  complete=n==round(h/.005);last=(t>h-b['hold_s']+1e-5)&(t<=h+1e-5);ev=d['actual_forward_speed']-v;ey=d['path_cross_track'];eh=d['path_heading_error']
  physical=bool(d['physical_failure'].any());peak=float(d['peak_roll'].max());domain=bool(d['domain_exit'].any());fault=bool(d['policy_fault'].any()|d['lower_fault'].any())
  held=bool(last.sum()==200 and np.all(abs(ey[last])<=b['final_y_error_m']) and np.all(abs(eh[last])<=b['final_path_heading_error_rad']) and np.all(abs(ev[last])<=b['final_speed_error_m_s']))
  crossed=(d['goal_section_signed_distance']>=0)&(d['path_progress']>=float(path['goal'])-1e-4)
  goal=bool(crossed.any()) if name!='straight' else None
  rmse=lambda x:float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
  metrics=dict(completed=complete,control_ticks=n,observed_s=n*.005,stop_reason=str(d['stop_reason']),physical_failure=physical,policy_fault=fault,domain_exit=domain,peak_roll_rad=peak,path_rmse_m=rmse(ey),heading_rmse_rad=rmse(eh),speed_rmse_m_s=rmse(ev),final_hold=held,goal_crossed=goal,goal_cross_time_s=float(t[np.flatnonzero(crossed)[0]]) if crossed.any() and name!='straight' else None,final_path_error_m=float(ey[-1]),final_heading_error_rad=float(eh[-1]),final_speed_error_m_s=float(ev[-1]),final_progress_m=float(d['path_progress'][-1]),goal_progress_m=float(path['goal']),lower_speed_rmse_m_s=rmse(d['lower_error'][:,0]),lower_steer_rmse_rad=rmse(d['lower_error'][:,1]),final_hold_violations=dict(samples=int(last.sum()),lateral=int((abs(ey[last])>b['final_y_error_m']).sum()),heading=int((abs(eh[last])>b['final_path_heading_error_rad']).sum()),speed=int((abs(ev[last])>b['final_speed_error_m_s']).sum())))
  metrics['accepted']=bool(complete and not physical and not fault and not domain and peak<=b['working_roll_rad'] and metrics['path_rmse_m']<=b['nominal_y_error_rmse_m'] and held and (goal is None or goal))
  windows={'initial_0_1s':t<=1+1e-5,'full':np.ones(n,bool),'last1s':last,'curvature_transition':((d['path_curvature']>1e-6)&(d['path_curvature']<.5-1e-6))}
  metrics['windows']={key:dict(samples=int(mask.sum()),path_rmse_m=rmse(ey[mask]),speed_rmse_m_s=rmse(ev[mask]),heading_rmse_rad=rmse(eh[mask])) if mask.any() else dict(samples=0) for key,mask in windows.items()}
  metrics['clip_counts']={key:val.sum(0).tolist() for key,val in dict(final=d['final_command']!=d['u_prelimit'],residual=abs(d['requested_residual'])>np.array([1.5,10]),reference=d['reference_clip_channels']).items()}
  ix=np.arange(3,n,4);metrics['signal_50Hz']={key:signal_metrics(val[ix])[0] for key,val in dict(nominal_steer=d['nominal'][:,1],governed_steer=d['governed'][:,1],actual_steer=d['actual_delta']).items()}
  diagnosis=dict(initial_error=float(d['initial_path_error']),last1_max_heading=float(abs(eh[last]).max()) if last.any() else None,last1_max_lateral=float(abs(ey[last]).max()) if last.any() else None,last1_max_speed=float(abs(ev[last]).max()) if last.any() else None,all_lower_finite=bool(np.isfinite(d['lower_action']).all() and not d['lower_fault'].any()),max_abs_nominal_speed_rate=float(abs(d['nominal_rates'][:,0]).max()),max_abs_nominal_steer_rate=float(abs(d['nominal_rates'][:,1]).max()),max_projection_increment=float(np.diff(d['path_progress']).max()),min_preview_body_x=float(d['preview_body'][:,0].min()))
  tail=t[ix]>h-4+1e-5
  for key,value in [('governed',d['governed'][:,1]),('actual',d['actual_delta']),('heading_error',eh)]:
   sm=signal_metrics(value[ix][tail])[0];diagnosis[key+'_last4s_psd_peak_hz']=sm['peak_frequency_hz'];diagnosis[key+'_last4s_peak_to_peak']=sm['peak_to_peak']
  metrics['endpoint_diagnosis']=diagnosis
  # Cross-track identity reconstructed from the logged geometric projection.
  # This is not an independent projection algorithm audit.
  points=d['reference_xy'];delta=d['actual_xy']-points;reconstructed=-np.sin(d['reference_yaw_unwrapped'])*delta[:,0]+np.cos(d['reference_yaw_unwrapped'])*delta[:,1]
  metrics['cross_track_reconstruction_max_abs']=float(np.max(abs(reconstructed-ey)));assert metrics['cross_track_reconstruction_max_abs']<1e-5
  np.testing.assert_allclose(d['lower_error'],np.column_stack([d['actual_forward_speed'],d['actual_delta']])-d['governed'],atol=2e-7)
  assert np.all(d['offsets']==0) and np.all(d['filtered_offset']==0) and np.all(d['target_offset']==0)
  assert np.all(np.diff(d['path_progress'])>=0)
  results[name]=metrics
  fig,axes=plt.subplots(6,2,figsize=(15,20))
  for ax in [axes[0,0],xyaxes[index]]:
   visible=path['s']<=float(d['path_progress'].max())+2.
   ax.plot(*path['xy'][visible].T,'k--',label='fixed original path (display extent only)')
   ax.plot(*np.vstack([d['initial_xy'],d['actual_xy']]).T,label='zero upper + local300 + ECBC/ESO')
   ax.plot(*d['actual_xy'][-1],marker='x' if physical or domain or fault else 'o',label='true endpoint')
   if name!='straight':
    pt=np.array([np.interp(path['goal'],path['s'],path['xy'][:,i]) for i in range(2)]);angle=np.interp(path['goal'],path['s'],path['heading']);normal=np.array([-np.sin(angle),np.cos(angle)])
    segment=np.array([pt-normal*.6,pt+normal*.6]);ax.plot(*segment.T,':',label='goal section turn_end+4m')
   ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_title(name+f" | accepted={metrics['accepted']}");ax.legend(fontsize=7);ax.grid(alpha=.25)
  axes[0,1].axhline(.05,color='tab:orange',ls=':',label='heading hold band ±.05rad');axes[0,1].axhline(-.05,color='tab:orange',ls=':')
  axes[0,1].plot(t,ey,label='left-positive cross-track m');axes[0,1].plot(t,eh,label='path tangent - actual heading rad');axes[0,1].axhline(.1,color='gray',ls=':');axes[0,1].axhline(-.1,color='gray',ls=':')
  for ch in range(2):
   for val,label,style in [(d['pp_target'],'PP target','-'),(d['nominal'],'published nominal','--'),(d['governed'],'governed',':'),(np.column_stack([d['actual_forward_speed'],d['actual_delta']]),'actual','-')]:axes[1,ch].plot(t,val[:,ch],style,label=label,lw=.9)
   if ch==0:axes[1,ch].axhline(v,color='k',label='external v_user')
   for key,label in [('target_offset','network target'),('filtered_offset','filtered'),('offsets','executed')]:axes[2,ch].plot(t,d[key][:,ch],label=label)
   axes[3,ch].plot(t,d['lower_error'][:,ch],label='lower actual-governed')
   for key,label in [('u_nom','ECBC'),('applied_residual','lower residual'),('final_command','final command')]:axes[4,ch].plot(t,d[key][:,ch],label=label,lw=.7)
   for r in range(1,4):axes[r,ch].set_ylabel(['speed m/s','steer rad'][ch])
   axes[4,ch].set_ylabel(['front rad/s','rear rad/s'][ch])
  axes[5,0].plot(t,d['phi'],label='roll rad');axes[5,0].plot(t,d['peak_roll'],label='substep abs peak rad');axes[5,0].axhline(.302,color='r',ls='--')
  axes[5,1].plot(t,d['path_progress'],label='monotone progress m');axes[5,1].plot(t,d['preview_body'][:,0],label='preview body X m');axes[5,1].plot(t,d['preview_body'][:,1],label='preview body Y m')
  for ax in axes.flat:
   if ax is axes[0,0]:continue
   ax.set_xlabel('Time (s)');ax.legend(fontsize=7);ax.grid(alpha=.2);ax.axvspan(h-1,h,color='gray',alpha=.08)
   if metrics['goal_cross_time_s'] is not None:ax.axvline(metrics['goal_cross_time_s'],color='green',ls=':',alpha=.4)
   if not complete:ax.axvline(t[-1],color='r',ls=':')
  fig.suptitle(f'{name} | frozen local300 | zero upper | seed{manifest["seed"]} | {n}/'+str(round(h/.005))+' lower ticks\nFixed geometry; no reanchor, time-target scoring, speed regulation or post-hoc smoothing; last1s shaded; no external disturbance')
  fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/f'{name}_overview.png',dpi=130);plt.close(fig)
  # Independent NumPy reconstruction of all supplied cost components.
  H=lambda x:np.where(abs(x)<=1,x*x,2*abs(x)-1)
  pos=lambda x:np.maximum(x,0)
  chi=d['chi'].astype(float);alpha=0.;roll=d['peak_roll'].astype(float)
  raw_cost=dict(speed=(10-9*chi)*H(ev.astype(float)/.1),path=10*(H(ey.astype(float)/.1)+.5*H(eh.astype(float)/.1)),primary=20*chi*H(pos(abs(ey.astype(float))-.1)/.1),roll=40*H(pos(roll-.26)/.04),working_roll=120*H(pos(roll-.30)/.02),roll_rate=H(pos(abs(d['phi_dot'].astype(float))-.8)/.8),overspeed=2*H(pos(ev.astype(float)-.05)/.1),low_speed=4*H(pos(1.5-d['actual_forward_speed'].astype(float))/.2),offset_magnitude=np.zeros(n),offset_rate=np.zeros(n))
  effective={key:np.minimum(value,cfg['path_reward']['component_caps'][key]) for key,value in raw_cost.items()}
  rebuilt=-.1*.005*sum(effective.values())
  regular=d['tick_reward'].astype(float)
  metrics['reward_reconstruction_max_abs']=float(np.max(abs(rebuilt-regular)))
  assert metrics['reward_reconstruction_max_abs']<2e-5
  scored=d['scored_tick_reward'];steps=np.arange(1,n+1)
  if physical:
   block_start=((n-1)//4)*4;remaining=int(np.ceil(h/.02))-(n-1)//4;gamma=cfg['training_future_phase_C']['gamma']
   terminal=-5-.1*.02*5080.2*(1-gamma**remaining)/(1-gamma)
   expected=regular.copy();expected[block_start:]=0.;expected[-1]=terminal
   np.testing.assert_allclose(scored,expected,rtol=2e-5,atol=2e-4)
  else:
   np.testing.assert_array_equal(scored,d['tick_reward'])
  metrics['scored_reward_verified']=True
  rf,ra=plt.subplots(3,2,figsize=(15,12))
  ra[0,0].plot(steps,scored,label='per actual lower transition');ra[0,1].plot(steps,np.cumsum(scored,dtype=float),label='cumulative from first actual transition')
  for key,val in effective.items():
   component=-.1*.005*val
   ra[1,0].plot(steps,component,label=key);ra[1,1].plot(steps,np.cumsum(component),label=key)
  ra[2,0].plot(t,d['actual_forward_speed'],label='true forward speed');ra[2,0].plot(t,d['wheel_speed_proxy'],label='rear shaft x .1m');ra[2,0].plot(t,d['nominal'][:,0],label='nominal');ra[2,0].plot(t,d['governed'][:,0],label='governed');ra[2,0].axhline(v,color='k',ls=':',label='external v_user')
  ra[2,1].plot(t,d['wheel_speed_proxy']-d['actual_forward_speed'],label='proxy - true m/s')
  for ax in ra.flat:ax.grid(alpha=.25);ax.legend(fontsize=7)
  for rr in range(2):
   for ax in ra[rr]:ax.set_xlabel('Actual lower steps');ax.set_ylabel('Signed reward contribution')
  for ax in ra[2]:ax.set_xlabel('Time (s)');ax.set_ylabel('m/s')
  rf.suptitle(name+' | phase-C reward diagnostic alpha0, upper=0 | local300 | seed77001 | no training\nFailure replacement retained in total; ordinary components are not terminal penalty; no external disturbance');rf.tight_layout(rect=(0,0,1,.95));rf.savefig(out/f'{name}_reward_speed.png',dpi=120);plt.close(rf)
  data=out/'data';data.mkdir(exist_ok=True)
  shutil.copyfile(run/f'{name}.npz',data/f'{name}.npz');shutil.copyfile(run/f'{name}_path.npz',data/f'{name}_path.npz')
 xyfig.suptitle(f'Fixed paths / actual XY | local300 | upper=0 | seed{manifest["seed"]} | no training');xyfig.tight_layout();xyfig.savefig(out/'XY_baselines.png',dpi=150);plt.close(xyfig)
 (out/'endpoint_diagnosis.json').write_text(json.dumps({k:v['endpoint_diagnosis'] for k,v in results.items()},indent=2))
 (out/'stage_B_metrics.json').write_text(json.dumps(results,indent=2))
 for name in ['manifest.json','status.json']:shutil.copyfile(run/name,out/name)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
