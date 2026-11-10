"""Stream saved contact records into paired mechanism measures; no simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.contact_torque_geometry import contact_torque_time_batch

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
r=a.run;out=r/'mechanism_analysis';out.mkdir(exist_ok=True);jobs=json.loads((r/'jobs.json').read_text());rows=[];events=[]
torch.set_num_threads(2)
for job in jobs:
 d=r/job['name'];z=np.load(d/'trajectory.npz');index=json.loads((d/'physics_contacts/index.json').read_text());dt=index['physics_dt_s'];pieces=[]
 for chunk in index['chunks']:
  with np.load(d/'physics_contacts'/chunk['file']) as raw:
   x={k:raw[k] for k in raw.files};ticks,n=x['alive_before'].shape;record={'time_s':x['physics_step']*dt,'alive':x['alive_before']}
   for wheel in ('frontwheel','rearwheel'):
    counts=x[wheel+'_normal_count'][...,0];starts=x[wheel+'_normal_start'][...,0];slots=np.arange(int(counts.max()));ids=starts[...,None]+slots;mask=slots<counts[...,None];ids=np.minimum(ids,x[wheel+'_normal_force'].shape[1]-1);ti=np.arange(ticks)[:,None,None]
    f=x[wheel+'_normal_force'][ti,ids,0];points=x[wheel+'_normal_point'][ti,ids];direction=x[wheel+'_normal_direction'][ti,ids]
    f=np.where(mask,f,0);record[wheel+'_normal_N']=f.sum(-1)
    record[wheel+'_normal_vector_N']=(f[...,None]*direction).sum(-2)
    record[wheel+'_raised_contact']=(mask & (points[...,2]>.002) & (f>1.)).any(-1)
   torque=contact_torque_time_batch({k:torch.from_numpy(v) for k,v in x.items() if k=='body_state' or k.startswith('frontwheel_')},index['body_order'])
   record['steer_contact_Nm']=(torque['normal_moment']+torque['friction_moment']).numpy()
   bounds=np.asarray(index['motor_limits']);request=x['motor_requested_torque'];record['motor_clipped']=(request<bounds[:,0])|(request>bounds[:,1]);record['motor_requested_Nm']=request
   pieces.append(record)
 data={k:np.concatenate([c[k] for c in pieces]) for k in pieces[0]};np.savez_compressed(out/(job['name']+'.npz'),**data)
 fork=json.loads((d/'fork.json').read_text())['time_s'];metrics=json.loads((d/'condition_metrics.json').read_text());trace=z['lower_trace'].reshape(-1,n,11);lower_time=np.arange(len(trace))*.005
 for lane in range(n):
  valid=data['alive'][:,lane].astype(bool)&(data['time_s']>=fork);ix=np.flatnonzero(valid);lo=(trace[:,lane,6]>0)&(lower_time>=fork);entry={'case':job['name'],'lane':lane,**metrics[lane],'post_fork_recorded_s':float(valid.sum()*dt)}
  for wheel in ('frontwheel','rearwheel'):
   entry[wheel+'_normal_impulse_Ns']=float(data[wheel+'_normal_N'][valid,lane].sum()*dt)
   entry[wheel+'_normal_vector_impulse_Ns']=(data[wheel+'_normal_vector_N'][valid,lane].sum(0)*dt).tolist()
  entry['steer_contact_abs_impulse_Nms']=float(np.abs(data['steer_contact_Nm'][valid,lane]).sum()*dt)
  entry['steer_contact_signed_impulse_Nms']=float(data['steer_contact_Nm'][valid,lane].sum()*dt)
  entry['motor_clip_fraction']=data['motor_clipped'][valid,lane].mean(0).tolist() if len(ix) else None
  entry['post_fork_path_rmse_m']=float(np.sqrt(np.mean(trace[lo,lane,3]**2)))
  entry['post_fork_speed_rmse_m_s']=float(np.sqrt(np.mean((trace[lo,lane,2]-metrics[lane]['requested_speed_m_s'])**2)))
  occupied=np.flatnonzero(valid&(data['frontwheel_raised_contact'][:,lane]|data['rearwheel_raised_contact'][:,lane]));groups=np.split(occupied,np.flatnonzero(np.diff(occupied)*dt>.05)+1) if len(occupied) else []
  for event_id,g in enumerate(groups):
   start=float(data['time_s'][g[0]]);end=float(data['time_s'][g[-1]]);next_start=float(data['time_s'][groups[event_id+1][0]]) if event_id+1<len(groups) else float(data['time_s'][ix[-1]])
   candidate=np.flatnonzero(lo&(lower_time>=end)&(lower_time<next_start));recovery=None
   for k in candidate:
    if k+40<=len(trace) and lower_time[k+39]<next_start and np.all(trace[k:k+40,lane,6]>0) and np.all(np.abs(trace[k:k+40,lane,3])<=.3):recovery=float(lower_time[k]-end);break
   evalid=valid&(data['time_s']>=start)&(data['time_s']<=end)
   events.append(dict(case=job['name'],lane=lane,event=event_id,start_s=start,end_s=end,next_event_or_observation_end_s=next_start,path_band_settle_delay_s=recovery,steer_contact_abs_impulse_Nms=float(np.abs(data['steer_contact_Nm'][evalid,lane]).sum()*dt)))
  rows.append(entry)
 (out/'status.json').write_text(json.dumps(dict(stage='running',completed=jobs.index(job)+1,planned=len(jobs))))
 print(job['name'],flush=True)
(out/'per_lane.json').write_text(json.dumps(rows,indent=2));(out/'contact_events.json').write_text(json.dumps(events,indent=2))
(out/'README.md').write_text('''# 接触与控制量复算

[原始实际/参考XY总览](../RESULTS.md)

per_lane.json覆盖全部分支，分别保留原任务指标、分叉后路径/速度RMSE、前后轮法向接触冲量、转向轴接触力矩的有符号/绝对冲量、电机请求力矩超过限幅的时间比例。每组NPZ保留0.2ms时序，只有真实存活步参与统计。

contact_events.json的事件操作定义：有效接触点世界z>2mm且法向力>1N；前后轮合并，相隔不超过50ms的接触合并为一段。地面基准z=0。按“凸起接触段”解释，不把它叫独立石子或净扰动。路径入带延迟要求事件结束后连续200ms |横向误差|≤0.3m，限于下一个事件/真实终点之前；null表示该窗口内未观察到保持，0表示已在带内，不算新恢复。接触窗口始于原记录0.8s。

冲量是承重在内的总接触量，未减去匹配平地反事实；不是等效侧倾扰动上界。转向轴力矩积分也不是侧倾角动量。不同失败长度分别报告观察时间，不按全程累计量直接排序。副本与仅转向的配对使用相同后缀起点，仍应先检查恢复快照对照。
''')
(out/'status.json').write_text(json.dumps(dict(stage='complete',completed=len(jobs),planned=len(jobs))))
