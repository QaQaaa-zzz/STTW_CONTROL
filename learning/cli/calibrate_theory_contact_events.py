"""Fit measured wheel-contact impulse summaries on groups101/102, assess103."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_roll_contacts import roll_state
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
z=np.load(r/'contact_analysis/contact_bins.npz');contact=z['mean'];fields=list(z['fields']);counts=z['valid_physics_steps'];rows=[]
for job in json.loads((r/'jobs.json').read_text()):
 folder=r/job['name'];c=json.loads((folder/'contract.json').read_text());height=json.loads((Path(job['terrain'])/'identity.json').read_text())['height_m'];start=c['shared_physics_contacts']['lane_start'];n=c['shared_physics_contacts']['lane_count']
 with np.load(folder/'trajectory.npz') as data:
  trace=data['lower_trace'].reshape(-1,n,11)
  for lane,label in enumerate(c['settings']['case_labels']):
   if label=='Zero':continue
   valid=trace[:,lane,6].astype(bool);phi,w=roll_state(data['body_state'][:,lane,c['bodies'].index('chasis')]);con=contact[:,start+lane];events={}
   for wheel in ['front','rear']:
    ids=np.flatnonzero(valid&(counts[:,start+lane]>0)&(con[:,fields.index(wheel+'_raised')]>0))
    if not len(ids):continue
    first=int(ids[0]);end=min(first+50,len(valid));take=np.arange(first,end);take=take[valid[take]&(counts[take,start+lane]>0)];moment=con[:,fields.index(wheel+'_normal_roll_Nm')]+con[:,fields.index(wheel+'_friction_roll_Nm')]
    events[wheel]=dict(time_s=(first+1)*.005,speed_m_s=float(trace[first,lane,2]),phi_deg=float(np.degrees(phi[first])),omega_deg_s=float(np.degrees(w[first])),signed_impulse_Nms=float(np.sum(moment[take])*.005),absolute_impulse_Nms=float(np.sum(abs(moment[take]))*.005),observed_s=len(take)*.005)
   rows.append(dict(case=job['name'],height_m=height,requested_speed_m_s=c['settings']['user_speed_m_s'],alpha=c['settings']['alphas'][lane],physical_group=job['physical_seed'],events=events))
# Fixed linear feature map; no outcome variables, model order or tuning on group103.
fits={};predictions=[]
for wheel in ['front','rear']:
 subset=[x for x in rows if wheel in x['events'] and x['events'][wheel]['observed_s']>=.25]
 def features(row):
  h=row['height_m'];e=row['events'][wheel];v=e['speed_m_s'];alpha=row['alpha'];return [1,h,v*v,h*v*v,alpha,alpha*alpha]
 train=[x for x in subset if x['physical_group'] in [101,102]];test=[x for x in subset if x['physical_group']==103]
 X=np.array([features(x) for x in train]);Y=np.array([[x['events'][wheel][k] for k in ['signed_impulse_Nms','absolute_impulse_Nms']] for x in train]);coef=np.linalg.lstsq(X,Y,rcond=None)[0]
 pred=np.array([features(x) for x in test])@coef;truth=np.array([[x['events'][wheel][k] for k in ['signed_impulse_Nms','absolute_impulse_Nms']] for x in test]);base=np.repeat(Y.mean(0)[None],len(test),axis=0)
 fits[wheel]=dict(features=['1','h','v_actual^2','h*v_actual^2','alpha','alpha^2'],coefficients=coef.tolist(),n_fit=len(train),n_test=len(test),MAE_Nms=np.mean(abs(pred-truth),axis=0).tolist(),training_mean_baseline_MAE_Nms=np.mean(abs(base-truth),axis=0).tolist())
 predictions.extend(dict(case=x['case'],alpha=x['alpha'],wheel=wheel,actual=truth[i].tolist(),predicted=pred[i].tolist()) for i,x in enumerate(test))
out=r.parent/'S2';(out/'contact_event_calibration.json').write_text(json.dumps(dict(definition='250ms from first raised contact for each wheel; projected total contact roll moment about vehicle COM, not Euler roll-rate impulse; truncated windows excluded only from fitting and retained in rows',fit_groups=[101,102],heldout_group=103,fits=fits,events=rows,predictions=predictions),indent=2))
lines=['# 接触事件标定','','每个轮首次高地接触后250ms，比较有符号力矩积分和绝对力矩积分。包含全部五档、4个高度、4个速度；101/102拟合，103留出。窗口不足250ms的失败记录仍保留在events，单独标明观察长度。模型使用实际接触速度，因此它解释接触机制，不直接作为入障前选档输入。','','|轮|拟合/留出|有符号MAE Nm·s|绝对量MAE Nm·s|常数基线的两项MAE|','|---|---:|---:|---:|---|']
for wheel,f in fits.items():lines.append(f'|{wheel}|{f["n_fit"]}/{f["n_test"]}|{f["MAE_Nms"][0]:.5f}|{f["MAE_Nms"][1]:.5f}|{f["training_mean_baseline_MAE_Nms"]}|')
lines+=['','这里的接触力矩包含支撑与摩擦。其积分单位为Nm·s，不能直接代入以rad/s定义的J_sat；完整车辆惯量与约束接触投影仍需要动力学映射。S3采用已标定的运行响应/余量网格预测，名义冲量方程用于解释两轮作用间隔及转向能力，不将其当作已验证的成功概率公式。']
(out/'CONTACT_CALIBRATION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(fits,indent=2))
