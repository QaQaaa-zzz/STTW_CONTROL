"""Freeze an interpolated alpha operating-response grid with a held physical group."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from analyze_roll_contacts import roll_state
from analyze_roll_braking_boundary import margin

def collect(run,family):
 jobs=json.loads((run/'jobs.json').read_text());summary=json.loads((run/'analysis_summary.json').read_text())['rows'];rows=[]
 for job in jobs:
  folder=run/job['name'];contract=json.loads((folder/'contract.json').read_text());cfg=json.loads((Path(contract['bundle'])/'config.json').read_text())['follower'];route=np.load(Path(contract['bundle'])/'route.npz')
  terrain=json.loads((Path(job['terrain'])/'identity.json').read_text());height=terrain['height_m'];group=job['physical_seed']
  with np.load(folder/'trajectory.npz') as z:
   tr=z['lower_trace'].reshape(-1,len(contract['settings']['alphas']),11)
   for lane,alpha in enumerate(contract['settings']['alphas']):
    if contract['settings']['zero_middle'][lane]:continue
    label=contract['settings']['case_labels'][lane]
    if 'duplicate' in label:continue
    valid=tr[:,lane,6].astype(bool);m=next(x for x in summary if x['case']==job['name'] and x['label']==label)
    ch=contract['bodies'].index('chasis');phi,w=roll_state(z['body_state'][valid,lane,ch]);H=margin(np.degrees(phi),np.degrees(w),np.degrees(.7),6146.053403416946)[2]
    speed=tr[valid,lane,2];progress=z['progress'][valid,lane];pose=z['pose'][valid,lane]
    look=np.clip(cfg['lookahead_time_s']*np.maximum(speed,cfg['lookahead_min_speed_m_s']),cfg['lookahead_min_m'],cfg['lookahead_max_m'])
    point=np.stack([np.interp(progress+look,route['s'],route['xy'][:,k]) for k in range(2)],-1);d=point-pose[:,:2];dom=np.cos(pose[:,2])*d[:,0]+np.sin(pose[:,2])*d[:,1]-cfg['preview_body_x_min_m']
    before=np.flatnonzero(progress<2.5);entry=before[-1] if len(before) else 0
    rows.append(dict(family=family,height_m=height,speed_m_s=float(contract['settings']['user_speed_m_s']),alpha=float(alpha),physical_group=group,case=job['name'],source=str(folder),min_H_deg=float(H.min()),min_domain_m=float(dom.min()),path_peak_m=m['path_peak_m'],speed_rmse_m_s=m['speed_rmse_m_s'],full_window=float(not m['failed']),final_hold=float(m['final_hold']),task_A=float(m['tasks']['A']),task_B=float(m['tasks']['B']),task_C=float(m['tasks']['C']),pre_speed_m_s=float(speed[entry]),pre_phi_deg=float(np.degrees(phi[entry])),pre_omega_deg_s=float(np.degrees(w[entry])),cause=m['cause']))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--ridge',type=Path,required=True);p.add_argument('--gravel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=collect(a.ridge,'ridge')+collect(a.gravel,'gravel');models={};checks=[]
 targets=['min_H_deg','min_domain_m','path_peak_m','speed_rmse_m_s','full_window','final_hold','task_A','task_B','task_C','pre_speed_m_s','pre_phi_deg','pre_omega_deg_s']
 for family in ['ridge','gravel']:
  sub=[r for r in rows if r['family']==family];axes=[sorted(set(r[k] for r in sub)) for k in ['height_m','speed_m_s','alpha']];values=np.empty(tuple(map(len,axes))+(len(targets),))
  for ih,h in enumerate(axes[0]):
   for iv,v in enumerate(axes[1]):
    for ia,alpha in enumerate(axes[2]):
     train=[r for r in sub if r['height_m']==h and r['speed_m_s']==v and r['alpha']==alpha and r['physical_group'] in [101,102]]
     assert len(train)==2,(family,h,v,alpha,len(train))
     values[ih,iv,ia]=np.mean([[r[k] for k in targets] for r in train],axis=0)
  model=RegularGridInterpolator(axes,values,bounds_error=True)
  for r in sub:
   if r['physical_group']==103:
    pred=model([[r[k] for k in ['height_m','speed_m_s','alpha']]])[0]
    checks.append(dict(family=family,case=r['case'],alpha=r['alpha'],actual={k:r[k] for k in targets},predicted=dict(zip(targets,map(float,pred)))))
  models[family]=dict(axes=dict(zip(['height_m','speed_m_s','alpha'],axes)),targets=targets,values=values.tolist())
 metrics={}
 for family in models:
  subset=[r for r in checks if r['family']==family];metrics[family]=dict(n=len(subset),scores={k:float(np.mean([(r['predicted'][k]-r['actual'][k])**2 for r in subset])) if k in ['full_window','final_hold','task_A','task_B','task_C'] else float(np.mean([abs(r['predicted'][k]-r['actual'][k]) for r in subset])) for k in targets})
 artifact=dict(role='Frozen DEV-calibrated trilinear response predictor; physical groups101/102 only;103 assessment',precontact_progress_m=2.5,inputs=['terrain_family','height_m','requested_speed_m_s','alpha'],targets=targets,models=models,fit_physical_groups=[101,102],heldout_physical_group=103,extrapolation='not allowed',selection=dict(task_thresholds={'A':[.3,.2],'B':[.6,.1],'C':[.4,.15]},margin_scales={'roll_deg':float(np.degrees(.7)),'domain_m':1.0},rule='Among predicted final_hold>=0.5 and task-error-feasible candidates, maximize min(H/roll_scale,domain/1m); otherwise maximize predicted task probability; then minimize squared normalized task errors; final tie smaller alpha',baseline='fixed alpha0/.25/.5/.75/1 and archived simple rule'))
 (a.output/'frozen_response_model.json').write_text(json.dumps(artifact,indent=2))
 (a.output/'heldout_validation.json').write_text(json.dumps(dict(metrics=metrics,rows=checks),indent=2));(a.output/'operating_points.json').write_text(json.dumps(rows,indent=2))
 lines=['# 冻结运行响应预测','','[单脊原始XY与全部结果]('+str(a.ridge/'RESULTS.md')+') · [石子原始XY与全部结果]('+str(a.gravel/'RESULTS.md')+')','','输入仅为地形类型、高度、请求速度和候选α。使用物理组101/102拟合规则网格三线性插值，组103只评估，不调参数。输出包括进入接触区前的运行状态、最小名义H、最小前视余量、原路径/速度误差及各任务概率。该模型把实测α响应连接到余量选档，是数据标定预测，不是从名义方程直接推导完整车辆成功。','','|地形|留出数|满窗Brier|保持Brier|任务C Brier|H MAE °|前视余量MAE m|','|---|---:|---:|---:|---:|---:|---:|']
 for f,m in metrics.items():
  s=m['scores'];lines.append(f'|{f}|{m["n"]}|{s["full_window"]:.4f}|{s["final_hold"]:.4f}|{s["task_C"]:.4f}|{s["min_H_deg"]:.3f}|{s["min_domain_m"]:.3f}|')
 lines+=['','模型在细alpha扫描运行前冻结，所有中间档预测只用相邻已测档插值。S3比较预测误差、概率可靠性与选档结果，不在测试数据上重新拟合。完整采样分母、逐条留出值、模型数组都保留。S1的完整代价检验显示共享策略不是每档精确最优解，因此不采用训练权重切线作为定量选档器，改用此实测响应映射。']
 (a.output/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
