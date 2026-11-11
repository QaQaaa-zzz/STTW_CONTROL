"""Freeze fine-alpha forecasts and finite matched evaluation manifests."""
import argparse,copy,itertools,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def select(predictions,task,thresholds):
 path_tol,speed_tol=thresholds[task]
 def score(row):
  feasible=row['final_hold']>=.5 and row['path_peak_m']<=path_tol and row['speed_rmse_m_s']<=speed_tol
  err=(row['path_peak_m']/path_tol)**2+(row['speed_rmse_m_s']/speed_tol)**2
  return (int(feasible),min(row['min_H_deg']/np.degrees(.7),row['min_domain_m']) if feasible else row['task_'+task],-err,-row['alpha'])
 return max(predictions,key=score)['alpha']

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);a=p.parse_args();r=a.root/'fine_scan';r.mkdir(exist_ok=False);cal=a.root/'calibration';model_path=a.root/'prediction/frozen_response_model.json';model=json.loads(model_path.read_text())
 alphas=[0,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,1];labels=[f'full_{x:g}' for x in alphas]+['Zero'];base=json.loads((cal/'single10_v2.6_seed101.json').read_text());old=[json.loads((cal/f'single10_v2.6_seed{s}.json').read_text())['randomization'] for s in [101,102,103]]
 # New joint parameter combinations drawn within the original three-group envelope.
 keys=['friction_range','mass_scale','servo_gain_scale','steer_zero_offset_rad'];bounds={k:[min(c[k][0] for c in old),max(c[k][0] for c in old)] for k in keys};groups={}
 for seed in [201,202,203,204]:
  rng=np.random.default_rng(seed);group=copy.deepcopy(old[0])
  for key in keys:
   value=float(rng.uniform(*bounds[key]));group[key]=[value,value]
  groups[seed]=group
 contexts=[('ridge',h,v,cal/'terrains'/f'single{round(h*100):02d}') for h,v in itertools.product([.04,.06,.08,.10],[1.8,2.1,2.3,2.6])]+[('gravel',h,v,a.runs_root/'h75_height_scan/terrains'/f'height{round(h*1000):02d}mm') for h,v in itertools.product([.005,.01,.02],[2.3,2.6])]
 jobs=[];forecasts=[];thresholds=model['selection']['task_thresholds']
 for family,h,v,terrain in contexts:
  assert (terrain/'terrain.npz').is_file(),terrain
  m=model['models'][family];interpolate=RegularGridInterpolator([m['axes'][k] for k in ['height_m','speed_m_s','alpha']],m['values'],bounds_error=True);values=interpolate([[h,v,alpha] for alpha in alphas]);pred=[dict(alpha=alpha,**dict(zip(m['targets'],map(float,row)))) for alpha,row in zip(alphas,values)]
  forecasts.append(dict(family=family,height_m=h,speed_m_s=v,predictions=pred,selected_alpha={task:select(pred,task,thresholds) for task in thresholds}))
  for seed,repeat in itertools.product(groups,[0,1]):
   name=f'{family}{round(h*1000):03d}_v{v:g}_seed{seed}_rep{repeat}';c=copy.deepcopy(base);c.pop('physics_contact_recording',None)
   c.update(num_envs=len(labels),trace_lanes=len(labels),alphas=alphas+[0],zero_middle=[False]*len(alphas)+[True],case_labels=labels,zero_reference_channels=[[False,False]]*len(labels),user_speed_m_s=v,seed=seed,randomization=groups[seed],omit_diagnostic_fields=['observation','policy_measurement'],evaluation_role='S3 frozen response prediction TEST; new parameter combinations within DEV envelope; fixed terrain layouts',role='Fine-alpha test, frozen H75/Local300, common original full-path start')
   path=r/(name+'.json');path.write_text(json.dumps(c,indent=2));jobs.append(dict(name=name,config=str(path),terrain=str(terrain),physical_seed=seed,repeat=repeat,speed_m_s=v,height_m=h,family=family))
 spec=json.loads((cal/'protocol.json').read_text());spec.update(experiment_id='H75_THEORY_FINE_ALPHA',title='S3：冻结预测与细alpha配对检验',case_labels=labels,terrains=sorted(set(j['name'].split('_v')[0] for j in jobs)),fixed_alphas=alphas,scenes=len(jobs),lanes=len(jobs)*len(labels),max_physics_lane_steps=len(jobs)*len(labels)*50000,batch_cases=len(jobs),max_batch_envs=672,retain_physics_contacts=False,data_role='Frozen prediction TEST: unseen physical combinations; seen terrain layouts; two technical repeats per group',physical_parameter_envelope=bounds,physical_groups=groups,repeat_unit='physical group; two repeats are nested technical repetitions',prediction_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),code_revision_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),stop_rule='176 contexts/2464 physical episodes or engineering error; no training, no outcome replacement, no automatic expansion',retention='200Hz states, controller terms, rewards, XY; omit network observation copies and all5kHz contact archives',estimated_wall_time_minutes=[35,90],prediction_assessment={'continuous':'MAE of H, preview margin, errors; all recorded failures retained','probability':'Brier and calibration by terrain/height/alpha; full-window, hold and task separately','selection':'paired selected versus each fixed alpha; pool repeats within physical group; report all contexts, no retuning','comparison':'forecast file frozen before first rollout'})
 for name,data in [('jobs.json',jobs),('protocol.json',spec),('status.json',dict(stage='prepared',planned=len(jobs),episodes=spec['lanes'])),('frozen_predictions.json',dict(model_sha256=spec['prediction_sha256'],contexts=forecasts))]: (r/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 (r/'README.md').write_text('# S3细alpha预测检验\n\n13档×22工况×4物理组×2重复=2288策略条，加176条Zero，共2464条。固定原H75/Local300，共同原始起点，10秒/条，0训练。\n\n预测见frozen_predictions.json；模型来自101/102，103只评估。新物理组201–204在原三组参数范围内联合抽样，其余噪声/延迟/质量结构不变。两次重复不计为独立物理组。672环境上限分四批；保留200Hz必需时序，省略观测副本和5kHz接触。\n\n运行后先看[全部XY](RESULTS.md)，再看[统计](ANALYSIS.md)。\n')
 q=a.root/'fine_scan_queue';q.mkdir(exist_ok=False);(q/'plan.json').write_text(json.dumps(dict(stages=[dict(name='h75_theory_fine_alpha',path=str(r),contexts=len(jobs),episodes=spec['lanes'])],total_episodes=spec['lanes'],optimizer_updates=0),indent=2));(q/'README.md').write_text('# S3有限队列\n\n[运行与原始XY](../fine_scan/RESULTS.md) · [汇总](../fine_scan/ANALYSIS.md) · [冻结预测](../fine_scan/frozen_predictions.json)\n\n4批，上限672环境，2464条。阶段完成/异常发送Bark。\n')
 print(json.dumps(dict(contexts=len(jobs),episodes=spec['lanes'],forecast_contexts=len(forecasts),groups=groups),indent=2))
if __name__=='__main__':main()
