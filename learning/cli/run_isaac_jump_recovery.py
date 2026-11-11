"""Bounded Isaac RecoveryBridge mechanism port, using the existing Go2 runner.

Fixed pulse proposals -> short diffusion repair -> witnessed full continuation
-> PPO + successful-action supervision + source retention -> fixed DEV panel.
This first port does not claim the original learned-explorer algorithm.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--stage',choices=['pipeline','collect','repair','train','evaluate'],default='pipeline')
a=p.parse_args();cfg=json.loads(a.config.read_text());out=a.output.resolve()
if a.stage=='pipeline':
 out.mkdir(parents=True,exist_ok=False)
 def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,updated_unix=time.time(),**kw),indent=2))
 def notify(message,name):
  subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章 RB 跳跃适配','--message',message,'--receipt',str(out/(name+'.json'))],check=False)
 status('starting');notify('定时任务已取得GPU使用权，开始批量示教采集。','start_bark')
 for stage in ('collect','repair','train','evaluate'):
  status(stage)
  with (out/(stage+'.log')).open('w') as stream:
   code=subprocess.run([sys.executable,__file__,'--config',str(a.config.resolve()),'--output',str(out),'--stage',stage],stdout=stream,stderr=subprocess.STDOUT).returncode
  if code:
   status('failed',failed_stage=stage,exit_code=code);notify(f'{stage}阶段失败，已停止后续步骤；请查看日志。','error_bark');raise SystemExit(code)
  result=json.loads((out/stage/'status.json').read_text())
  if result['stage']=='no_training_evidence':
   status('stopped_no_training_evidence',detail=result);notify('未获得足够的真实成功示教，已按预算停止，未以普通PPO冒充RB训练。','completion_bark');raise SystemExit(0)
  if stage=='train':notify('跳跃学生训练阶段结束，开始并行固定开发集选模。','training_bark')
 status('completed');notify('示教、训练和固定开发集选模结束，结果已保存。','completion_bark');raise SystemExit(0)

stage_dir=out/a.stage;stage_dir.mkdir(parents=True,exist_ok=False)
def stage_exception(kind,error,traceback):
 (stage_dir/'status.json').write_text(json.dumps(dict(stage='failed',error_type=kind.__name__,error=str(error)),indent=2))
 sys.__excepthook__(kind,error,traceback);sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.excepthook=stage_exception

if a.stage in ('collect','repair','evaluate'):
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True,device='cuda:0').app
 import numpy as np
 import torch
 from sttw_control.phase_u_env import PhaseUEnv
 from sttw_control.phase_u_actor_critic import PhaseUActorCritic
 from sttw_control.ditch_ground_handoff import VectorDitchHandoff
 from sttw_control.jump_recovery_learning import JumpSequenceTeacher,fit_teacher,teaching_arrays
 torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
 roots=cfg['training_roots'] if a.stage!='evaluate' else cfg['development_roots']
 source_count=cfg['source_repeats'];reviews=cfg['teacher_reviews']
 candidates=1 if a.stage=='collect' else cfg['teacher_candidates']*reviews+source_count
 models=[]
 if a.stage=='evaluate':
  models=[('source',None)]+[(f'update{u}',out/'train/training'/f'model_update_{u:05d}.pt') for u in cfg['evaluated_updates'] if (out/'train/training'/f'model_update_{u:05d}.pt').exists()]
  trained=json.loads((out/'train/status.json').read_text())
  final=trained['completed_updates']
  final_path=out/'train/training'/f'model_{final}.pt'
  if final not in cfg['evaluated_updates'] and final_path.exists():models.append((f'final{final}',final_path))
  if cfg.get('evaluation_models'):
   models=[('source',None)]+[(m['name'],Path(m['checkpoint'])) for m in cfg['evaluation_models']]
  candidates=len(models)
 n=len(roots)*candidates
 env_cfg=json.loads(Path(cfg['nominal_environment']).read_text());env_cfg.update(num_envs=n,record_verified_diagnostics=False)
 env=PhaseUEnv(env_cfg,stage_dir);task=VectorDitchHandoff(env,env_cfg['ground_handoff'],stage_dir)
 source=PhaseUActorCritic(env.num_obs,106,4,env_cfg['bundle']).to(env.device).eval()
 policy_models=[source]
 for name,path in models[1:]:
  model=PhaseUActorCritic(env.num_obs,106,4,env_cfg['bundle']).to(env.device).eval()
  model.load_state_dict(torch.load(path,map_location=env.device,weights_only=False)['model_state_dict']);policy_models.append(model)
 teacher=None
 if a.stage=='repair':
  with np.load(out/'collect/teaching.npz') as data:
   if not len(data['sequence_observation']):
    (stage_dir/'status.json').write_text(json.dumps(dict(stage='no_training_evidence',reason='No successful source H16 jump sequences')));os._exit(0)
   teacher=JumpSequenceTeacher(source.mean,source.std).to(env.device)
   history=fit_teacher(teacher,torch.as_tensor(data['sequence_observation'],device=env.device),torch.as_tensor(data['sequence_action'],device=env.device),cfg['teacher_updates'])
  torch.save(teacher.state_dict(),stage_dir/'teacher.pt');(stage_dir/'teacher_fit.json').write_text(json.dumps(history,indent=2))
 obs,_=task.reset()
 root_ids=torch.arange(len(roots),device=env.device).repeat_interleave(candidates)
 candidate_ids=torch.arange(candidates,device=env.device).repeat(len(roots))
 starts=torch.tensor([r['pulse_start_step'] for r in roots],device=env.device)[root_ids]
 pulse=torch.tensor([r['normalized_action_delta'] for r in roots],device=env.device)[root_ids]
 duration=cfg['pulse_steps'];repair_start=starts+duration
 sequences=torch.zeros(n,16,4,device=env.device);queried=torch.zeros(n,device=env.device,dtype=torch.bool)
 rows={k:[] for k in ('root_xyzw','observation','action','valid','jump_active','phase','stage','reward','end_code')}
 started=time.monotonic()
 with torch.no_grad():
  for step in range(env.max_episode_length):
   valid=~task.finished
   latent=source.act_inference(obs)
   if a.stage=='evaluate':
    for i,policy in enumerate(policy_models[1:],1):
     mask=candidate_ids==i;latent[mask]=policy.act_inference(obs[mask])
   action=torch.tanh(latent)
   pulsing=(step>=starts)&(step<repair_start)&(task.phase==1)
   action=(action+torch.where(pulsing[:,None],pulse,0.)).clamp(-.999999,.999999)
   if teacher is not None:
    begin=(step==repair_start)&(candidate_ids>=source_count)&((candidate_ids-source_count)%reviews==0)&valid&(task.phase==1)
    ids=begin.nonzero().flatten()
    if len(ids):
     generated=teacher.sample(obs[ids])
     for repeat in range(reviews):sequences[ids+repeat]=generated;queried[ids+repeat]=True
    repair=(step>=repair_start)&(step<repair_start+16)&queried&valid&(task.phase==1)
    tick=(step-repair_start).clamp(0,15)
    generated=sequences[torch.arange(n,device=env.device),tick]
    action=torch.where(repair[:,None],generated,action)
   rows['observation'].append(obs.clone());rows['action'].append(action.clone());rows['valid'].append(valid.clone())
   obs,_,_,done,info=task.step(torch.atanh(action))
   for k,v in dict(root_xyzw=env.physical()[0],jump_active=info['jump_active'],phase=task.phase.clone(),stage=info['jump_stage'],reward=info['unscaled_reward'],end_code=info['end_code']).items():rows[k].append(v.clone())
   if step%50==0:(stage_dir/'status.json').write_text(json.dumps(dict(stage='rollout',step=step,active=int((~task.finished).sum()),elapsed_s=time.monotonic()-started)))
   if bool(task.finished.all()):break
 trace={k:torch.stack(v).cpu().numpy() for k,v in rows.items()};trace['time']=(np.arange(len(rows['valid']))+1)*env.dt
 success=task.reached_goal.cpu().numpy();np.savez_compressed(stage_dir/'trajectories.npz',**trace)
 (stage_dir/'environment.json').write_text(json.dumps(env_cfg,indent=2))
 result=dict(stage='completed',success=success.tolist(),root_ids=root_ids.cpu().tolist(),candidate_ids=candidate_ids.cpu().tolist(),elapsed_s=time.monotonic()-started,physics_lane_steps=env.total_physics_steps,roots=roots)
 if a.stage=='collect':
  data=teaching_arrays(trace,success)
  # Retention targets are the frozen source, not the externally added pulse.
  data['executed_action']=data['action'].copy()
  if len(data['observation']):
   with torch.no_grad():data['action']=torch.tanh(source.act_inference(torch.as_tensor(data['observation'],device=env.device))).cpu().numpy()
  np.savez_compressed(stage_dir/'teaching.npz',**data)
  if not success.any():result.update(stage='no_training_evidence',reason='Source produced no witnessed goal-reaching continuation')
 elif a.stage=='repair':
  matrix=success.reshape(len(roots),candidates);repaired=np.zeros_like(matrix);accepted=[]
  for i in range(len(roots)):
   witnessed=queried.cpu().numpy().reshape(len(roots),candidates)[i,source_count:]
   winners=np.flatnonzero((matrix[i,source_count:]&witnessed).reshape(-1,reviews).all(1))
   if not matrix[i,:source_count].any() and len(winners):
    chosen=source_count+int(winners[0])*reviews;repaired[i,chosen]=True
    accepted.append(dict(root=i,candidate=chosen,source_failures=source_count,teacher_successful_reviews=reviews))
  data=teaching_arrays(trace,repaired.ravel());np.savez_compressed(stage_dir/'teaching.npz',**data)
  result.update(accepted=accepted,scope='Paired replicated-scene source failure and diffusion repair success; fixed proposals, not learned exploration; no snapshot restoration')
  if not accepted:result.update(stage='no_training_evidence',reason='No source-failure/teacher-success pair; student training not launched')
 else:
  matrix=success.reshape(len(roots),candidates);baseline=matrix[:,0];scores=[]
  for j,(name,path) in enumerate(models):
   scores.append(dict(name=name,checkpoint=str(path) if path else None,completed=int(matrix[:,j].sum()),gained=int((matrix[:,j]&~baseline).sum()),lost=int((~matrix[:,j]&baseline).sum()),labels=matrix[:,j].tolist()))
  selection_rule=cfg.get('selection_rule','retain_source')
  eligible=scores if selection_rule=='completion' else [r for r in scores if r['lost']==0]
  best=max(eligible,key=lambda r:r['completed'])
  result.update(models=scores,selected=best,source_positive_count=int(baseline.sum()),retention_vacuous=not bool(baseline.any()),selection='Among evaluated checkpoints retain all source-positive fixed DEV roots, then maximize completed goals; ties keep earlier/source. DEV selection only, not independent TEST.')
  if selection_rule=='completion':result['selection']='Maximize fixed DEV completed goals; report gained/lost without a zero-loss veto; ties keep earlier/source. DEV selection only.'
  (out/'selected_model.json').write_text(json.dumps(result,indent=2))
 (stage_dir/'status.json').write_text(json.dumps(result,indent=2));os.sync();os._exit(0)

if a.stage=='train':
 training=json.loads(Path(cfg['training_environment']).read_text())
 training['recovery_teaching']=dict(demo=str(out/'repair/teaching.npz'),keep=str(out/'collect/teaching.npz'),**cfg['student_loss'])
 path=out/'training_config.json';path.write_text(json.dumps(training,indent=2))
 # train_isaac_phase_u creates its output and owns the single resident Isaac app.
 stage_dir.rmdir()
 os.execv(sys.executable,[sys.executable,'learning/cli/train_isaac_phase_u.py','--config',str(path),'--output',str(stage_dir)])
