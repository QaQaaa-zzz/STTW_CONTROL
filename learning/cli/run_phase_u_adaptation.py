"""Run the existing trainer/exporters/evaluators for the declared 150-update stage."""
import argparse,os,select,subprocess,json,time,shutil
from pathlib import Path
parser=argparse.ArgumentParser(description='Bounded 150-update jump adaptation and matched existing-task evaluation')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--approach',type=Path,required=True)
parser.add_argument('--after-pid',type=int)
args=parser.parse_args();r=args.output.resolve();r.mkdir(parents=True,exist_ok=True)
pid=args.after_pid
try:fd=os.pidfd_open(pid) if pid is not None else None
except ProcessLookupError:fd=None
if fd is not None:
 (r/'pipeline_status.json').write_text(json.dumps(dict(stage='waiting_for_other_gpu_evaluation',pid=pid,started=time.time())))
 select.select([fd],[],[]);os.close(fd)
(r/'pipeline_status.json').write_text(json.dumps(dict(stage='training',started=time.time())))
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--message','第三章跳跃：开始从update2适配当前车型，150更新，4096环境，保留Go2随机化。','--receipt',str(r/'start_notification.json')])
with (r/'train.log').open('w') as log:
 result=subprocess.run(['bash','learning/scripts/gpu_queue.sh','/home/qy/miniconda3/envs/env_isaaclab/bin/python','learning/cli/train_isaac_phase_u.py','--config','learning/configs/isaac_rb_jump_current150.json','--output',str(r/'adaptation')],stdout=log,stderr=subprocess.STDOUT)
(r/'pipeline_status.json').write_text(json.dumps(dict(stage='training_complete' if result.returncode==0 else 'failed',exit_code=result.returncode)))
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--message','第三章跳跃150更新训练阶段已结束，待固定工况评测。可以继续推进任务。' if result.returncode==0 else '第三章跳跃训练失败，请检查日志。可以继续推进任务。','--receipt',str(r/'stage_notification.json')])
if result.returncode:raise SystemExit(result.returncode)
lab='/home/qy/miniconda3/envs/env_isaaclab/bin/python'
analysis='/home/qy/miniconda3/envs/mjp/bin/python'
bundles=r/'bundles';bundles.mkdir()
shutil.copytree('runs/rb_sttw_platform/evaluation_bundles/update2',bundles/'update2')
for name,checkpoint in [('best',r/'adaptation/training/model_step_best.pt'),('last150',r/'adaptation/training/model_150.pt')]:
 subprocess.run([lab,'learning/cli/export_trained_phase_u.py','--checkpoint',str(checkpoint),'--source-bundle','runs/rb_sttw_platform/training_bundle','--output',str(bundles/name)],check=True)
source=args.approach.resolve()
all_results=[]
for model in ['update2','best','last150']:
 for root_config in sorted(source.glob('rb_connection*.json')):
  cfg=json.loads(root_config.read_text());origin=root_config.stem;folder=r/'evaluation'/model/origin
  candidates=cfg['batch_candidates'][6:]
  cfg.update(output=str(folder),bundles=str(bundles),skills=[model]*6,batch_candidates=candidates,rear_speed_feedback_gain=[1.]*6,stop_at_ground_handoff=False,maximum_physics_lane_steps=52800,role='Matched six training origins x six connection candidates; current Kp1 ground; frozen jump comparison DEV, not holdout')
  path=r/(model+'_'+origin+'.json');path.write_text(json.dumps(cfg,indent=2))
  (r/'pipeline_status.json').write_text(json.dumps(dict(stage='evaluation',model=model,origin=origin,completed_cases=len(all_results),total_cases=108)))
  with (r/(model+'_'+origin+'.log')).open('w') as log:
   subprocess.run([lab,'learning/cli/evaluate_isaac_pendulum_jump.py','--config',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
  for c in json.loads((folder/'batch_results.json').read_text())['cases']:
   d=Path(c['output'])
   for tool in ['analyze_isaac_pendulum_landing.py','analyze_rb_continuous_task.py']:
    subprocess.run([analysis,'learning/cli/'+tool,'--run',str(d)],stdout=subprocess.DEVNULL,check=True)
   all_results.append(dict(model=model,origin=origin,candidate=c['candidate'],output=str(d)))
  (r/'evaluation_index.json').write_text(json.dumps(all_results,indent=2))
subprocess.run([analysis,'learning/cli/summarize_jump_adaptation.py','--run',str(r)],check=True)
(r/'pipeline_status.json').write_text(json.dumps(dict(stage='complete',completed_cases=len(all_results))))
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--message','第三章跳跃150轮及108条配对评测已完成，成功率和轨迹已整理。可以继续推进任务。','--receipt',str(r/'completion_notification.json')])
