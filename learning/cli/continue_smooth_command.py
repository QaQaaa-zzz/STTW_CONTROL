#!/usr/bin/env python3
"""User-authorized serial handoff to250; no wall deadline and no automatic retry."""
import os,sys,json,time,argparse,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.direct_command_training import write,notify
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tb-port',type=int,default=6011);a=p.parse_args()
parent=a.parent.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
if (root/'queue_status.json').exists():raise FileExistsError('queue already exists; no automatic restart')
launch=json.loads((parent/'launch.json').read_text());pid=launch['pid']
status=dict(state='waiting_for_parent',parent_run=str(parent),parent_pid=pid,declared_total_updates=250,wall_limits_enabled=False,started_epoch=time.time(),pid=os.getpid())
write(root/'queue_status.json',status)
try:
    while True:
        proc=Path(f'/proc/{pid}/cmdline');cmd=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
        if not cmd:break
        if 'train_smooth_command.py' not in cmd or str(parent) not in cmd:raise RuntimeError('parent PID identity changed')
        d=json.loads((parent/'status.json').read_text());status.update(parent_completed=d['completed_updates'],parent_stage=d['stage'],checked_epoch=time.time());write(root/'queue_status.json',status);time.sleep(15)
    d=json.loads((parent/'status.json').read_text())
    if d['state'] not in ('complete','partial','budget_stopped'):raise RuntimeError('parent did not finish safely: '+str(d))
    for alpha in [0,1]:
        last=json.loads((parent/f'alpha{alpha}/last_completed.json').read_text())
        if not 60<=last['update']<=150 or not Path(last['checkpoint']).exists():raise RuntimeError('missing safe parent checkpoint')
        dst=root/f'alpha{alpha}/checkpoints';dst.mkdir(parents=True,exist_ok=True)
        (dst/'update_0060.pt').symlink_to(parent/f'alpha{alpha}/checkpoints/update_0060.pt')
    (root/'evaluation60').symlink_to(parent/'evaluation60',target_is_directory=True)
    write(root/'continuation_contract.json',dict(parent_run=str(parent),target_updates=250,initialization='full_upper_learner_resume_environment_reset',restore=['Actor','Critic','Adam','log_std','accepted_updates','RNG'],physical_state_restored=False,episode_rule='first unused episode key after parent case manifest',wall_limits_enabled=False,final_evaluation=250,reuse_evaluation=60,best_selection='physical fixed-six candidates60/250; training continues from last, evaluation uses best'))
    env=os.environ.copy();env['XLA_PYTHON_CLIENT_PREALLOCATE']='false';env['OMP_NUM_THREADS']='2';env['MUJOCO_GL']='egl'
    command=[sys.executable,'-u',str(Path(__file__).with_name('train_smooth_command.py')),'--output',str(root),'--fresh','--resume-parent',str(parent),'--target-updates','250','--unlimited-wall','--train-wall','3600','--total-wall','9000']
    with (root/'execution.log').open('w') as log:worker=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'launch.json',dict(pid=worker.pid,started_epoch=time.time(),command=command))
    status.update(state='running',worker_pid=worker.pid);write(root/'queue_status.json',status)
    with (root/'tensorboard.log').open('w') as log:tb=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),'--host','0.0.0.0','--port',str(a.tb_port),'--reload_interval','5'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'tensorboard_service.json',dict(pid=tb.pid,url=f'http://localhost:{a.tb_port}/#scalars',logdir=str(root/'tensorboard')))
    # Verify actual continued rewards while the worker runs, no assumption from server availability.
    import urllib.request
    verified=False
    while worker.poll() is None:
        if not verified:
            try:
                url=f'http://localhost:{a.tb_port}/data/plugin/scalars/scalars?run=alpha0&tag=train%2Fmean_step_reward'
                values=json.load(urllib.request.urlopen(url,timeout=3));values=[v for v in values if v[1]>150]
                if values:
                    write(root/'tensorboard_verified.json',dict(url=f'http://localhost:{a.tb_port}/#scalars',checked_epoch=time.time(),actual_reward_latest=values[-1]));verified=True;notify('STTW续训250已开始',f'TensorBoard http://localhost:{a.tb_port}/#scalars')
            except (OSError,ValueError):pass
        time.sleep(15)
    status.update(state='complete' if worker.returncode==0 else 'error',returncode=worker.returncode,ended_epoch=time.time(),reward_http_verified=verified);write(root/'queue_status.json',status)
except BaseException as exc:
    status.update(state='error',reason=str(exc),ended_epoch=time.time());write(root/'queue_status.json',status);notify('STTW续训衔接停止',str(exc));raise
