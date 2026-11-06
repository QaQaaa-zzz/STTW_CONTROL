import json,os,subprocess,time
from pathlib import Path
folder=Path(__file__).resolve().parent
base=folder.parent
os.chdir('/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad')
os.environ['PYTHONPATH']='learning/src'
def record(title,body):
 subprocess.run(['/usr/bin/python3','/tmp/chapter3_ledger.py',title],input=body,text=True,check=True)
for stage,command in json.loads((folder/'resume_commands.json').read_text()):
 if stage=='prepare_online':
  inventory=json.loads((base/'chapter3_midpoint_representative/completed_records/inventory.json').read_text())
  if inventory['evaluated_cases']!=216:raise RuntimeError('Incomplete midpoint evidence')
 (folder/'resume_status.json').write_text(json.dumps(dict(status='running',stage=stage,started_unix=time.time(),pid=os.getpid(),command=command),indent=2))
 record('性能优化后接续阶段 '+stage,str(command)+'; 原预算、冻结模型与频率不变，复用已完成结果。')
 with (folder/('resume_'+stage+'.log')).open('w') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  (folder/'resume_status.json').write_text(json.dumps(dict(status='failed',stage=stage,returncode=result.returncode)))
  record('接续阶段失败 '+stage,str(result.returncode));raise SystemExit(result.returncode)
 record('接续阶段退出 '+stage,'入口正常结束；结果待统一分析，不代表任务书完成。')
(folder/'resume_status.json').write_text(json.dumps(dict(status='completed_analysis_pending',finished_unix=time.time())))
