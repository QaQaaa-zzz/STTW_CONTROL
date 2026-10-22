"""Wait for the existing campaign, evaluate once, and render results; no restart."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_best import atomic_json
from sttw_control.tolerance_comparison import resolve_comparison_actors


def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 campaign=Path(a.campaign).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'status.json').exists():raise FileExistsError('postprocessing attempt already exists')
 status=dict(state='running',phase='running',pid=os.getpid(),campaign=str(campaign),evaluation=str(out/'evaluation'))
 def update(**kw):
  status.update(kw);status['checked_unix']=time.time();atomic_json(out/'status.json',status)
 def child(script,*args):
  with (out/(Path(script).stem+'.log')).open('w') as stream:
   process=subprocess.Popen([sys.executable,script,*args],stdout=stream,stderr=subprocess.STDOUT)
   while process.poll() is None:
    update(child_pid=process.pid);time.sleep(15)
   if process.returncode:raise RuntimeError(f'{script} exited {process.returncode}')
 try:
  update(stage='waiting_for_training')
  while True:
   current=json.loads((campaign/'status.json').read_text())
   if current['phase']=='completed':break
   if current['phase'] in ('error','failed','interrupted'):raise RuntimeError('training ended without completion')
   pid=current.get('pid')
   if pid is None:raise RuntimeError('missing training controller PID')
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('training controller process missing; no implicit restart')
   update(stage='waiting_for_training',training_pid=pid,policy_transitions=current.get('policy_transitions_total'),completed_jobs=current.get('completed_jobs'))
   time.sleep(30)
  resolve_comparison_actors(campaign)
  update(stage='evaluating')
  child('learning/cli/evaluate_tolerance_comparison.py','--campaign',str(campaign),'--output',str(out/'evaluation'))
  update(stage='rendering_report')
  child('learning/cli/report_tolerance_comparison.py',str(out/'evaluation'))
  report=out/'evaluation/report/REPORT.md'
  if not report.exists():raise RuntimeError('report output missing')
  update(state='completed',phase='completed',stage='report_ready_for_review',report=str(report),adoption_status='pending_analysis')
  atomic_json(campaign/'analysis_status.json',dict(phase='completed',state='completed',report=str(report),adoption_status='pending_analysis',checked_unix=time.time()))
 except BaseException as exc:
  update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':main()
