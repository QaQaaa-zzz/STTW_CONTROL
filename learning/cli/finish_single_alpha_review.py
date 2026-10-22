"""Wait for the existing single-alpha run and analyze saved panels, no new physics."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_best import atomic_json
from analyze_alpha_practical import analyze


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 run=Path(a.run).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'status.json').exists():raise FileExistsError('analysis attempt exists')
 state=dict(state='running',phase='running',pid=os.getpid(),source=str(run),new_physics=0,new_training=0)
 def update(**kw):
  state.update(kw);state['checked_unix']=time.time();atomic_json(out/'status.json',state)
 try:
  update(stage='waiting_for_single_alpha')
  while True:
   status=json.loads((run/'status.json').read_text())
   if status['state']=='completed':break
   if status['state'] in ('error','failed','interrupted'):raise RuntimeError('source run ended unsuccessfully')
   try:os.kill(status['pid'],0)
   except ProcessLookupError:raise RuntimeError('source process missing before completed receipt')
   update(completed_updates=status.get('completed_updates'),training_pid=status['pid']);time.sleep(30)
  if status.get('completed_updates')!=400:raise RuntimeError('expected completed400 run')
  panels=sorted((run/'training/validation').glob('update_0[34]*_stage3'))
  final=run/'training/validation/update_0400_stage3'
  if not (final/'metrics.json').exists():raise RuntimeError('final400 panel missing')
  summaries=[]
  for panel in panels:
   if not (panel/'metrics.json').exists():continue
   update(stage='analyzing_saved_panel',panel=str(panel));target=out/panel.name;analyze(panel,target)
   result=json.loads((target/'analysis.json').read_text());rows=result['cases']
   relaxed=sum(r['passed'] for r in result['sensitivity'] if r['path_limit_m']==.2 and r['roll_limit_rad']==.33)
   turns=[r for r in result['endpoint_comparison'] if not r['case'].startswith('straight')]
   summaries.append(dict(panel=panel.name,source=str(panel),relaxed_pass=relaxed,n=len(rows),physical_failures=sum(r['physical'] for r in rows),domain_failures=sum(r['domain'] for r in rows),expected_turn_tradeoff=sum(r['alpha1_speed_rmse_advantage']>0 and r['alpha0_path_rmse_advantage']>0 for r in turns),turn_cases=len(turns),maximum_tail_path_m=max(r['tail_max_path_m'] for r in rows),maximum_peak_roll_rad=max(r['peak_roll'] for r in rows),report=str(target/'REPORT.md')))
  atomic_json(out/'comparison.json',dict(candidates=summaries,interpretation='posthoc practical criteria; not automatic adoption',new_physics=0))
  lines=['# Single-alpha final-stage review','', 'Existing stage3 panels only. Practical sensitivity: final1s path<=0.20m, heading<=0.05rad, speed<=0.10m/s, peak roll<=0.33rad, no physical/domain failure and goal reached. Original labels remain in each source.','', '| Checkpoint panel | Practical pass | Expected turn trade-offs | Max final path error | Peak roll |','|---|---:|---:|---:|---:|']
  for r in summaries:lines.append(f"| [{r['panel']}]({r['panel']}/REPORT.md) | {r['relaxed_pass']}/{r['n']} | {r['expected_turn_tradeoff']}/{r['turn_cases']} | {r['maximum_tail_path_m']:.3f}m | {r['maximum_peak_roll_rad']:.3f}rad |")
  lines+=['','Next decision: inspect final and prior candidate XY/velocity curves, retain useful speed-path trade-offs, and distinguish tolerable steady bias from failures requiring training changes. This report alone does not mark user satisfaction or adoption.']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
  update(state='completed',phase='completed',stage='ready_for_analysis',adoption_status='pending',report=str(out/'REPORT.md'))
 except BaseException as exc:update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise
if __name__=='__main__':main()
