"""Run a declared single-lane approach matrix sequentially through existing tools."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--cuda-graph-observation',action='store_true');a=p.parse_args()
m=json.loads(a.manifest.read_text());root=a.manifest.parent
for case in m['cases'][a.start:a.start+a.count]:
 config=Path(case['config']);spec=json.loads(config.read_text());out=Path(spec['output'])
 if a.resume and (root/(case['case_id']+'_summary.json')).exists():continue
 if a.resume and not case.get('analysis_kind') and not spec.get('batch_candidates') and (out/'chain_summary.json').exists() and (out/'xy_jump.png').exists() and (not (out/'task_execution.json').exists() or (out/'task_outcome.json').exists()):
  (root/(case['case_id']+'_summary.json')).write_text((out/'chain_summary.json').read_text())
  continue
 reuse=a.resume and (out/'results.json').exists()
 if not reuse:
  if a.resume and out.exists():out.rename(out.with_name(out.name+'_interrupted_'+str(time.time_ns())))
  out.mkdir(parents=True,exist_ok=False)
 (root/'status.json').write_text(json.dumps(dict(status='running',case_id=case['case_id'],started_unix=time.time())))
 if not reuse:
  command=[m['simulation_python'],'learning/cli/evaluate_isaac_pendulum_jump.py','--config',str(config)]
  if a.cuda_graph_observation:command.append('--cuda-graph-observation')
  with (out/'runtime.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:
   (root/'status.json').write_text(json.dumps(dict(status='failed',case_id=case['case_id'],returncode=r.returncode)))
   sys.exit(r.returncode)
 if case.get('analysis_kind')=='ground_speed':
  import numpy as np
  d=np.load(out/'trajectory.npz');t=d['trace'][d['valid'][:,0],0];tail=t[t[:,0]>=2.];physical=json.loads((out/'results.json').read_text())['cases'][0]
  measured=dict(role='Flat ground speed readiness, not a jump task',requested_speed_m_s=case['speed_m_s'],full_window=not physical['physical_termination'] and physical['duration_s']>=3.99,measured_last2s_mean_m_s=float(tail[:,4].mean()),measured_last2s_std_m_s=float(tail[:,4].std()),max_abs_roll_deg=float(np.rad2deg(abs(t[:,5])).max()))
  (root/(case['case_id']+'_summary.json')).write_text(json.dumps(measured,indent=2)+'\n')
  continue
 outputs=[Path(r['output']) for r in json.loads((out/'batch_results.json').read_text())['cases']] if spec.get('batch_candidates') else [out]
 summaries=[]
 for result_out in outputs:
  scripts=['analyze_isaac_pendulum_landing.py','plot_isaac_pendulum_jump.py'] if case.get('analysis_kind')=='jump_only' else ['analyze_isaac_pendulum_landing.py','analyze_rb_continuous_task.py','plot_isaac_pendulum_jump.py']
  if (result_out/'task_execution.json').exists():scripts.append('finalize_task_execution.py')
  for script in scripts:
   command=[m['analysis_python'],'learning/cli/'+script,'--run',str(result_out)]
   if script=='finalize_task_execution.py':command.extend(['--output',str(result_out/'task_outcome.json')])
   analysis=subprocess.run(command)
   if analysis.returncode:
    (root/'status.json').write_text(json.dumps(dict(status='failed',stage='analysis',script=script,case_id=case['case_id'],returncode=analysis.returncode)))
    sys.exit(analysis.returncode)
  summary_file='landing.json' if case.get('analysis_kind')=='jump_only' else 'chain_summary.json'
  summaries.append(json.loads((result_out/summary_file).read_text()))
 (root/(case['case_id']+'_summary.json')).write_text(json.dumps(dict(cases=summaries) if spec.get('batch_candidates') else summaries[0],indent=2)+'\n')
(root/'status.json').write_text(json.dumps(dict(status='completed_slice',start=a.start,count=a.count,finished_unix=time.time())))
