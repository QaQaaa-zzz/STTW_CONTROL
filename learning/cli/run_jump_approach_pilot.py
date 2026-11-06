"""Run a declared single-lane approach matrix sequentially through existing tools."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text());root=a.manifest.parent
for case in m['cases'][a.start:a.start+a.count]:
 config=Path(case['config']);out=Path(json.loads(config.read_text())['output']);out.mkdir(parents=True,exist_ok=False)
 (root/'status.json').write_text(json.dumps(dict(status='running',case_id=case['case_id'],started_unix=time.time())))
 with (out/'runtime.log').open('w') as log:
  r=subprocess.run([m['simulation_python'],'learning/cli/evaluate_isaac_pendulum_jump.py','--config',str(config)],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  (root/'status.json').write_text(json.dumps(dict(status='failed',case_id=case['case_id'],returncode=r.returncode)))
  sys.exit(r.returncode)
 for script in ['analyze_isaac_pendulum_landing.py','analyze_rb_continuous_task.py','plot_isaac_pendulum_jump.py']:
  subprocess.run([m['analysis_python'],'learning/cli/'+script,'--run',str(out)],check=True)
 (root/(case['case_id']+'_summary.json')).write_text((out/'chain_summary.json').read_text())
(root/'status.json').write_text(json.dumps(dict(status='completed_slice',start=a.start,count=a.count,finished_unix=time.time())))
