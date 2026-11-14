"""Run the already frozen E1 pilot; acquire the shared GPU lock outside this CLI."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text());started=time.time();completed=[]
for entry in manifest['entries']:
    if entry.get('reused_evaluation'):
        completed.append(entry['layout']);continue
    status=dict(stage='running',layout=entry['layout'],completed_layouts=completed,total_layouts=len(manifest['entries']),elapsed_s=time.time()-started)
    (a.root/'status.json').write_text(json.dumps(status,indent=2))
    command=[sys.executable,'learning/cli/evaluate_playground_handoff.py','--config',entry['config'],'--output',entry['evaluation'],'--skills','rb_run3','--modes','h75','--routes',*[str(i) for i in manifest.get('routes',range(6))],'--seconds',str(manifest['seconds']),'--physics-substeps','5','--ground-only']
    with (Path(entry['config']).parent/'run.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status.update(stage='failed',exit_code=result.returncode,elapsed_s=time.time()-started);(a.root/'status.json').write_text(json.dumps(status,indent=2));sys.exit(result.returncode)
    completed.append(entry['layout'])
status=dict(stage='complete',completed_layouts=completed,total_layouts=len(manifest['entries']),elapsed_s=time.time()-started);(a.root/'status.json').write_text(json.dumps(status,indent=2))
