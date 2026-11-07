"""Evaluate frozen jump candidates together per route, reusing completed batches."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True);p.add_argument('--approach',type=Path,required=True)
a=p.parse_args();r=a.run.resolve();source=a.approach.resolve()
lab='/home/qy/miniconda3/envs/env_isaaclab/bin/python';analysis='/home/qy/miniconda3/envs/mjp/bin/python'
models=['last150','update2','best'];rows=[]
def collect(folder,origin):
    for c in json.loads((folder/'batch_results.json').read_text())['cases']:
        d=Path(c['output']);model=json.loads((d/'protocol.json').read_text())['spec']['skills'][0]
        for tool in ['analyze_isaac_pendulum_landing.py','analyze_rb_continuous_task.py']:
            subprocess.run([analysis,'learning/cli/'+tool,'--run',str(d)],stdout=subprocess.DEVNULL,check=True)
        rows.append(dict(model=model,origin=origin,candidate=c['candidate'],output=str(d)))
    (r/'evaluation_index.json').write_text(json.dumps(rows,indent=2))
for result in sorted((r/'evaluation').glob('*/*/batch_results.json')):
    collect(result.parent,result.parent.name)
for path in sorted(source.glob('rb_connection*.json')):
    origin=path.stem;cfg=json.loads(path.read_text())
    for model in models:
        folder=r/'evaluation'/model/origin
        if (folder/'batch_results.json').exists() and not any(x['model']==model and x['origin']==origin for x in rows):collect(folder,origin)
    done={x['model'] for x in rows if x['origin']==origin}
    pending=[m for m in models if m not in done]
    if not pending:continue
    candidates=cfg['batch_candidates'][6:];skills=[m for m in pending for _ in candidates];n=len(skills)
    folder=r/'evaluation'/'mixed'/origin
    cfg.update(output=str(folder),bundles=str(r/'bundles'),skills=skills,batch_candidates=candidates*len(pending),rear_speed_feedback_gain=[1.]*n,stop_at_ground_handoff=False,maximum_physics_lane_steps=8800*n,role='Same matched DEV tasks; models packed per route; batch layout changed, no physics/control/criteria change; not independent TEST')
    config=r/('mixed_'+origin+'.json');config.write_text(json.dumps(cfg,indent=2))
    (r/'pipeline_status.json').write_text(json.dumps(dict(stage='evaluation',models=pending,origin=origin,completed_cases=len(rows),total_cases=108,batch_lanes=n)))
    with (r/('mixed_'+origin+'.log')).open('w') as log:
        subprocess.run([lab,'learning/cli/evaluate_isaac_pendulum_jump.py','--config',str(config)],stdout=log,stderr=subprocess.STDOUT,check=True)
    collect(folder,origin)
subprocess.run([analysis,'learning/cli/summarize_jump_adaptation.py','--run',str(r)],check=True)
(r/'pipeline_status.json').write_text(json.dumps(dict(stage='complete',completed_cases=len(rows))))
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--message','第三章跳跃108条配对评测已完成，含最新update150。可以继续推进任务。','--receipt',str(r/'completion_notification.json')])
