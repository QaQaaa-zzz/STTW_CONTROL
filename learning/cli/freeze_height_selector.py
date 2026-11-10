"""Freeze task-conditioned alpha from height DEV before held-out layout evaluation."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--test-run',type=Path,required=True);a=p.parse_args()
source=a.development/'analysis_summary.json';data=json.loads(source.read_text())['rows'];spec=json.loads((a.test_run/'protocol.json').read_text());rules={}
for height in spec['heights_mm']:
 for speed in spec['speeds']:
  for task,limits in spec['tasks'].items():
   candidates=[]
   for alpha in spec['fixed_alphas']:
    label=f'fixed{alpha:g}'
    rows=[r for r in data if r['terrain']==f'height{height:02d}mm' and abs(r['requested_speed_m_s']-speed)<1e-6 and r['label']==label]
    if len(rows)!=3:raise ValueError('Expected three declared DEV physical groups per candidate')
    success=sum(r['tasks'][task] for r in rows)
    cost=sum(10*int(r['failed'])+r['path_rmse_m']/limits['W_m']+r['speed_rmse_m_s']/limits['speed_rmse_m_s'] for r in rows)/len(rows)
    candidates.append(dict(alpha=alpha,task_pass=success,mean_cost=cost))
   best=min(candidates,key=lambda c:(-c['task_pass'],c['mean_cost'],c['alpha']))
   rules[f'{height}:{speed}:{task}']=dict(selected_alpha=best['alpha'],candidates=candidates)
frozen=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rule='maximize DEV task pass, then minimize mean failure penalty plus normalized path/speed RMSE, then lower alpha',rules=rules,test_data_used=False)
(a.test_run/'frozen_selector.json').write_text(json.dumps(frozen,indent=2)+'\n')
for job in json.loads((a.test_run/'jobs.json').read_text()):
 path=Path(job['config']);cfg=json.loads(path.read_text())
 for task in spec['tasks']:
  lane=spec['case_labels'].index('rule_'+task);cfg['alphas'][lane]=rules[f"{job['height_mm']}:{job['speed_m_s']}:{task}"]['selected_alpha']
 path.write_text(json.dumps(cfg,indent=2)+'\n')
print('Selector frozen from DEV; no test outputs inspected')
