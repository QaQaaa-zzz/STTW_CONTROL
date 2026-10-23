"""Offline finite-intervention complementarity; not a learned selector evaluation."""
from pathlib import Path
import argparse,json,hashlib
RUNS=('preference_train_coordination_records_v1','tolerance_recovery_opportunity_v1','tolerance_teaching_search_pilot_v1')
KEYS=('route','speed','alpha','force_N','onset_s')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();reports=[]
 for name in RUNS:
  root=a.root/'runs'/name;run=root/'evaluation';decl=json.loads((root/'declaration.json').read_text());status=json.loads((run/'status.json').read_text());assert status['phase']=='completed' and 'TRAIN' in decl['data_role']
  rows=json.loads((run/'results.json').read_text())['cases'];assert len(rows)==status['declared_cases']
  receipts=list(run.glob('intervention_branch_t*.json'));assert len(receipts)==len(decl['onsets_s'])
  for q in receipts:
   d=json.loads(q.read_text());assert all(d[k] for k in ('complete_state_tree_copied','history_and_filter_preserved','alpha_unchanged'))
  groups={}
  for r in rows:
   if r['force_N']==0:continue
   k=tuple(r[x] for x in KEYS);g=groups.setdefault(k,{});assert r['policy'] not in g;g[r['policy']]=r
  cases=[]
  for k,g in groups.items():
   assert set(g)=={'control','speed_off','steer_off'} and all(r['prefix_valid'] for r in g.values())
   c=g['control'];items={}
   for pol,r in g.items():
    items[pol]=dict(success=r['task_recovered'],return_value=r['logged_return'],peak_roll=r['peak_roll_rad'],gain=not c['task_recovered'] and r['task_recovered'],harm=c['task_recovered'] and not r['task_recovered'],useful=not c['task_recovered'] and r['task_recovered'] and r['logged_return']>c['logged_return'])
   cases.append(dict(context=dict(zip(KEYS,k)),arms=items))
  count=lambda fn:sum(fn(x['arms']) for x in cases)
  summary=dict(n=len(cases),success={pol:count(lambda g:g[pol]['success']) for pol in ('control','speed_off','steer_off')},union_success=count(lambda g:any(r['success'] for r in g.values())),control_speed_union=count(lambda g:g['control']['success'] or g['speed_off']['success']),steer_unique=count(lambda g:g['steer_off']['success'] and not(g['control']['success'] or g['speed_off']['success'])),gains={pol:count(lambda g:g[pol]['gain']) for pol in ('speed_off','steer_off')},harms={pol:count(lambda g:g[pol]['harm']) for pol in ('speed_off','steer_off')},useful={pol:count(lambda g:g[pol]['useful']) for pol in ('speed_off','steer_off')})
  reports.append(dict(run=name,role=decl['data_role'],window=decl['recovery_intervention'],summary=summary,cases=cases,sources={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [root/'declaration.json',run/'status.json',run/'results.json',*receipts]}))
 out=dict(scope='Existing TRAIN data; posthoc finite-bank union is not physical upper bound, deployable selector, or new test. No new simulations or fitting.',runs=reports)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in reports:print(r['run'],r['summary'])
if __name__=='__main__':main()
