"""Verify complete fixed witness-selection results; no model selection."""
import json,hashlib,argparse
from pathlib import Path
KEYS=('route','speed','alpha','force_N','onset_s')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.root/'runs/preference_witness_selection_v1';ev=run/'evaluation_attempt2';s=json.loads((ev/'status.json').read_text());assert s['phase']=='completed' and s['completed_cases']==1620
 results={};sources={};cases=ticks=0
 for seed in (918501,918502,918503):
  for arm in ('witness','random_success'):
   t=run/f'{arm}_{seed}'/'training';st=json.loads((t/'status.json').read_text());assert st['completed_updates']==400 and st['phase']=='completed'
   rec=json.loads((t/'candidate_record.json').read_text())
   for field in ('actor','checkpoint'):assert hashlib.sha256(Path(rec[field]).read_bytes()).hexdigest()==rec[field+'_sha256']
  for panel in ('strong','straight','low_speed'):
   folder=ev/f'seed{seed}_{panel}';raw=folder/'raw';status=json.loads((raw/'status.json').read_text());assert status['phase']=='completed' and (folder/'analysis/REPORT.md').exists()
   file=raw/'results.json';rows=json.loads(file.read_text())['cases'];assert len(rows)==status['declared_cases'];cases+=len(rows);ticks+=status['active_lower_ticks'];sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
   groups={p:{tuple(r[k] for k in KEYS):r for r in rows if r['policy']==p} for p in ('witness','random_success')};x=groups['random_success'];y=groups['witness'];assert x.keys()==y.keys() and len(x)+len(y)==len(rows)
   pair=dict(n=len(x),random_success=sum(r['task_recovered'] for r in x.values()),witness=sum(r['task_recovered'] for r in y.values()),gains=sum(not x[k]['task_recovered'] and y[k]['task_recovered'] for k in x),losses=sum(x[k]['task_recovered'] and not y[k]['task_recovered'] for k in x),working={p:sum(r['performance_success'] for r in g.values()) for p,g in groups.items()})
   results[f'{seed}_{panel}']=pair
 assert cases==1620 and ticks==s['active_lower_ticks']
 out=dict(status=s,results=results,sources=sources,scope='Used DEV, shared TRAIN corpus; optimization/subset repeats, not independent environments; aggregate counts repeat same contexts.')
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n');print('Verified',cases,'cases',ticks,'ticks')
if __name__=='__main__':main()
