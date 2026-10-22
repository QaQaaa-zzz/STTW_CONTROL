"""Post-hoc outcome decomposition with original effort thresholds frozen."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sttw_control.tolerance_outcome_audit import split_outcome
from audit_tolerance_model_energy import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 raw=json.loads((a.run/'results.json').read_text())['cases'];model=json.loads((a.model/'summary.json').read_text());energies={r['name']:r for r in model['rows']};rows=[]
 for r in raw:
  with np.load(a.run/(r['name']+'.npz')) as data:x=split_outcome(data,onset=r['onset_s'],limit=.302,final_hold=r['final_hold'],goal=r['goal'])
  assert x['reconstructed_full_success']==r['success'],r['name']
  rows.append(dict(**energies[r['name']],**x))
 summaries={}
 for role in ['calibration','speed_transfer']:
  subset=[r for r in rows if r['data_role']==role];summaries[role]={}
  for group,rr in [('all',subset),('valid_prefix',[r for r in subset if not any(r['prefix_'+k] for k in ['working_failure','physical_failure','domain_failure'])])]:
   summaries[role][group]={}
   for label in ['suffix_working_failure','suffix_task_success']:
    valid=[r for r in rr if r[label] is not None];scored=[dict(r,actual_success=not r[label] if label.endswith('failure') else r[label]) for r in valid]
    summaries[role][group][label]=dict(n=len(valid),successes=sum(r['actual_success'] for r in scored),**{m:score(scored,model['methods'][m]['effort_threshold'],m) for m in ['static','dynamic']})
 a.output.mkdir(parents=True);(a.output/'summary.json').write_text(json.dumps(dict(summaries=summaries,rows=rows,scope='Post-hoc descriptive decomposition; no refit; original episode labels retained'),indent=2))
 (a.output/'REPORT.md').write_text('# Outcome-window audit\n\n'+json.dumps(summaries,indent=2)+'\n\nPost-hoc diagnostic only. The original effort thresholds are unchanged. Suffix success does not repair an earlier episode failure; full original labels reconstruct exactly. Same-state samples are correlated.\n')
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
