"""Reuse matched panel cases and expand a frozen adapted pair on original protocols."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifests',type=Path,nargs='+',required=True)
p.add_argument('--adaptation',type=Path,required=True)
p.add_argument('--update',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
ignored={'output','role','comparison_reference'}
def contract(spec):return {k:v for k,v in spec.items() if k not in ignored}
all_cases=[];pending=[];budget=0
for path in a.manifests:
 m=json.loads(path.read_text())
 for case in m['cases']:
  skill=case['skill'];stage=a.adaptation/skill/f'update{a.update:03d}'
  old=json.loads(Path(case['config']).read_text())
  spec=dict(old,bundles=str(stage/'bundles'),output=str(a.output/case['case_id']))
  row=dict(case,original_reference_config=case['config'],adapted_update=a.update)
  panel=json.loads((stage/'manifest.json').read_text())
  matches=[]
  for item in panel['cases']:
   prior=json.loads(Path(item['config']).read_text())
   if contract(prior)==contract(spec) and (Path(prior['output'])/'chain_summary.json').exists():matches.append(item)
  if matches:
   row.update(config=matches[0]['config'],execution='reused_completed')
  else:
   dest=a.output/(case['case_id']+'.json');dest.write_text(json.dumps(spec,indent=2)+'\n')
   row.update(config=str(dest),execution='prepared_not_started');pending.append(row)
   budget+=round(spec['seconds']/spec['physics_dt'])
  all_cases.append(row)
meta=dict(role='Matched equal-budget adapted source/RB DEV; not independent TEST or a claim of original-policy recovery advantage',simulation_python=m['simulation_python'],analysis_python=m['analysis_python'],maximum_physics_steps=budget,optimizer_updates=0,adapted_update=a.update)
for name,cases in [('manifest.json',pending),('all_cases.json',all_cases)]:
 (a.output/name).write_text(json.dumps(dict(meta,cases=cases),indent=2)+'\n')
print(json.dumps(dict(total=len(all_cases),reused=len(all_cases)-len(pending),pending=len(pending),maximum_physics_steps=budget)))
