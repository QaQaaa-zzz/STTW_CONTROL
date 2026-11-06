"""Expand identical approach candidates across frozen source and RB skills."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--approach-manifest',type=Path,required=True)
p.add_argument('--existing-manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--disturbance-configs',type=Path,nargs='+',help='Existing configs supplying exact external_force_pulse protocols')
p.add_argument('--distance',type=float,help='Optional approach distance subset')
a=p.parse_args()
approaches=json.loads(a.approach_manifest.read_text());existing=json.loads(a.existing_manifest.read_text())
skills={c['skill']:json.loads(Path(c['config']).read_text())['bundles'] for c in existing['cases']}
ignored={'output','role','comparison_reference'}
def physical_spec(s):
    return {k:v for k,v in s.items() if k not in ignored}
old=[(c,json.loads(Path(c['config']).read_text())) for c in existing['cases']]
a.output.mkdir(parents=True,exist_ok=False)
all_cases=[];pending=[];budget=0
expanded=[]
for case in approaches['cases']:
    if a.distance is not None and case['distance_m']!=a.distance:continue
    source=json.loads(Path(case['config']).read_text())
    if a.disturbance_configs:
        for index,path in enumerate(a.disturbance_configs):
            pulse=json.loads(path.read_text())['external_force_pulse']
            expanded.append((dict(case,case_id=case['case_id']+'_pulse'+str(index)),dict(source,external_force_pulse=pulse)))
    else:expanded.append((case,source))
for case,source in expanded:
    for skill,bundles in skills.items():
        ident=case['case_id']+'_'+skill
        spec=dict(source,skills=[skill],bundles=bundles,output=str(a.output/ident),
            role='Matched cross-skill DEV; source/RB identical adaptation; update2 separate engineering reference',
            comparison_reference=dict(config=case['config'],run=source['output'],skill=source['skills'][0]))
        if a.disturbance_configs:spec.pop('comparison_reference')
        matches=[(c,s) for c,s in old if physical_spec(s)==physical_spec(spec) and (Path(s['output'])/'chain_summary.json').exists()]
        row=dict(case,case_id=ident,skill=skill,origin_id=source.get('origin_id','rb_nominal_upstream_01'))
        if matches:
            c,s=matches[0];row.update(config=c['config'],execution='reused_completed')
        else:
            config=a.output/(ident+'.json');config.write_text(json.dumps(spec,indent=2)+'\n')
            row.update(config=str(config),execution='prepared_not_started');pending.append(row)
            budget+=round(spec['seconds']/spec['physics_dt'])
        all_cases.append(row)
meta=dict(role='Matched source/RB approach DEV; no training, not independent TEST',
    simulation_python=existing['simulation_python'],analysis_python=existing['analysis_python'],
    maximum_physics_steps=budget,optimizer_updates=0,
    limits='DEV roots only; optional prescribed disturbances are deterministic conditions, not a sampled success probability. Independent roots still required. Cross-selection requires all six candidates per skill and disturbance.')
(a.output/'manifest.json').write_text(json.dumps(dict(meta,cases=pending),indent=2)+'\n')
(a.output/'all_cases.json').write_text(json.dumps(dict(meta,cases=all_cases),indent=2)+'\n')
print(json.dumps(dict(total=len(all_cases),reused=len(all_cases)-len(pending),pending=len(pending),maximum_physics_steps=budget)))
