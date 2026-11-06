"""Prepare declared DEV roots for two-stage enumeration; reuse prior configurations."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--approaches',type=Path,nargs='+',required=True)
p.add_argument('--origins',nargs='+',required=True)
p.add_argument('--existing-sequences',type=Path,required=True)
p.add_argument('--distance',type=float,default=9.6)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
old=json.loads(a.existing_sequences.read_text());existing={c['case_id']:c for c in old['cases']}
templates={}
for file in a.approaches:
    for c in json.loads(file.read_text())['cases']:
        if c['distance_m']!=a.distance:continue
        s=json.loads(Path(c['config']).read_text());origin=s['origin_id']
        if origin in a.origins:templates.setdefault(origin,{})[(c['speed_m_s'],c['alpha'])]=(c,s)
if any(len(templates.get(origin,{}))!=6 for origin in a.origins):p.error('Each declared origin requires six distinct approach candidates')
a.output.mkdir(parents=True,exist_ok=False)
cases=[]
for origin in a.origins:
    candidates=templates[origin]
    combinations=[(c,c) for c in candidates]+[(c,d) for c in candidates for d in candidates if c!=d]
    for first,second in combinations:
        ident=f'{origin}_first_{first[0]}_{first[1]}_then_{second[0]}_{second[1]}'
        if ident in existing:
            cases.append(existing[ident]);continue
        source,spec=candidates[first]
        spec=dict(spec,protocol_id=old['protocol_id'],output=str(a.output/ident),
            role='Declared representative DEV two-stage connection; no independent TEST claim',
            paired_C6_config=source['config'],
            connection_midpoint_candidate=dict(decision_x_m=(spec['connection_candidate']['decision_x_m']+spec['handoff_x'])/2,speed_m_s=second[0],alpha=second[1]))
        config=a.output/(ident+'.json');config.write_text(json.dumps(spec,indent=2)+'\n')
        cases.append(dict(case_id=ident,config=str(config),origin_id=origin,distance_m=a.distance,
            first_candidate=dict(speed_m_s=first[0],alpha=first[1]),second_candidate=dict(speed_m_s=second[0],alpha=second[1]),diagonal=first==second))
meta={k:v for k,v in old.items() if k not in ('cases','maximum_physics_steps','role')}
meta.update(role='Explicit representative DEV roots; prior upstream failures retained in original matrix, no TEST outcomes used',
    origins=a.origins,maximum_physics_steps=sum(round(json.loads(Path(c['config']).read_text())['seconds']/json.loads(Path(c['config']).read_text())['physics_dt']) for c in cases),cases=cases)
(a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
missing=[]
for origin in a.origins:
    for c,spec in templates[origin].values():
        if not (Path(spec['output'])/'chain_summary.json').exists():missing.append(c)
(a.output/'missing_C6_manifest.json').write_text(json.dumps(dict(
    role='Original hold controls needed before diagonal comparison; no policy updates',
    simulation_python=old['simulation_python'],analysis_python=old['analysis_python'],
    maximum_physics_steps=sum(round(json.loads(Path(c['config']).read_text())['seconds']/json.loads(Path(c['config']).read_text())['physics_dt']) for c in missing),
    optimizer_updates=0,cases=missing),indent=2)+'\n')
print(json.dumps(dict(roots=len(a.origins),sequences=len(cases),reused_configurations=sum(c['case_id'] in existing for c in cases))))
