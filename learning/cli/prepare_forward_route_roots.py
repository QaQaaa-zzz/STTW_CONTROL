"""Expand existing forced-route protocols on preassigned training roots."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--roots-manifest',type=Path,required=True)
p.add_argument('--route-manifest',type=Path,required=True)
p.add_argument('--count',type=int,default=6)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();roots=json.loads(a.roots_manifest.read_text());templates=json.loads(a.route_manifest.read_text())
selected={}
for c in roots['cases']:
 if c['split']=='train' and c['origin_id'] not in selected:selected[c['origin_id']]=c
selected=list(selected.values())[:a.count]
a.output.mkdir(parents=True,exist_ok=False);cases=[];budget=0
for root in selected:
 initial=json.loads(Path(root['config']).read_text())
 for template in templates['cases']:
  spec=json.loads(Path(template['config']).read_text());ident=root['origin_id']+'_'+template['case_id']
  for name in ('start_x','start_y','start_heading_rad','initial_offset','origin_id'):spec[name]=initial[name]
  spec.update(data_split='train',output=str(a.output/ident),role='Preassigned training root route outcome; frozen skills, no RL')
  config=a.output/(ident+'.json');config.write_text(json.dumps(spec,indent=2)+'\n')
  cases.append(dict(template,case_id=ident,config=str(config),origin_id=root['origin_id'],split='train'))
  budget+=round(spec['seconds']/spec['physics_dt'])
manifest=dict(role='Paired original/bypass, D0 and two prescribed forces; independent preassigned train roots only',
 simulation_python=roots['simulation_python'],analysis_python=roots['analysis_python'],root_split=roots['root_split'],
 selected_origins=[r['origin_id'] for r in selected],cases=cases,maximum_physics_steps=budget,optimizer_updates=0,status='prepared_not_started',
 scope='All descendants inherit original split; failed upstream roots remain in denominator. No test roots, outcome-based replacement or synthetic exit states.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(roots=len(selected),cases=len(cases),maximum_physics_steps=budget)))
