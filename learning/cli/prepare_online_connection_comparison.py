"""Freeze shared evidence and prepare actual exit-only C6 versus rolling R6 runs."""
import argparse
import json
import shutil
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--template',type=Path,nargs='+',required=True)
p.add_argument('--records',type=Path,nargs='+',required=True)
p.add_argument('--selector-config',type=Path,default=Path('learning/configs/jump_connection_capability_dev.json'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sources=[json.loads(path.read_text()) for path in a.template]
if len({s['origin_id'] for s in sources})!=len(sources):p.error('Use one template per distinct origin')
a.output.mkdir(parents=True,exist_ok=False)
inputs=a.output/'input';inputs.mkdir();records=[]
for i,path in enumerate(a.records):
    dest=inputs/f'evidence_{i}.jsonl';shutil.copy2(path,dest);records.append(str(dest))
config=inputs/'selector.json';shutil.copy2(a.selector_config,config);cases=[]
for source in sources:
    folder=a.output/source['origin_id'] if len(sources)>1 else a.output
    folder.mkdir(exist_ok=True)
    for method,stages in [('C6',['exit']),('R6',['exit','midpoint'])]:
        spec=dict(source,output=str(folder/method),role='Matched online DEV C6/R6; evidence frozen before execution; queried origin excluded',
            connection_selector=dict(config=str(config),records=records,stages=stages))
        spec.pop('connection_midpoint_candidate',None)
        if method=='R6':spec['connection_midpoint_candidate']=dict(source['connection_candidate'],decision_x_m=(source['connection_candidate']['decision_x_m']+source['handoff_x'])/2)
        path=folder/(method+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
        cases.append(dict(case_id=source['origin_id']+'_'+method,method=method,origin_id=source['origin_id'],config=str(path),distance_m=source['platform_front_x']-source['connection_candidate']['decision_x_m'],speed_m_s=source['connection_candidate']['speed_m_s'],alpha=source['connection_candidate']['alpha']))
(a.output/'manifest.json').write_text(json.dumps(dict(role='Paired online DEV methods with shared frozen evidence; not hindsight oracle or independent TEST',simulation_python='/home/qy/miniconda3/envs/env_isaaclab/bin/python',analysis_python='/home/qy/miniconda3/envs/mjp/bin/python',cases=cases,maximum_physics_steps=sum(2*round(s['seconds']/s['physics_dt']) for s in sources),optimizer_updates=0,status='prepared_not_started'),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),origins=[s['origin_id'] for s in sources],maximum_physics_steps=sum(2*round(s['seconds']/s['physics_dt']) for s in sources))))
