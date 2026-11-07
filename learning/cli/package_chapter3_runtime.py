"""Collect the actual chapter-three runtime inputs into relocatable Git assets."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--spec',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--training-record',type=Path)
p.add_argument('--include',type=Path,action='append',default=[])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
mapped={};references=[]
file_fields={'plant_config','model_config','initial_state','ground_initial_state',
 'geometry','ground_settings','ground_route','controller_config','lower_actor',
 'map_contract','planner_config','branch_reference','allowed_route_map',
 'task_spec','asset_spec','initial_checkpoint','reachable_entry_bank','resume_checkpoint'}

def collect(path):
    source=Path(path).resolve()
    if str(source) in mapped:return mapped[str(source)]
    target=a.output/'inputs'/f'{len(mapped):03d}_{source.name}'
    target.parent.mkdir(parents=True,exist_ok=True)
    mapped[str(source)]=str(target)
    references.append(dict(source=str(source),packaged=str(target)))
    if source.is_dir():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','*.egg-info'))
    elif source.suffix=='.json':target.write_text(json.dumps(rewrite(json.loads(source.read_text())),indent=2)+'\n')
    else:shutil.copyfile(source,target)
    return str(target)

def rewrite(value):
    if isinstance(value,list):return [rewrite(x) for x in value]
    if not isinstance(value,dict):return value
    result={}
    for key,item in value.items():
        if key=='asset' and isinstance(item,str):
            result[key]=str(Path(collect(Path(item).parent))/Path(item).name)
        elif key in file_fields and isinstance(item,str):result[key]=collect(item)
        elif key in ('record_files','records') and isinstance(item,list) and all(isinstance(x,str) for x in item):result[key]=[collect(x) for x in item]
        elif key=='config' and isinstance(item,str) and Path(item).is_file():result[key]=collect(item)
        elif key in ('scene_terrain','ground_bundle','bundle') and isinstance(item,str):result[key]=collect(item)
        elif key=='go2_checkout':
            target=a.output/'go2';target.mkdir(exist_ok=True)
            for sub in ('rsl_rl','legged_gym'):
                if not (target/sub).exists():shutil.copytree(Path(item)/sub,target/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
            shutil.copyfile(Path(item)/'LICENSE',target/'LICENSE')
            result[key]=str(target)
        elif key=='bundles':
            target=a.output/'policies';target.mkdir(exist_ok=True)
            for skill in value['skills']:
                if not (target/skill).exists():shutil.copytree(Path(item)/skill,target/skill)
            result[key]=str(target)
        else:result[key]=rewrite(item)
    return result

cases=[]
for source in a.spec:
    spec=json.loads(source.read_text());spec=rewrite(spec)
    name=source.parent.name+'_'+source.stem
    spec['output']='runs/chapter3_portable/'+name
    dest=a.output/(name+'.json');dest.write_text(json.dumps(spec,indent=2)+'\n')
    cases.append(dict(source=str(source.resolve()),config=str(dest),output=spec['output']))
if a.training_record:
    training=rewrite(json.loads(a.training_record.read_text())['task'])
    (a.output/'historical_training_config.json').write_text(json.dumps(training,indent=2)+'\n')
for path in a.include:collect(path)
source_paths=subprocess.check_output(['git','ls-files','--','learning/src','learning/cli','learning/configs','learning/scripts','learning/pyproject.toml'],text=True).splitlines()
source_paths=sorted(set(source_paths+[str(Path(__file__).resolve().relative_to(Path.cwd()))]))
sources=[dict(path=name,sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest()) for name in source_paths]
inventory=[]
for path in sorted(a.output.rglob('*')):
    if path.is_file():inventory.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
manifest=dict(code_revision_before_delivery=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    cases=cases,inputs=references,files=inventory,source_files=sources,
    scope='Project runtime inputs and Go2 training source. Isaac Lab/Isaac Sim/CUDA and Python dependencies remain declared external installations. Historical identity provenance paths are not runtime reads.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),files=len(inventory),bytes=sum(f['bytes'] for f in inventory))))
