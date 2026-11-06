"""Execute the frozen two-policy adaptation budget sequentially, then score saved updates."""
import argparse,copy,json,os,subprocess
from pathlib import Path
import numpy as np
from sttw_control.phase_u_selection import select_adapted_skill
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--nominal-baseline-manifest',type=Path)
a=p.parse_args();plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
baselines=json.loads(a.nominal_baseline_manifest.read_text())['cases'] if a.nominal_baseline_manifest else []
os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
status={}
def run(args,stage,log):
 status.update(stage=stage,command=[str(v) for v in args]);(a.output/'status.json').write_text(json.dumps(status,indent=2))
 with log.open('w') as stream:subprocess.run([str(v) for v in args],stdout=stream,stderr=subprocess.STDOUT,check=True)
all_results=[]
for cfgpath,panelpath in zip(plan['training_configs'],plan['panels']):
 cfg=json.loads(Path(cfgpath).read_text());panel=json.loads(Path(panelpath).read_text());skill=Path(cfg['bundle']).name
 root=a.output/skill;root.mkdir();sim=panel['simulation_python'];analysis=panel['analysis_python']
 # The training process exits before any evaluation process starts; no extra resident training GPU app.
 run([sim,'learning/cli/train_isaac_phase_u.py','--config',cfgpath,'--output',root/'training_run'],f'{skill}:training',root/'training.log')
 trained=json.loads((root/'training_run/status.json').read_text())
 if trained['stage']!='training_complete':raise RuntimeError('Training did not reach its declared end')
 rows=[]
 for update in plan['evaluated_updates']:
  folder=root/f'update{update:03d}';folder.mkdir();bundle=folder/'bundles'/skill
  checkpoint=root/'training_run/training'/('model_initial.pt' if update==0 else f'model_update_{update:05d}.pt')
  run([sim,'learning/cli/export_trained_phase_u.py','--checkpoint',checkpoint,'--source-bundle',cfg['bundle'],'--output',bundle],f'{skill}:export{update}',folder/'export.log')
  current=copy.deepcopy(panel)
  result_paths={}
  if update==0 and baselines:
   with np.load(bundle/'actor.npz') as exported,np.load(Path(cfg['bundle'])/'actor.npz') as original:
    if set(exported.files)!=set(original.files) or any(not np.array_equal(exported[k],original[k]) for k in original.files):
     raise ValueError('Original baseline reuse requires identical exported initial policy arrays')
  for case in current['cases']:
   spec=json.loads(Path(case['config']).read_text())
   if update==0 and case['retention'] and baselines:
    match=next(b for b in baselines if b['skill']==skill and Path(b['source_panel_config']).resolve()==Path(case['config']).resolve())
    old=json.loads(Path(match['config']).read_text());old_out=Path(old['output'])
    if {k:v for k,v in old.items() if k!='output'}!={k:v for k,v in spec.items() if k!='output'}:
     raise ValueError('Nominal baseline physics, timing or task differs from frozen panel')
    protocol=json.loads((old_out/'protocol.json').read_text())
    identity=json.loads((Path(cfg['bundle'])/'identity.json').read_text())
    if protocol['policies'][0]['artifact_sha256']!=identity['artifact_sha256']:
     raise ValueError('Executed nominal baseline used a different original policy')
    result_paths[case['case_id']]=old_out
   spec.update(bundles=str(bundle.parent),output=str(folder/case['case_id']))
   path=folder/(case['case_id']+'.json');path.write_text(json.dumps(spec,indent=2)+'\n');case['config']=str(path)
  pending=dict(current,cases=[c for c in current['cases'] if c['case_id'] not in result_paths])
  path=folder/'manifest.json';path.write_text(json.dumps(pending,indent=2)+'\n')
  run([sim,'learning/cli/run_jump_approach_pilot.py','--manifest',path,'--count',len(pending['cases'])],f'{skill}:evaluate{update}',folder/'evaluation.log')
  row=dict(update=update,checkpoint=str(checkpoint),nominal={},hard={},reused_original_nominal={k:str(v) for k,v in result_paths.items()})
  for case in current['cases']:
   out=result_paths.get(case['case_id'],folder/case['case_id'])
   if case['retention']:
    result=json.loads((out/'results.json').read_text())['cases'][0];landing=json.loads((out/'landing.json').read_text())['cases'][0]
    row['nominal'][case['case_id']]=dict(full_window=not result['physical_termination'] and result['duration_s']>=case['seconds']-.006,strict_support=landing['landing_hold_1s'])
   else:
    result=json.loads((out/'chain_summary.json').read_text())
    row['hard'][case['case_id']]=dict(success=result['complete'],time_s=result['success_time_s'])
  rows.append(row);(root/'evaluated_checkpoints.json').write_text(json.dumps(rows,indent=2)+'\n')
 selected=select_adapted_skill(rows);(root/'selected_model.json').write_text(json.dumps(selected,indent=2)+'\n')
 all_results.append(dict(skill=skill,results=str(root/'evaluated_checkpoints.json'),selection=selected))
status.update(stage='completed',results=all_results);(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
