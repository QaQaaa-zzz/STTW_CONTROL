"""Interpret a task with the configured model and use the frozen scene pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from sttw_control.language_task import call_existing_model, parse_task_request, public_catalog, task_tools

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--catalog',type=Path,required=True)
p.add_argument('--task',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['execute','replay','plan'],default='plan')
p.add_argument('--settings',type=Path,default=Path.home()/'.claude/settings.json')
a=p.parse_args()
catalog=json.loads(a.catalog.read_text());scenes=catalog['scenes']
if len({s['scene_id'] for s in scenes})!=len(scenes):p.error('Scene IDs must be unique')
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'task_request.json').write_text(json.dumps(dict(task=a.task,mode=a.mode,catalog=str(a.catalog),maximum_physics_lane_steps=10400,maximum_model_requests=2),ensure_ascii=False,indent=2)+'\n')


def run():
    messages=[dict(role='user',content=json.dumps(dict(task=a.task,scene_catalog=public_catalog(catalog)),ensure_ascii=False))]
    tools=task_tools(catalog)
    response=call_existing_model(messages,tools,a.settings,a.output,1)
    call=parse_task_request(response,catalog)
    (a.output/'selected_request.json').write_text(json.dumps(call,ensure_ascii=False,indent=2)+'\n')
    if call['name']=='request_task_revision':
        return dict(status='revision_required',reason=call['input']['reason'],physics_executed=False)
    scene=next(s for s in scenes if s['scene_id']==call['input']['scene_id'])
    if a.mode=='plan':return dict(status='planned_not_executed',scene_id=scene['scene_id'],physics_executed=False)
    spec=json.loads(Path(scene['config']).read_text())
    if len(spec['skills'])!=1 or spec['seconds']>26 or round(spec['seconds']/spec['physics_dt'])>10400:
        raise ValueError('Selected scene exceeds the declared single-episode budget')
    if a.mode=='execute':
        spec['output']=str(a.output/'evaluation')
        config=a.output/'config.json';config.write_text(json.dumps(spec,indent=2)+'\n')
        manifest=dict(cases=[dict(case_id=scene['scene_id'],config=str(config))],simulation_python=catalog['simulation_python'],analysis_python=catalog['analysis_python'],optimizer_updates=0,maximum_physics_lane_steps=10400)
        manifest_path=a.output/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        (a.output/'status.json').write_text(json.dumps(dict(status='running',stage='physical_execution',scene_id=scene['scene_id'])))
        command=['/usr/bin/flock',catalog['gpu_lock'],catalog['analysis_python'],'learning/cli/run_jump_approach_pilot.py','--manifest',str(manifest_path),'--count','1']
        subprocess.run(command,check=True)
        subprocess.run([catalog['analysis_python'],'learning/cli/summarize_multiscene_chain.py','--manifest',str(manifest_path)],check=True)
        physical_run=a.output/'evaluation'
    else:
        physical_run=Path(spec['output'])
    report=json.loads((physical_run/'chain_summary.json').read_text())
    protocol=scene['completion_protocol']
    checks=report['checks'] if protocol=='original_strict' else report['contact_filtered_protocol']['checks']
    complete=report['complete'] if protocol=='original_strict' else report['contact_filtered_protocol']['complete']
    result=dict(status='complete' if complete else 'revision_required',scene_id=scene['scene_id'],physics_executed=a.mode=='execute',
                evidence_mode='new_physical_execution' if a.mode=='execute' else 'saved_physical_replay',physical_run=str(physical_run),
                completion_protocol=protocol,physical_complete=complete,strict_complete=report['complete'],checks=checks,
                approach_decision=json.loads((physical_run/'connection_decision.json').read_text()) if (physical_run/'connection_decision.json').exists() else None,
                unmet_checks=[k for k,v in checks.items() if not v],stages=report['stages'],stage_times=report['stage_times'])
    # Persist authoritative physical status before requesting any model narration.
    (a.output/'physical_outcome.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    public_result={k:v for k,v in result.items() if k not in ('physical_run','approach_decision')}
    decision=result['approach_decision']
    public_result['planner']=dict(reached=decision is not None,candidate=decision['candidate'] if decision else None,status=decision.get('selection',{}).get('status','fixed') if decision else 'not_reached')
    assistant=dict(role='assistant',content=response['content'])
    if 'provider_message' in response:assistant['provider_message']=response['provider_message']
    messages.extend([assistant,dict(role='user',content=[dict(type='tool_result',tool_use_id=call['id'],content=json.dumps(public_result,ensure_ascii=False))])])
    final=call_existing_model(messages,tools,a.settings,a.output,2,allow_tools=False)
    summary='\n'.join(b['text'] for b in final['content'] if b['type']=='text')
    (a.output/'model_summary.txt').write_text(summary+'\n')
    result['model_summary_role']='Narration only; status above comes from physical checks'
    return result


started=time.monotonic()
try:
    result=run()
except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as error:
    (a.output/'status.json').write_text(json.dumps(dict(status='failed',error_type=type(error).__name__,elapsed_s=time.monotonic()-started)))
    raise
result['elapsed_s']=time.monotonic()-started
(a.output/'outcome.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(a.output/'status.json').write_text(json.dumps(dict(status='completed',task_status=result['status'],elapsed_s=result['elapsed_s'])))
print(json.dumps(result,ensure_ascii=False))
