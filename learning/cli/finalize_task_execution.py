"""Confirm task progress or replay recorded E5 execution without new physics."""
import argparse,json
from pathlib import Path
from sttw_control.continuous_task import ContinuousTask
from sttw_control.task_execution import confirm_task_execution

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path)
p.add_argument('--manifest',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--require-platform',action='store_true')
p.add_argument('--snapshot-output',type=Path,help='Export portable recorded execution and physical-summary pairs')
a=p.parse_args()
if (a.run is None)==(a.manifest is None):p.error('Supply exactly one run or recorded manifest')
if a.run:
    result=confirm_task_execution(json.loads((a.run/'task_execution.json').read_text()),json.loads((a.run/'chain_summary.json').read_text()),require_platform=a.require_platform)
else:
    rows=[]
    for case in json.loads(a.manifest.read_text())['cases']:
        if case.get('method')!='capability':continue
        spec=json.loads(Path(case['config']).read_text());run=Path(spec['output']);planner=json.loads(Path(spec['forward_branch']['planner_config']).read_text());branch_file=run/'forward_branch_decision.json';branch=json.loads(branch_file.read_text()) if branch_file.exists() else None;report=json.loads((run/'chain_summary.json').read_text())
        manager=ContinuousTask(planner['program'],planner['routes'],planner['candidates'],None)
        initial=next(c['candidate_id'] for c in planner['candidates'] if abs(c['speed_m_s']-spec['approach_speed'])<1e-6 and abs(c['alpha']-spec['ground_alpha'])<1e-6)
        manager.record_execution('original_jump',initial,time_s=0.,reason='Recorded upstream executable reference')
        choice=branch.get('selection') if branch else None
        if choice:
            manager.events.append(choice)
            if choice['status']=='reference_selected':manager.active=choice['reference'].copy()
        if branch is not None:
            cid=choice['reference']['candidate_id'] if choice and choice['status']=='reference_selected' and choice['reference']['route_id']==branch['executed_route'] else initial
            manager.record_execution(branch['executed_route'],cid,time_s=branch['time_s'],reason='Recorded actual route after availability/deadline checks')
        goal_id=manager.active_reference['goal_id']
        snapshot=dict(program=planner['program'],routes=planner['routes'],candidates=planner['candidates'],events=manager.events,outcome_goal_id=goal_id)
        if a.snapshot_output:
            folder=a.snapshot_output/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
            protocol=json.loads((run/'protocol.json').read_text());ground=json.loads((run/'ground_controller.json').read_text())
            snapshot['source_identity']=dict(initial_reference_provenance='Reconstructed from executed run config approach speed/alpha, not a historical initial execution event',source_run=str(run),source_config=case['config'],ground_sha256=ground['upper_sha256'],jump_sha256=[p['artifact_sha256'] for p in protocol['policies']],physics_dt=protocol['physics_dt'])
            (folder/'task_execution.json').write_text(json.dumps(snapshot,indent=2)+'\n')
            (folder/'chain_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        result=confirm_task_execution(snapshot,report,require_platform=a.require_platform)
        expected=report['forward_task']['mandatory_platform_task_complete' if a.require_platform else 'optional_platform_task_complete']
        rows.append(dict(case_id=case['case_id'],source_run=str(run),recorded_complete=expected,agrees_with_recorded=bool(expected==result['accepted_goal']),**result))
    result=dict(role='Recorded capability-run replay, not live task management or new physics',cases=len(rows),complete=sum(r['status']=='complete' for r in rows),revision_required=sum(r['status']=='revision_required' for r in rows),mismatches=sum(not r['agrees_with_recorded'] for r in rows),rows=rows)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','events')},indent=2))
