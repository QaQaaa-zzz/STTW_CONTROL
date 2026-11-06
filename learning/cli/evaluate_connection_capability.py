"""Evaluate executed paired connection outcomes, holding out whole origin runs."""
import argparse,json
from pathlib import Path
from sttw_control.empirical_capability import evaluate_connection_groups
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,help='Known-map task program and current observable state; emit a numerical execution recommendation');p.add_argument('--records',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());rows=[json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
if a.request:
    from sttw_control.empirical_capability import ConnectionCapability
    from sttw_control.continuous_task import ContinuousTask,available_forward_routes
    request=json.loads(a.request.read_text())
    excluded=request.get('exclude_origin_id')
    evidence=[r for r in rows if excluded is None or r['origin_id']!=excluded]
    capability=ConnectionCapability(evidence,c['feature_scales'],c['bandwidth'],c['minimum_support'])
    manager=ContinuousTask(request['program'],request['routes'],request['candidates'],capability)
    availability=available_forward_routes(request['routes'],request['available_routes'],**request['execution_state']) if 'execution_state' in request else dict(available_routes=request['available_routes'],blocked_routes=[])
    result=manager.choose(request['state'],availability['available_routes'],
        request.get('minimum_completion',.7),request.get('probability_tolerance',.05))
    result['route_availability']=availability
    result.update(role='Numerical task recommendation only; no physical execution or goal acknowledgement',
                  evidence_rows=len(evidence),excluded_origin_id=excluded,completed=manager.completed,
                  request=str(a.request.resolve()))
else:
    result=evaluate_connection_groups(rows,c['candidates'],c['feature_scales'],c['bandwidth'],c['minimum_support'],c.get('minimum_completion',.7),c.get('probability_tolerance',.05))
result['inputs']=dict(records=str(a.records.resolve()),config=str(a.config.resolve()))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
