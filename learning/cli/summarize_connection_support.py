"""Summarize executed Q_hold data and whole-origin-excluded kernel support."""
import argparse
import json
from pathlib import Path
from sttw_control.jump_connection import JumpConnectionSelector

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--records',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
config=json.loads(a.config.read_text());origins=sorted({r['origin_id'] for r in rows})
forecasts=[];groups=[]
for origin in origins:
    group=[r for r in rows if r['origin_id']==origin]
    groups.append(dict(origin=origin,branches=len(group),decision_reached=sum(r['decision_reached'] for r in group),strict=sum(r['strict_Y'] for r in group),complete=sum(r['Y'] for r in group),outcomes=[dict(candidate=r['candidate'],strict=r['strict_Y'],complete=r['Y'],decision_reached=r['decision_reached']) for r in group]))
    for row in group:
        if not row['supervised_input_complete'] or (row['decision_prefix'] or {}).get('corridor_intact_at_decision') is not True:continue
        selector=JumpConnectionSelector(rows,config,origin_id=origin,policy_sha256=row['policy_sha256'],model_contract_sha256=row['model_contract_sha256'],physics_dt=row['physics_dt'],zero_lower=row['zero_lower'])
        query=selector.choose(row['state'],'exit',row['candidate'])
        cid=f"v{row['candidate']['speed_m_s']}_a{row['candidate']['alpha']}"
        estimate=next(e for e in query['estimates'] if e['candidate_id']==cid)
        error=(estimate['complete_probability']-row['Y'])**2 if estimate['known'] else None
        forecasts.append(dict(case_id=row['case_id'],origin=origin,actual_Y=row['Y'],actual_candidate=row['candidate'],known=estimate['known'],support=estimate['support'],predicted_p=estimate['complete_probability'],squared_error=error,selection=query))
known=[r for r in forecasts if r['known']]
summary=dict(role='Whole-origin-excluded predictions on each actual executed state; no online selection effect inferred',label_protocols=sorted({r['label_protocol'] for r in rows}),roots=len(origins),branches=len(rows),strict=sum(r['strict_Y'] for r in rows),complete=sum(r['Y'] for r in rows),eligible_observed_states=len(forecasts),known_actual_candidate=len(known),conditional_brier=sum(r['squared_error'] for r in known)/len(known) if known else None,limits='Actual branch states may differ within an origin; this does not assert a common simulator snapshot. Upstream failures remain in the full denominator. DEV, uncalibrated probabilities, no model fitting.',per_origin=groups,predictions=forecasts)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('per_origin','predictions')},indent=2))
