"""Matched whole-origin-held-out DEV ablation using executed candidate outcomes."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
from sttw_control.jump_connection import JumpConnectionSelector
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--records',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--distance-rule-manifest',type=Path,help='Earlier nominal pilot only; build fastest successful candidate per declared distance')
a=p.parse_args();records=[json.loads(s) for s in a.records.read_text().splitlines() if s.strip()];cfg=json.loads(a.config.read_text())
records=[r for r in records if r['decision_stage']=='exit' and r['target_semantics']=='Q_hold']
groups=defaultdict(list)
for r in records:
 spec=json.loads(Path(r['source_config']).read_text())
 groups[(r['origin_id'],spec['handoff_x']-spec['connection_candidate']['decision_x_m'])].append(r)
rows=[];anchor=dict(speed_m_s=2.,alpha=.5)
methods=('fixed_2.0_0.5','fixed_2.0_0.0','distance_only_kernel','full_state_kernel','exact_distance_full_state_kernel','exact_distance_only_kernel')
distance_by_config={r['source_config']:distance for (_,distance),rs in groups.items() for r in rs}
rule={};rule_sources=[]
if a.distance_rule_manifest:
 for case in json.loads(a.distance_rule_manifest.read_text())['cases']:
  spec=json.loads(Path(case['config']).read_text())
  if spec['origin_id'] in {r['origin_id'] for r in records}:raise ValueError('Rule-building origins must be separate from evaluated roots')
  result=json.loads((Path(spec['output'])/'chain_summary.json').read_text())
  distance=spec['handoff_x']-spec['connection_candidate']['decision_x_m'];rule.setdefault(distance,None)
  rule_sources.append(dict(case_id=case['case_id'],source_config=case['config'],origin_id=spec['origin_id']))
  if result['complete'] and (rule[distance] is None or result['success_time_s']<rule[distance]['T_s']):
   rule[distance]=dict(candidate=dict(speed_m_s=case['speed_m_s'],alpha=case['alpha']),T_s=result['success_time_s'],case_id=case['case_id'])
 methods=methods+('nominal_distance_rule',)
for (origin,distance),group in sorted(groups.items()):
 by_candidate={(r['candidate']['speed_m_s'],r['candidate']['alpha']):r for r in group}
 if len(by_candidate)!=6:continue
 reference=by_candidate[(2.,.5)]
 ready=reference['supervised_input_complete'] and (reference['decision_prefix'] or {}).get('corridor_intact_at_decision') is True
 outcomes={}
 for method in methods:
  choice=dict(speed_m_s=2.,alpha=0.) if method=='fixed_2.0_0.0' else anchor;status='fixed';query=None
  if method=='nominal_distance_rule':
   status='upstream_unavailable'
   if ready:
    selected=rule.get(distance);choice=selected['candidate'] if selected else anchor;status='distance_rule' if selected else 'no_feasible'
  elif not method.startswith('fixed_'):
   status='upstream_unavailable'
   if ready:
    subset=[12] if method in ('distance_only_kernel','exact_distance_only_kernel') else list(range(len(reference['state'])))
    data=[dict(r,state=[r['state'][i] for i in subset] if r['state'] is not None else None) for r in records]
    if method.startswith('exact_distance_'):data=[r for r in data if distance_by_config[r['source_config']]==distance]
    config=dict(cfg,feature_scales=[cfg['feature_scales'][i] for i in subset])
    selector=JumpConnectionSelector(data,config,origin_id=origin,policy_sha256=reference['policy_sha256'],model_contract_sha256=reference['model_contract_sha256'],physics_dt=reference['physics_dt'],zero_lower=reference['zero_lower'])
    query=selector.choose([reference['state'][i] for i in subset],'exit',anchor);choice=query['candidate'];status=query['status']
  executed=by_candidate[(choice['speed_m_s'],choice['alpha'])]
  outcomes[method]=dict(status=status,candidate=choice,Y=executed['Y'],T_s=executed['T_total_s'],case_id=executed['case_id'],query=query)
 rows.append(dict(origin_id=origin,remaining_release_distance_m=distance,decision_ready=ready,methods=outcomes))
scores={}
for method in methods:
 values=[r['methods'][method] for r in rows];common=[r for r in rows if r['methods'][method]['Y'] and r['methods']['fixed_2.0_0.5']['Y']]
 scores[method]=dict(successes=sum(v['Y'] for v in values),task_groups=len(rows),statuses=dict(Counter(v['status'] for v in values)),
  released=sum(v['status'] in ('selected','distance_rule') for v in values),false_release=sum(v['status'] in ('selected','distance_rule') and not v['Y'] for v in values),
  common_success_with_anchor=len(common),mean_time_saved_against_anchor_s=sum(r['methods']['fixed_2.0_0.5']['T_s']-r['methods'][method]['T_s'] for r in common)/len(common) if common else None)
result=dict(distance_rule=rule,distance_rule_sources=rule_sources,role='Offline paired DEV ablation, not online execution or independent TEST',scores=scores,groups=rows,
 feature_ablation='Distance-only uses remaining path to release, full uses frozen 23 observable features; same kernel, support and choice thresholds; exact-distance variant restricts evidence to the same declared geometry; no tuning on held-out origin',
 fallback='Unknown/no_feasible retains initial 2.0/.5; scored from its actually executed paired outcome. Upstream failures remain in full-route denominator.',
 holdout='All descendants of query origin excluded, including other distances. Distances are clustered within origin.',
 limits='Deterministic D0 data only; empirical scores are not calibrated probabilities. Selecting stored outcomes does not substitute physical online validation.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(scores,indent=2))
