"""Freeze E5 evidence and pair rule/capability execution on identical route roots."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--records', type=Path, required=True)
p.add_argument('--rule-config', type=Path, required=True)
p.add_argument('--kernel-config', type=Path, default=Path('learning/configs/jump_connection_capability_dev.json'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = json.loads(a.manifest.read_text())
kernel = json.loads(a.kernel_config.read_text())
rule = json.loads(a.rule_config.read_text())
records = [json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
if rule['mode'] != 'rule' or not records:
    raise ValueError('An existing rule and nonempty observed route evidence are required')
a.output.mkdir(parents=True, exist_ok=False)
inputs = a.output / 'input'
inputs.mkdir()
evidence = inputs / 'records.jsonl'
shutil.copy2(a.records, evidence)
rule_path = inputs / 'rule.json'
shutil.copy2(a.rule_config, rule_path)
cases = []
budget = 0
for case in source['cases']:
    spec = json.loads(Path(case['config']).read_text())
    if spec['forward_branch']['requested_route'] != 'original_jump':
        continue  # Each forced original/bypass pair shares one physical prefix.
    contract_path = Path(spec['forward_branch']['map_contract'])
    contract = json.loads(contract_path.read_text())
    candidate = {k: spec['connection_candidate'][k] for k in ('speed_m_s', 'alpha')}
    candidate['candidate_id'] = f"v{candidate['speed_m_s']}_a{candidate['alpha']}"
    planner = {k: kernel[k] for k in ('feature_scales', 'bandwidth', 'minimum_support',
        'minimum_completion', 'probability_tolerance', 'time_tolerance_s')}
    planner.update(mode='capability', record_files=[str(evidence.resolve())],
        map_id=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        program=dict(goals=[dict(goal_id='destination', allowed_routes=['original_jump', 'bypass'])]),
        candidates=[candidate], routes=[dict(route_id=route, goal_id='destination',
            task_id=route + ':optional', skill_id='matched_runtime',
            candidates=[candidate['candidate_id']], reference_path=reference)
            for route, reference in [('original_jump', spec['ground_route']),
                                     ('bypass', contract['branch_reference'])]],
        role='Leave entire queried origin out; optional-platform primary task; mandatory-platform scored separately')
    planner_path = inputs / (case['case_id'] + '_capability.json')
    planner_path.write_text(json.dumps(planner, indent=2) + '\n')
    for method, config in [('rule', rule_path), ('capability', planner_path)]:
        paired = copy.deepcopy(spec)
        ident = case['case_id'] + '_' + method
        paired.update(output=str((a.output / ident).resolve()),
            role='Matched online E5 DEV; frozen evidence, shared map/timing/reference, no RL')
        paired['forward_branch'].update(planner_config=str(config.resolve()),
            asynchronous=True, query_deadline_wall_s=.5)
        path = a.output / (ident + '.json')
        path.write_text(json.dumps(paired, indent=2) + '\n')
        cases.append(dict(case, case_id=ident, method=method, config=str(path.resolve())))
        budget += round(paired['seconds'] / paired['physics_dt'])
manifest = dict(role='Paired online rule/capability DEV, not oracle or independent TEST',
    simulation_python=source['simulation_python'], analysis_python=source['analysis_python'],
    cases=cases, maximum_physics_steps=budget, optimizer_updates=0, status='prepared_not_started',
    evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
    source_manifest=str(a.manifest.resolve()),
    limits='Both methods share disturbance realization and decision trigger. Whole-origin exclusion occurs in ForwardRouteSelector. Optional-platform completion and mandatory-platform safe exit remain distinct. No threshold fitting or outcome-based root replacement.')
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(cases=len(cases), maximum_physics_steps=budget)))
