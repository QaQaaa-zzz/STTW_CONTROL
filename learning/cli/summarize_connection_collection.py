"""Summarize a declared root matrix without treating pending roots as failures."""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--group', default='nominal')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
groups, roots, records = [], [], []
for case in manifest[a.group]:
    folder = Path(case['output'])
    spec = json.loads(Path(case['config']).read_text())
    candidates = [c['candidate_id'] for c in spec['candidates']]
    status_file = folder / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {'status': 'pending'}
    record_file = folder / 'records.jsonl'
    rows = [json.loads(line) for line in record_file.read_text().splitlines() if line.strip()] if record_file.exists() else []
    records.extend(rows)
    roots.append(dict(origin_id=case['origin_id'], output=str(folder), status=status,
        recorded_suffixes=len(rows)))
    for task in spec['tasks']:
        selected = [r for r in rows if r['task_id'] == task['task_id']]
        by_id = {r['candidate_id']: r for r in selected}
        if len(by_id) != len(selected) or set(by_id) - set(candidates):
            raise ValueError(f'Duplicate or undeclared candidates in {folder}/{task["task_id"]}')
        observed = len(by_id) == len(candidates)
        successes = [c for c in candidates if c in by_id and by_id[c]['complete']]
        state = ('all_success' if len(successes) == len(candidates) else 'partial_success' if successes else 'all_failed') if observed else ('source_failed' if status['status'] == 'source_failed' else 'incomplete')
        groups.append(dict(origin_id=case['origin_id'], case_id=case.get('case_id', case['origin_id']), output=str(folder),
            task_id=task['task_id'], classification=state, candidates=candidates,
            success_vector=[by_id[c]['complete'] if c in by_id else None for c in candidates],
            success_costs=[by_id[c]['success_cost'] if c in by_id else None for c in candidates],
            state_named=selected[0].get('state_named') if selected else None,
            successful_candidates=successes,
            original_path_max_m=[by_id[c]['max_original_path_error_m'] if c in by_id else None for c in candidates],
            b_entries=[by_id[c].get('b_entry') if c in by_id else None for c in candidates],
            input_target_limits=[by_id[c].get('input_target_limit') if c in by_id else None for c in candidates]))

complete = [g for g in groups if g['classification'] in ('all_success', 'partial_success', 'all_failed')]
fixed, pairs = [], []
if complete:
    ids = complete[0]['candidates']
    if any(g['candidates'] != ids for g in complete):
        raise ValueError('Fixed-candidate comparison requires the same candidate ordering')
    for i, candidate in enumerate(ids):
        costs = [g['success_costs'][i] for g in complete if g['success_vector'][i]]
        fixed.append(dict(candidate_id=candidate, successes=len(costs), evaluated_groups=len(complete),
            mean_cost_on_own_successes=sum(costs)/len(costs) if costs else None))
    for i, j in itertools.combinations(range(len(ids)), 2):
        common = [g for g in complete if g['success_vector'][i] and g['success_vector'][j]]
        pairs.append(dict(first=ids[i], second=ids[j], common_success_groups=len(common),
            mean_first_minus_second_s=sum(g['success_costs'][i]-g['success_costs'][j] for g in common)/len(common) if common else None,
            only_first=[dict(origin_id=g['origin_id'], case_id=g['case_id'], task_id=g['task_id']) for g in complete if g['success_vector'][i] and not g['success_vector'][j]],
            only_second=[dict(origin_id=g['origin_id'], case_id=g['case_id'], task_id=g['task_id']) for g in complete if g['success_vector'][j] and not g['success_vector'][i]]))
summary = dict(role=manifest['role'], manifest=str(a.manifest.resolve()),
    manifest_group=a.group,
    declared_cases=len(roots), distinct_origins=len({r['origin_id'] for r in roots}),
    root_status_counts=dict(Counter(r['status']['status'] for r in roots)),
    group_counts=dict(Counter(g['classification'] for g in groups)), roots=roots, groups=groups,
    fixed_candidates=fixed, paired_comparisons=pairs,
    oracle_success_groups=sum(bool(g['successful_candidates']) for g in complete),
    limits='Descriptive DEV matrix; repeated roots are clustered, not independent trials. Source failures remain in root accounting and have no invented suffix labels. Pending entries are null. Own-success mean costs are not matched method comparisons; use the pairwise common-success costs. No online selector result.')
a.output.mkdir(parents=True, exist_ok=True)
(a.output/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(a.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
print(json.dumps({k: summary[k] for k in ('declared_cases','distinct_origins','root_status_counts','group_counts','oracle_success_groups')}, indent=2))
