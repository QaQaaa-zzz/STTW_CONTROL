"""Read-only failure semantics/exposure audit; no simulation or training."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from sttw_control.preference_best import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path('runs/reserve_feature_expansion_v1')
    receipt, exposure = {}, {}
    stages = dict(uniform=[Path('runs/low_speed_recovery_baseline_v1/uniform'), source/'uniform_stage2'],
                  physics=[source/'physics_stage1', source/'physics_stage2'],
                  polynomial=[source/'polynomial_stage1', source/'polynomial_stage2'])
    for name, paths in stages.items():
        records = []
        for run in paths:
            for filename in ['metrics.jsonl', 'manifest.json', 'effective_config.json']:
                f = (run/filename).resolve()
                receipt[str(f)] = digest(f)
            records += [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        if [r['update'] for r in records] != list(range(1, 78)):
            raise ValueError('Missing/overlapping updates')
        ppo = [r for r in records if r['training_stage']=='ppo']
        if len(ppo) != 75:
            raise ValueError('PPO and value warmup differ')
        total = lambda key: sum(r[key] for r in records)
        exposure[name] = dict(records=77, critic_warmup=2, ppo_updates=75,
            physical_flag_count=total('physical_failures'), domain_flag_count=total('domain_failures'),
            completed=sum(r['pulse']['completed'] for r in records),
            fired_pulses=sum(r['pulse']['fired'] for r in records),
            accepted_minibatches=sum(r['optimizer']['accepted_minibatches'] for r in ppo),
            attempted_minibatches=sum(r['optimizer']['attempted_minibatches'] for r in ppo),
            hard_kl_stop_updates=sum(bool(r['optimizer']['hard_kl_stop']) for r in ppo),
            nonfinite_stop_updates=sum(bool(r['optimizer']['nonfinite_stop']) for r in ppo),
            flags_may_overlap=True, per_transition_advantages='NOT_SAVED_BY_STANDARD_CHECKPOINT')
    historical = []
    revision = '58724612802cc1494995425d5e265f6682af2e20'
    for name in ['teleop_env', 'path_command_env', 'path_command_reward']:
        f = Path('learning/src/sttw_control')/(name+'.py')
        blob = subprocess.check_output(['git', 'show', revision+':'+str(f)])
        if hashlib.sha256(blob).hexdigest() != digest(f):
            raise ValueError('Terminal source differs from manifest Git '+name)
        historical.append(dict(file=str(f.resolve()), revision=revision, sha256=digest(f)))
    declaration = json.loads((source/'extension_declaration.json').read_text())
    for filename in ['tolerance_training.py', 'short_recovery.py']:
        f, sha = next((f, sha) for f, sha in declaration['source_sha256'].items() if Path(f).name==filename)
        if digest(f) != sha:
            raise ValueError('Declared training source changed')
        receipt[f] = sha
    paired_file = Path('runs/midpoint_physical_check_v1/analysis/analysis.json')
    paired = json.loads(paired_file.read_text())
    receipt[str(paired_file.resolve())] = digest(paired_file)
    for f, sha in paired['raw_sha256'].items():
        if digest(f) != sha:
            raise ValueError('Matched evidence changed '+f)
        receipt[f] = sha
    cases = []
    for row in paired['cases']:
        file = Path(row['snapshot']).parent/(row['name']+'.npz')
        if str(file) not in paired['raw_sha256']:
            raise ValueError('Missing authenticated raw trace')
        with np.load(file) as data:
            active = data['active_tick'].astype(bool)
            t = data['time'][active]
            roll = data['peak_roll'][active]
            domain = data['domain_exit'][active]
            contact = data['contact_failure'][active]
            fallback = data['fallback_active'][active]
        first = lambda flag: float(t[np.flatnonzero(flag)[0]]) if np.any(flag) else None
        safe = row['balance_label']=='NO_FLOOR_CONTACT'
        cases.append(dict(model=row['model'], side=row['side'], speed=row['requested_speed'],
            fraction=row['fraction'], force_N=row['force_N'], label=row['balance_label'],
            domain_time_s=first(domain), legacy_roll_limit_time_s=first(roll>.7),
            contact_time_s=first(contact), full_no_contact=safe,
            no_contact_with_domain_handover=bool(safe and np.any(fallback)),
            no_contact_after_roll_limit=bool(safe and np.any(roll>.7)),
            scope='Reused diagnostic branch with common fallback, not a training-episode counterfactual'))
    summary = {}
    for model in sorted({r['model'] for r in cases}):
        group = [r for r in cases if r['model']==model and r['force_N']]
        summary[model] = dict(disturbed=len(group), full_no_contact=sum(r['full_no_contact'] for r in group),
            no_contact_with_domain_handover=sum(r['no_contact_with_domain_handover'] for r in group),
            no_contact_after_roll_limit=sum(r['no_contact_after_roll_limit'] for r in group))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'audit.json', dict(exposure=exposure, reused_branch_summary=summary,
        reused_cases=cases, authenticated_terminal_sources=historical, input_sha256=receipt,
        analysis_source_sha256=digest(__file__), new_physics_ticks=0, optimizer_updates=0,
        limits=['physical_flag is not contact count', 'domain/physical flags may overlap',
                'same formula does not imply same realized penalty/advantage',
                'accepted minibatches do not prove useful risk gradients',
                'diagnostic fallback survival is not current-actor/task recovery']))
    print(json.dumps(dict(exposure=exposure, reused=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()
