"""Read-only physical-tail target audit; historical actions are NOT PPO samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from sttw_control.balance_probe import classify_contact_balance
from sttw_control.survival_credit import resolve_survival_credit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    (out/'cases').mkdir()
    source = root/'runs/midpoint_physical_check_v1/analysis/analysis.json'
    original = json.loads(source.read_text())
    receipts = {str(source): sha(source)}
    for path, expected in original['raw_sha256'].items():
        actual = sha(path)
        if actual != expected:
            raise ValueError('physical source changed: '+path)
        receipts[path] = actual
    records = []
    for i, case in enumerate(original['cases']):
        campaign = Path(case['snapshot']).parent
        prep_file = campaign/'prepare_plan.json'
        prep = json.loads(prep_file.read_text())
        assert prep['observation_s'] == 4.
        actor_id = prep['model']['actor_sha256']
        assert sha(prep['model']['actor']) == actor_id
        raw = campaign/(case['name']+'.npz')
        assert str(raw) in original['raw_sha256'] and sha(raw) == original['raw_sha256'][str(raw)]
        with np.load(raw, allow_pickle=False) as z:
            trace = {k: z[k] for k in z.files}
        label = classify_contact_balance(trace, horizon_s=4.)
        assert label['balance_label'] == case['balance_label']
        active = trace['active_tick'].astype(bool)
        upper = trace['upper_active'].astype(bool)
        assert not np.any(upper & ~active)
        # Logs retain original logical groups of four lower ticks per sampled
        # upper action. Gaps at handover must not shift these boundaries.
        action = upper & (np.arange(len(active)) % 4 == 0)
        for start in range(0, len(active), 4):
            use = upper[start:start+4]
            if use.any():
                assert use[0]
                values = trace['original_policy_latent'][start:start+4][use]
                np.testing.assert_array_equal(values, np.repeat(values[:1], len(values), axis=0))
        ids = np.flatnonzero(active)
        # Time is logged before each real .005s transition. Quantize only its
        # float representation; retain/check every observed tick and all gaps.
        clock = np.rint(trace['time']/.005).astype(np.int64)
        np.testing.assert_allclose(trace['time'][active], clock[active]*.005, atol=2e-6, rtol=0.)
        horizon = np.zeros_like(active)
        if ids.size == 800:
            horizon[ids[-1]] = True
        fault = trace['nonfinite'] | trace['lower_fault'] | trace['policy_fault']
        credit = resolve_survival_credit(active_tick=active, actor_action_valid=action,
            contact_failure=trace['contact_failure'].astype(bool),
            engineering_fault=fault.astype(bool), horizon_reached=horizon,
            physical_tick=clock, window_start_tick=0, window_ticks=800,
            collection_policy_id=actor_id, learner_policy_id=actor_id)
        expected = {'NO_FLOOR_CONTACT': 'survived', 'FLOOR_CONTACT': 'contact'}[case['balance_label']]
        assert credit['outcome'] == expected
        assert not np.any(credit['actor_mask'] & trace['fallback_active'])
        assert credit['physical_ticks'] == int(active.sum()) == case['active_ticks']
        assert credit['actor_actions'] == int(action.sum())
        assert credit['actor_actions'] <= (credit['physical_ticks']+3)//4
        handover = bool(case['fallback_triggered'])
        np.savez_compressed(out/'cases'/f'case_{i:03d}.npz', targets=credit['targets'],
            actor_mask=credit['actor_mask'], actor_indices=credit['actor_indices'],
            physical_active=active, hold_active=trace['fallback_active'],
            physical_tick=clock,
            time=trace['time'], contact_failure=trace['contact_failure'])
        records.append(dict(case_index=i, source_trace=str(raw), model=case['model'],
            state_id=case['state_id'], force_N=case['force_N'],
            outcome=credit['outcome'], target=credit['target'],
            actor_actions=credit['actor_actions'], physical_ticks=credit['physical_ticks'],
            hold_ticks=int(np.sum(trace['fallback_active'] & active)),
            domain_handover=handover, collection_policy_id=actor_id,
            historical_deterministic_data=True, eligible_for_PPO_update=False))
    assert len(records) == 96
    survivors = [x for x in records if x['force_N'] != 0 and x['outcome'] == 'survived']
    assert len(survivors) == 13 and all(x['domain_handover'] and x['actor_actions'] > 0 for x in survivors)
    summary = dict(cases=records, outcome_counts=dict(Counter(x['outcome'] for x in records)),
        actor_actions=sum(x['actor_actions'] for x in records),
        physical_ticks=sum(x['physical_ticks'] for x in records),
        hold_ticks=sum(x['hold_ticks'] for x in records),
        disturbed_domain_hold_survivors=len(survivors),
        input_sha256=receipts, source_sha256={str(Path(__file__).resolve()): sha(__file__),
            str(root/'learning/src/sttw_control/survival_credit.py'): sha(root/'learning/src/sttw_control/survival_credit.py')},
        new_physics_ticks=0, optimizer_updates=0,
        scope='Observed 4s suffix targets only. Historical deterministic branches are off-policy diagnostics, '
              'not fresh PPO samples or proof of full-prefix collection/GAE/optimizer integration.')
    (out/'receipt.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['cases', 'input_sha256', 'source_sha256']}))


if __name__ == '__main__':
    main()
