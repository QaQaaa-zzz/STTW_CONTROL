import json
from pathlib import Path

from sttw_control.jump_connection import JumpConnectionSelector


def test_hold_lookup_excludes_query_origin_and_incompatible_physics():
    config = json.loads(Path('learning/configs/jump_connection_capability_dev.json').read_text())
    def row(origin, candidate, success, cost, **changes):
        return dict(origin_id=origin, target_semantics='Q_hold', supervised_input_complete=True,
            policy_sha256={'ground': 'g', 'jump': 'j'}, model_contract_sha256='model',
            physics_dt=.0025, zero_lower=True, decision_prefix={'corridor_intact_at_decision': True},
            state=[0.] * 23, decision_stage='midpoint', candidate=candidate, Y=success,
            T_after_decision_s=cost, **changes)
    slow, fast = config['candidates'][0], config['candidates'][3]
    evidence = [row('held_out', fast, 1, .1), row('train', slow, 1, 2.), row('train', fast, 0, None)]
    bad = row('other', fast, 1, .1)
    bad['physics_dt'] = .005
    evidence.append(bad)
    selector = JumpConnectionSelector(evidence, config, origin_id='held_out',
        policy_sha256={'ground': 'g', 'jump': 'j'}, model_contract_sha256='model',
        physics_dt=.0025, zero_lower=True)
    result = selector.choose([0.] * 23, 'midpoint', fast)
    assert result['evidence_rows'] == 2
    assert result['status'] == 'selected'
    assert result['candidate'] == slow
    # Midpoint outcomes cannot provide exit-stage support.
    result = selector.choose([0.] * 23, 'exit', fast)
    assert result['status'] == 'unknown'
    assert result['candidate'] == fast
    assert result['fallback'] == 'retain_current_reference'


def test_no_support_retains_actual_request_without_claiming_feasibility():
    config = json.loads(Path('learning/configs/jump_connection_capability_dev.json').read_text())
    selector = JumpConnectionSelector([], config, origin_id='test', policy_sha256={},
        model_contract_sha256='model', physics_dt=.0025, zero_lower=True)
    current = {'speed_m_s': 1.95, 'alpha': .75}
    result = selector.choose([0.] * 23, 'midpoint', current)
    assert result['status'] == 'unknown'
    assert result['candidate'] == current
