"""Frozen context rules with exact baseline aliases and explicit missing states."""


def validate_path_evidence(actual, expected, *, atol_m=1e-6):
    """Artifact roundoff tolerance, independent of task/controller thresholds."""
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=atol_m, rtol=0)
    return float(np.max(np.abs(actual - expected)))


def capture_stop_reason(fault, domain_exit, physical_failed):
    if any(fault):
        return 'ENGINEERING_FAULT'
    if any(domain_exit) or any(physical_failed):
        return 'PHYSICAL_OR_DOMAIN_EXIT'
    return None


def capture_result_action(receipt, returncode):
    if returncode == 0 and receipt.get('phase') == 'completed':
        return 'CAPTURED'
    if receipt.get('capture_stop_reason') in ('PHYSICAL_OR_DOMAIN_EXIT', 'PHASE_NOT_REACHED'):
        return 'UNAVAILABLE'
    raise RuntimeError('capture engineering error: ' + repr(receipt))


def confirmation_cases(prefixes, contexts, forces):
    key = lambda row: (row['route'], row['speed'], row['alpha'])
    captured = {}
    for row in prefixes:
        if key(row) in captured:
            raise ValueError('duplicate captured context')
        captured[key(row)] = row
    if len({key(row) for row in contexts}) != len(contexts):
        raise ValueError('duplicate declared context')
    if set(captured) - {key(row) for row in contexts}:
        raise ValueError('undeclared captured context')
    cases, pairs = [], []
    for context in contexts:
        prefix = captured.get(key(context))
        for force in forces:
            pair = dict(route=context['route'], speed=context['speed'], alpha=context['alpha'],
                        force_N=force, baseline_index=None, lookup_index=None,
                        availability='AVAILABLE' if prefix else 'UNAVAILABLE')
            if prefix:
                common = {k: prefix[k] for k in ('state_id', 'route', 'speed', 'alpha', 'snapshot', 'snapshot_sha256')}
                pair['baseline_index'] = len(cases)
                cases.append(dict(**common, candidate='continue_original', target_delta=[0., 0.],
                                  override_mask=[False, False], force_N=force))
                pair['lookup_index'] = pair['baseline_index']
                if context['lookup'] != 'continue_original':
                    delta = list(context['target_delta'])
                    pair['lookup_index'] = len(cases)
                    cases.append(dict(**common, candidate=context['lookup'], target_delta=delta,
                                      override_mask=[value != 0 for value in delta], force_N=force))
            pairs.append(pair)
    return cases, pairs
