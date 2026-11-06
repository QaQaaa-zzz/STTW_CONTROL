"""Descriptive finite-grid physical outcomes, never continuous safety bounds."""
import math


def directional_grid_summary(magnitudes, outcomes):
    """Summarize one signed direction, given actual zero and increasing queries.

    Unknowns interrupt evidence, and a larger successful force never fills an
    earlier failure. A prefix consists solely of the supplied measured points.
    It does not assert that intervening untested amplitudes are safe.
    """
    f=list(magnitudes); y=list(outcomes)
    if (not f or len(f)!=len(y) or f[0]!=0 or
            not all(math.isfinite(v) for v in f) or
            any(b<=a for a,b in zip(f,f[1:])) or
            not set(y)<={'survived','contact','unknown'}):
        raise ValueError('zero followed by strictly increasing finite magnitudes and physical outcomes required')
    stop=next((i for i,v in enumerate(y) if v!='survived'),len(y))
    passes=[v for v,o in zip(f,y) if o=='survived']
    contacts=[i for i,o in enumerate(y) if o=='contact']
    return dict(zero_force_survived=y[0]=='survived',
        all_pass_grid_prefix_N=f[stop-1] if stop else None,
        first_nonpass_N=f[stop] if stop<len(y) else None,
        first_nonpass_outcome=y[stop] if stop<len(y) else None,
        largest_observed_pass_N=max(passes) if passes else None,
        first_observed_contact_N=f[contacts[0]] if contacts else None,
        right_censored=stop==len(y),
        nonmonotone_observed=any(o=='survived' for i in contacts for o in y[i+1:]),
        measured_count=len(f),
        scope='Supplied measured force grid only; no interpolation or continuous safety certification.')


class ConnectionCapability:
    """Frozen empirical connection outcomes; Gaussian support is not confidence.

    Each row is one actually executed complete connection plus next task.
    Task IDs include direction/geometry and skill IDs identify exact policies.
    Feature scales and bandwidth are supplied from the declared development
    protocol; no refitting on a query or on held-out outcomes.
    """
    def __init__(self, records, feature_scales, bandwidth, minimum_support):
        import numpy as np
        self.np = np
        self.records = list(records)
        self.scales = np.asarray(feature_scales, dtype=float)
        self.bandwidth = float(bandwidth)
        self.minimum_support = float(minimum_support)
        if (self.scales.ndim != 1 or not np.isfinite(self.scales).all() or
                (self.scales <= 0).any() or not math.isfinite(self.bandwidth) or
                self.bandwidth <= 0 or not math.isfinite(self.minimum_support) or
                self.minimum_support <= 0):
            raise ValueError('Positive finite feature scales, bandwidth and support threshold required')
        for row in self.records:
            z = np.asarray(row['state'], dtype=float)
            if z.shape != self.scales.shape or not np.isfinite(z).all() or row['complete'] not in (0, 1):
                raise ValueError('Each outcome needs a finite state and binary complete-task label')
            if row['complete'] and not math.isfinite(row['success_cost']):
                raise ValueError('Successful outcomes require a finite cost')

    def query(self, state, task_id, skill_id, candidates):
        """Rank supported candidates by completion first, successful cost second.

        No success samples means success_cost=None, not zero. Failed-row costs
        never enter that conditional estimate. Unsupported candidates are not
        selected, even when a distant example happened to succeed.
        """
        np = self.np
        state = np.asarray(state, dtype=float)
        if state.shape != self.scales.shape or not np.isfinite(state).all():
            raise ValueError('Query state must match the frozen feature schema')
        estimates = []
        for candidate in candidates:
            rows = [r for r in self.records if r['task_id'] == task_id and
                    r['skill_id'] == skill_id and r['candidate_id'] == candidate]
            weights = np.asarray([math.exp(-.5 * float(np.square(
                (np.asarray(r['state'])-state)/self.scales).sum()) / self.bandwidth**2) for r in rows])
            support = float(weights.sum())
            known = support >= self.minimum_support
            success_weight = sum(w for w,r in zip(weights,rows) if r['complete'])
            probability = float(success_weight/support) if known else None
            cost = (float(sum(w*r['success_cost'] for w,r in zip(weights,rows) if r['complete'])/success_weight)
                    if known and success_weight > 0 else None)
            estimates.append(dict(candidate_id=candidate,known=known,support=support,
                                  complete_probability=probability,success_cost=cost,sample_count=len(rows)))
        ranked = sorted((e for e in estimates if e['known']), key=lambda e:(
            -e['complete_probability'],float('inf') if e['success_cost'] is None else e['success_cost']))
        return dict(estimates=estimates,ranked_candidates=[e['candidate_id'] for e in ranked],
                    best_candidate=ranked[0]['candidate_id'] if ranked else None,
                    scope='Empirical complete-task outcomes; executor must apply feasibility threshold and fallback')
