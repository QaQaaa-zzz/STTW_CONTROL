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


def select_connection_estimate(estimates, minimum_completion=.7, probability_tolerance=.05, *, previous_candidate=None, previous_route=None, time_tolerance=0.):
    """Shared executor/evaluation decision: feasible completion, then efficiency."""
    feasible=[e for e in estimates if e['known'] and
              e['complete_probability']>=minimum_completion and e['success_cost'] is not None]
    if not feasible:return None
    best=max(e['complete_probability'] for e in feasible)
    comparable=[e for e in feasible if e['complete_probability']>=best-probability_tolerance]
    fastest=min(comparable,key=lambda e:e['success_cost'])
    previous=next((e for e in comparable if e['candidate_id']==previous_candidate and
                   (previous_route is None or e.get('route_id')==previous_route) and
                   e['success_cost']<=fastest['success_cost']+time_tolerance),None)
    return previous if previous is not None else fastest


def connection_selection_status(estimates, selected):
    if selected is not None:return 'selected'
    # Partial support cannot establish that every available candidate is infeasible.
    return 'no_feasible' if estimates and all(e['known'] for e in estimates) else 'unknown'


def evaluate_connection_groups(records, candidates, feature_scales, bandwidth, minimum_support, minimum_completion=.7, probability_tolerance=.05):
    """Leave-one-origin-run-out DEV evaluation of complete paired candidate sets.

    origin_id groups every exit/branch from one upstream physical run; snapshot_id
    identifies the actual saved closed-loop state. Uninterrupted jump records instead
    supply decision_id; this does not claim a replayable PhysX snapshot. Neither
    identifies an RL seed.
    This consumes executed outcomes only. Missing candidate results remain missing.
    """
    from collections import defaultdict
    rows=list(records);groups=defaultdict(list)
    for row in rows:
        groups[(row['origin_id'],row.get('snapshot_id',row.get('decision_id')),row['task_id'],row['skill_id'])].append(row)
    folds={origin:ConnectionCapability([r for r in rows if r['origin_id']!=origin],feature_scales,bandwidth,minimum_support)
           for origin in {r['origin_id'] for r in rows}}
    results=[]
    for (origin,snapshot,task,skill),paired in sorted(groups.items()):
        actual={r['candidate_id']:r for r in paired}
        if len(actual)!=len(paired) or set(actual)!=set(candidates):
            raise ValueError('Each snapshot/task/skill requires exactly one executed outcome for every candidate')
        state=paired[0]['state']
        if any(r['state']!=state for r in paired):
            raise ValueError('Paired candidates must share the saved observable exit state')
        query=folds[origin].query(state,task,skill,candidates)
        choice=select_connection_estimate(query['estimates'],minimum_completion,probability_tolerance)
        selected=choice['candidate_id'] if choice else None
        observed=actual[selected] if selected is not None else None
        successful=[r for r in paired if r['complete']]
        oracle=min(successful,key=lambda r:r['success_cost']) if successful else None
        calibration=[dict(candidate_id=e['candidate_id'],probability=e['complete_probability'],actual_complete=actual[e['candidate_id']]['complete'],squared_error=(e['complete_probability']-actual[e['candidate_id']]['complete'])**2)
                     for e in query['estimates'] if e['known']]
        results.append(dict(origin_id=origin,snapshot_id=snapshot if 'snapshot_id' in paired[0] else None,decision_id=snapshot if 'decision_id' in paired[0] else None,task_id=task,skill_id=skill,
            selection_status=connection_selection_status(query['estimates'],choice),
            selected_candidate=selected,selected_complete=observed['complete'] if observed else None,
            selected_success_cost=observed['success_cost'] if observed and observed['complete'] else None,
            observed_oracle_candidate=oracle['candidate_id'] if oracle else None,
            observed_oracle_success_cost=oracle['success_cost'] if oracle else None,
            predictions=query['estimates'],calibration=calibration))
    decided=[r for r in results if r['selected_candidate'] is not None]
    errors=[e['squared_error'] for r in results for e in r['calibration']]
    # Fixed references use the same executed groups, including failures. Unknown
    # model queries remain abstentions; do not turn them into simulated outcomes.
    fixed=[]
    decided_keys={(r['origin_id'],r['snapshot_id'] or r['decision_id'],r['task_id'],r['skill_id']) for r in decided}
    for candidate in candidates:
        outcomes=[r for r in rows if r['candidate_id']==candidate]
        matched=[r for r in outcomes if (r['origin_id'],r.get('snapshot_id',r.get('decision_id')),r['task_id'],r['skill_id']) in decided_keys]
        successful=[r['success_cost'] for r in outcomes if r['complete']]
        fixed.append(dict(candidate_id=candidate,groups=len(outcomes),
            complete_count=sum(r['complete'] for r in outcomes),
            successful_mean_cost=sum(successful)/len(successful) if successful else None,
            matched_decided_groups=len(matched),matched_decided_complete_count=sum(r['complete'] for r in matched)))
    return dict(fixed_candidates=fixed,
        observed_oracle_complete_groups=sum(r['observed_oracle_candidate'] is not None for r in results),
        role='leave-one-origin-run-out DEV; not independent TEST or deployed planning success',
        groups=len(results),selected_groups=len(decided),unknown_groups=sum(r['selection_status']=='unknown' for r in results),
        no_feasible_groups=sum(r['selection_status']=='no_feasible' for r in results),
        selection_rule=dict(minimum_completion=minimum_completion,probability_tolerance=probability_tolerance),
        selected_complete_count=sum(r['selected_complete'] for r in decided),
        false_clearance_count=sum(not r['selected_complete'] for r in decided),
        clearance_denominator=len(decided),
        selection_coverage=len(decided)/len(results) if results else None,
        false_clearance_scope='Selected candidate fails the full task; abstentions excluded from clearance denominator and retained in query coverage',
        brier_score=sum(errors)/len(errors) if errors else None,brier_count=len(errors),
        rows=results)
