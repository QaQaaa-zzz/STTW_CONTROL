"""Offline bounded-reference scenario baseline; no online event-conditioned action.

This deliberately ordinary CEM/random comparison changes the search objective,
not the frozen controller or six-coordinate reference representation. Rollout
callbacks must charge every candidate/scenario, including repeated incumbents.
"""
from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import numpy as np


EVENT_FIELDS = ('event_id', 'force_N', 'pulse_onset_task_tick',
                'pulse_duration_s', 'world_direction')


@dataclass(frozen=True)
class ScenarioScore:
    nominal_feasible: bool
    survived: int
    arrived: int
    mean_disturbed_cost: float
    nominal_cost: float

    @property
    def key(self):
        # Endpoint counts first, then ORIGINAL-alpha full-horizon cost. No
        # scalar coefficient pretends to convert falls into tracking metres.
        return (not self.nominal_feasible, -self.survived, -self.arrived,
                self.mean_disturbed_cost)


def validate_scenarios(scenarios):
    if len(scenarios) < 3:
        raise ValueError('one nominal and multiple disturbed scenarios required')
    if scenarios[0]['force_N'] != 0 or any(s['force_N'] == 0 for s in scenarios[1:]):
        raise ValueError('exactly one nominal, first; no duplicated zero-force weight')
    ids = [s['event_id'] for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError('unique scenario identities required')
    for s in scenarios:
        if any(k not in s for k in EVENT_FIELDS):
            raise ValueError('complete declared event identity required')
        values = [s['force_N'], s['pulse_onset_task_tick'], s['pulse_duration_s'], *s['world_direction']]
        if not np.isfinite(values).all() or s['pulse_duration_s'] <= 0:
            raise ValueError('finite positive-duration scenarios required')
        if len(s['world_direction']) != 3 or np.linalg.norm(s['world_direction']) == 0:
            raise ValueError('physical world direction required')
        if type(s['pulse_onset_task_tick']) is not int or s['pulse_onset_task_tick'] < 0:
            raise ValueError('nonnegative integer task clock required')


def aggregate(candidates, rows, scenarios, *, nominal_cost_limit):
    """Require a complete candidate-major rectangle and original-alpha identity.

    Counts concern measured discrete scenarios, not a continuous safe force
    interval. A known fall is a valid outcome; unknown/fault never is a fall.
    A 1m arrival before contact and full-window survival remain separate.
    """
    validate_scenarios(scenarios)
    candidates = np.asarray(candidates, dtype=np.float32)
    if (candidates.ndim != 2 or candidates.shape[1] != 6 or
            not np.isfinite(candidates).all() or np.any(abs(candidates) > 1)):
        raise ValueError('bounded six-coordinate candidates required')
    if len(rows) != len(candidates)*len(scenarios):
        raise ValueError('complete rectangular scenario outcomes required')
    if not np.isfinite(nominal_cost_limit) or nominal_cost_limit < 0:
        raise ValueError('finite nonnegative nominal task-cost limit required')
    scores = []
    task = (scenarios[0]['context'], scenarios[0]['alpha'])
    if any((s['context'], s['alpha']) != task for s in scenarios):
        raise ValueError('one context and ORIGINAL user alpha per search required')
    for i, vector in enumerate(candidates):
        group = rows[i*len(scenarios):(i+1)*len(scenarios)]
        for r, expected in zip(group, scenarios):
            if any(r[k] != expected[k] for k in (*EVENT_FIELDS, 'context', 'alpha')):
                raise ValueError('event/task identity or row order changed')
            if not np.array_equal(np.asarray(r['vector'], np.float32), vector):
                raise ValueError('one identical reference process for all its scenarios required')
            if r['physical_outcome'] not in ('survived', 'contact'):
                raise ValueError('unknown engineering outcome cannot be a physical failure')
            arrived = r['goals']['1.0']['goal_arrived']
            if type(arrived) is not bool:
                raise ValueError('known original-world goal label required')
            if isinstance(r['scored_cost'], bool) or not np.isfinite(r['scored_cost']) or r['scored_cost'] < 0:
                raise ValueError('finite ORIGINAL-alpha scored cost required')
        nominal = group[0]
        scores.append(ScenarioScore(
            nominal_feasible=(nominal['physical_outcome'] == 'survived' and
                              nominal['goals']['1.0']['goal_arrived'] and
                              nominal['scored_cost'] <= nominal_cost_limit),
            survived=sum(r['physical_outcome'] == 'survived' for r in group[1:]),
            arrived=sum(r['goals']['1.0']['goal_arrived'] for r in group[1:]),
            mean_disturbed_cost=float(np.mean([r['scored_cost'] for r in group[1:]])),
            nominal_cost=float(nominal['scored_cost'])))
    return scores


def search(evaluate, scenarios, *, seed, method, nominal_cost_fraction=.1):
    """Fixed 4x16 CEM versus matched fixed-distribution random.

    Nominal cost may increase by a declared fraction of ZERO-bias cost. This
    search constraint does not redefine task success. Disturbed endpoint counts
    allow individual failures and losses; aggregate improvement is sought.
    No holdout callback is accepted: freeze this output before evaluation.
    """
    validate_scenarios(scenarios)
    if method not in ('cem', 'random'):
        raise ValueError('CEM or matched random only')
    if not np.isfinite(nominal_cost_fraction) or nominal_cost_fraction < 0:
        raise ValueError('predeclared nonnegative nominal cost allowance required')
    rng = np.random.default_rng(seed)
    mean = np.zeros(6); std = np.full(6, .5)
    incumbent = np.zeros(6, np.float32); best = None; limit = None; history = []
    for generation in range(4):
        vectors = np.clip(mean+std*rng.standard_normal((16, 6)), -1, 1).astype(np.float32)
        vectors[0] = incumbent
        rows = evaluate(vectors.copy())
        if limit is None:
            # Full validation follows; do not let a malformed callback assign a
            # convenient limit or silently replace an infeasible zero baseline.
            aggregate(vectors, rows, scenarios, nominal_cost_limit=np.finfo(float).max)
            baseline = rows[0]['scored_cost']
            limit = float(baseline*(1+nominal_cost_fraction))
        scores = aggregate(vectors, rows, scenarios, nominal_cost_limit=limit)
        if not scores[0].nominal_feasible:
            raise ValueError('nominal incumbent invalid; stop without switching the task')
        order = sorted(range(16), key=lambda i:scores[i].key)
        winner = order[0]
        if best is None or scores[winner].key < best.key:
            incumbent, best = vectors[winner].copy(), scores[winner]
        elite = [i for i in order if scores[i].nominal_feasible][:4]
        if method == 'cem':
            mean = .4*mean+.6*vectors[elite].mean(axis=0)
            std = np.maximum(.1, .4*std+.6*vectors[elite].std(axis=0))
        history.append(dict(generation=generation, vectors=vectors.tolist(),
                            scores=[asdict(s) for s in scores],
                            incumbent=incumbent.tolist(), best=asdict(best)))
    return dict(vector=incumbent.tolist(), score=asdict(best), method=method,
                nominal_cost_limit=limit, nominal_cost_fraction=nominal_cost_fraction,
                scored_candidate_processes=64, charged_rollout_calls=64*len(scenarios),
                note='Every duplicate incumbent is charged; physical ticks/slots come from runtime, not this call count',
                history=history)


def evaluate_generation(runtime, scenarios, vectors, directory):
    """Execute the real BiasRuntime interface, candidate-major, without padding.

    All sixteen processes see the SAME predeclared physical scenario set. Events
    enter the physical rollout, never the BiasedActor. Runtime's immutable plan
    and source validation still apply; this adapter cannot bypass stale seals.
    """
    from .preference_best import atomic_json, digest
    validate_scenarios(scenarios)
    vectors = np.asarray(vectors, np.float32)
    if vectors.shape != (16, 6) or not np.isfinite(vectors).all() or np.any(abs(vectors) > 1):
        raise ValueError('one fixed sixteen-process generation required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases, repeated = [], []
    for i, vector in enumerate(vectors):
        for j, scenario in enumerate(scenarios):
            case = copy.deepcopy(scenario)
            case.update(case_id=f'{scenario["case_id"]}/candidate_{i:02d}',
                        candidate_index=i, scenario_index=j)
            cases.append(case); repeated.append(vector)
    atomic_json(directory/'inventory.json', dict(cases=cases, vectors=np.asarray(repeated).tolist(),
                candidate_processes=16, scenario_count=len(scenarios),
                grouping='candidate-major; one identical vector across all events'))
    rows = []; batches = []; offset = 0
    # 16 * scenario_count is always divisible by sixteen. True lane counts,
    # unlike active physical ticks, are equal across CEM and random searches.
    while offset < len(cases):
        n = 64 if len(cases)-offset >= 64 else 16
        out = directory/f'batch_{len(batches):02d}'
        got = runtime.rollout(cases[offset:offset+n], np.asarray(repeated[offset:offset+n]),
                              out, allowed_real_batch_sizes=(16, 64))
        if len(got) != n:
            raise ValueError('incomplete physical batch; no implicit retry or negative labels')
        status = json.loads((out/'status.json').read_text())
        if status['phase'] != 'completed' or status['completed_cases'] != n:
            raise ValueError('completed audited runtime outputs required')
        batches.append(dict(status_file=str((out/'status.json').resolve()),
                            status_sha256=digest(out/'status.json'),
                            active_lower_ticks=status['active_lower_ticks'],
                            allocated_lower_slots=status['allocated_lower_slots'], real_lanes=n))
        rows.extend(got); offset += n
    # Validate the whole rectangle before allowing its outcomes into selection.
    aggregate(vectors, rows, scenarios, nominal_cost_limit=np.finfo(float).max)
    atomic_json(directory/'results.json', rows)
    atomic_json(directory/'receipt.json', dict(phase='completed',
                candidate_processes=16, charged_rollout_calls=len(cases),
                actual_lower_ticks=sum(b['active_lower_ticks'] for b in batches),
                allocated_lower_slots=sum(b['allocated_lower_slots'] for b in batches),
                batches=batches, padding=0, optimizer_updates=0))
    return rows
