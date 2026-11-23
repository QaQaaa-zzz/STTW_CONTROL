"""Finite-panel completion bounds with unmatched or unobserved P0-B prefixes."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_chapter3_continuous_p0_branches import ENTRY, FORCES, LEGACY, TASKS, read, read_batch

MEASURES = ('A','B','C','D','B-A','C-A','D-B','D-C','D-A','D-B-C+A')
KEY = ['layout','root','section','side']
METHODS = ('old_filter','rolling')


def shared_score_bounds(lower, upper, old):
    """Extrema, in number of pushes, over binary completions inside the input box.

    lower/upper axes are policy, entry, push. For any fixed policy, each entry's
    total is any integer between its lower and upper sum. The maximum over a
    disjoint set of entries attains every integer between the maxima of these
    bounds. A and C share the SAME maximum over original entries, as do B and D.
    """
    groups=[]
    for p in range(2):
        for entries in (old, ~old):
            low=int(lower[p,entries].sum(axis=1).max())
            high=int(upper[p,entries].sum(axis=1).max())
            groups.append(range(low,high+1))
    values=[]
    for a,extra_old_filter,b,extra_rolling in product(*groups):
        c=max(a,extra_old_filter)
        d=max(b,extra_rolling)
        values.append([a,b,c,d,b-a,c-a,d-b,d-c,d-a,d-b-c+a])
    values=np.asarray(values,dtype=int)
    return values.min(axis=0),values.max(axis=0)


def evidence_class(matched, no_handback, no_scorer_or_query, no_exit_applied):
    if matched:
        return 'matched'
    if no_handback and no_scorer_or_query and no_exit_applied:
        return 'unmatched_no_handback_before_any_exit'
    return 'unmatched_other'


def prefix_evidence(run, plan, output):
    """Identify completed groups from physical state and suffix receipts, keeping the full frozen panel."""
    evidence, known = [], []
    official=run/'analysis/measured_exits.csv.gz'
    canonical=pd.read_csv(official) if official.exists() else None
    for batch in plan['batches']:
        name=batch['batch']
        folder=run/name/'physical'
        variants=read(run/name/'config.json')['ground_handoff']['handoff_variants']
        status=read(folder/'status.json') if (folder/'status.json').exists() else {}
        rows=status.get('results',[]) if status.get('stage')=='complete' else []
        if rows:
            if len(rows)!=batch['worlds'] or len(variants)!=len(rows):
                raise ValueError(f'Incomplete finished batch {name}')
            with np.load(folder/'p0_branch_audit.npz') as z:
                pairing=z['pair_ok']
                bad_tick=z['first_bad_tick']
                query_count=z['queries']
            acted=set()
            published=set()
            with (folder/'p0_events.jsonl').open() as stream:
                for line in stream:
                    event=json.loads(line)
                    if event['kind'] in ('old_selection','query'):
                        acted.update(event['lanes'])
                    elif event['kind']=='publication':
                        published.update(i for i,issued in zip(event['lanes'],event['published'],strict=True) if issued)
            cache=output/'observed'/name/'matched_exits.csv.gz'
            stamp=cache.with_name('source_ids.json')
            sources=[run/'protocol.json',run/name/'config.json',folder/'status.json',
                     folder/'p0_branch_audit.npz',folder/'p0_events.jsonl',
                     Path(__file__).with_name('analyze_chapter3_continuous_p0_branches.py')]
            identity={str(path):sha256(path.read_bytes()).hexdigest() for path in sources}
            if canonical is not None and 'batch' in canonical and name in set(canonical.batch):
                observed=canonical[canonical.batch.eq(name)]
            elif cache.exists() and stamp.exists() and read(stamp)==identity:
                observed=pd.read_csv(cache)
            else:
                _,_,observed,_,_=read_batch(run,name,plan)
                cache.parent.mkdir(parents=True,exist_ok=True)
                observed[observed.planner.isin(METHODS)].to_csv(cache,index=False)
                stamp.write_text(json.dumps(identity,indent=2)+'\n')
            known.append(observed[observed.planner.isin(METHODS)])
        for k in range(0,len(variants),7):
            g=variants[k:k+7]
            v=g[0]
            identity=dict(layout=name.split('/')[0],root=v['bank_row'],geometry=v['geometry'],
                          alpha=v['approach_alpha'],force=v['force_y_N'],batch=name,group_start_lane=k)
            if [r['arm'] for r in g]!=plan['arms'] or len({(r['bank_row'],r['geometry'],r['approach_alpha'],
                      r['force_y_N']) for r in g})!=1:
                raise ValueError(f'Invalid input group {name}/{k}')
            row=dict(**identity,state='batch_pending',first_mismatch_tick=None,all_seven_no_handback=None,
                     no_suffix_published_or_applied=None,no_scorer_or_query=None,any_goal=None)
            if rows:
                rr=rows[k:k+7]
                no_hand=all(r['handoff_s'] is None for r in rr)
                ids=set(range(k,k+7))
                no_output=(not ids & published and all(r['departure_applied_tick']<0 for r in rr))
                no_decision=not ids & acted and not query_count[k:k+7].any()
                matched=bool(pairing[k//7] and len({r['handoff_s'] for r in rr})==1)
                state=evidence_class(matched,no_hand,no_decision,no_output)
                row.update(state=state,first_mismatch_tick=int(bad_tick[k//7]),
                           all_seven_no_handback=no_hand,no_suffix_published_or_applied=no_output,
                           no_scorer_or_query=no_decision,any_goal=any(r['goal'] for r in rr))
            evidence.append(row)
    score_cols=['layout','root','section','side','force']+ENTRY+['planner','success']
    scores=pd.concat(known,ignore_index=True)[score_cols] if known else pd.DataFrame(columns=score_cols)
    return pd.DataFrame(evidence),scores


def calculate(run,output):
    plan=read(run/'protocol.json')
    actions=[tuple(v) for v in plan['actions']]
    old=np.array([a in LEGACY for a in actions])
    if len(actions)!=18 or old.sum()!=5 or set(map(tuple,plan['old_actions']))!=LEGACY:
        raise ValueError('Unexpected entry library')
    output.mkdir(parents=True,exist_ok=True)
    sources,scores=prefix_evidence(run,plan,output)
    prefix_key=['layout','root']+ENTRY+['force']
    if len(sources)!=plan['unique_prefixes'] or sources.duplicated(prefix_key).any():
        raise ValueError('Frozen prefix denominator changed')
    sources.to_csv(output/'prefix_evidence.csv',index=False)
    tasks=pd.DataFrame(TASKS,columns=['section','side']).merge(pd.DataFrame({'planner':METHODS}),how='cross')
    cases=sources.merge(tasks,how='cross').merge(scores,on=prefix_key+['section','side','planner'],how='left',
                                                  validate='one_to_one')
    if not (cases.state.eq('matched')==cases.success.notna()).all():
        raise ValueError('A matched outcome is missing or an unmatched outcome was assigned an exact label')
    cases['lower_strict']=cases.success.eq(True).astype(int)
    cases['upper_strict']=(~cases.success.eq(False)).astype(int)
    cases['lower_observed_terminal']=cases.lower_strict
    cases['upper_observed_terminal']=cases.upper_strict.mask(cases.state.eq('unmatched_no_handback_before_any_exit'),0)
    cases.to_csv(output/'atomic_task_intervals.csv.gz',index=False)
    records=[]
    cell_index=pd.MultiIndex.from_tuples([(p,*a,f) for p,a,f in product(METHODS,actions,FORCES)],
                                         names=['planner']+ENTRY+['force'])
    for key,group in cases.groupby(KEY):
        if group.batch.nunique()!=1:
            raise ValueError('A nominal reset root has been split over multiple physical batches')
        batch=group.batch.iat[0]
        values=group.set_index(cell_index.names).reindex(cell_index)
        for rule in ('strict','observed_terminal'):
            cols=[f'lower_{rule}',f'upper_{rule}']
            if values[cols].isna().any().any():
                raise ValueError(f'Incomplete frozen design at {key}')
            lower,upper=(values[col].to_numpy(dtype=int).reshape(2,len(actions),3) for col in cols)
            lo,hi=shared_score_bounds(lower,upper,old)
            for measure,l,h in zip(MEASURES,lo,hi,strict=True):
                records.append(dict(zip(KEY,key),batch=batch,rule=rule,measure=measure,n_contracts=3,lower=l,upper=h))
    root_bounds=pd.DataFrame(records)
    root_bounds.to_csv(output/'shared_entry_bounds_by_root_gate.csv',index=False)
    by_batch=root_bounds.groupby(['layout','batch','rule','measure'],as_index=False)[['n_contracts','lower','upper']].sum()
    by_batch['lower_pp']=100*by_batch.lower/by_batch.n_contracts
    by_batch['upper_pp']=100*by_batch.upper/by_batch.n_contracts
    by_batch.to_csv(output/'shared_entry_bounds_by_planned_batch.csv',index=False)
    by_map=root_bounds.groupby(['layout','rule','measure'],as_index=False)[['n_contracts','lower','upper']].sum()
    overall=root_bounds.groupby(['rule','measure'],as_index=False)[['n_contracts','lower','upper']].sum()
    totals=pd.concat([overall.assign(layout='all'),by_map],ignore_index=True)
    totals['lower_pp']=100*totals.lower/totals.n_contracts
    totals['upper_pp']=100*totals.upper/totals.n_contracts
    totals.to_csv(output/'shared_entry_bounds.csv',index=False)
    receipt=dict(prefix_counts={k:int(v) for k,v in sources.state.value_counts().items()},
       nominal_prefixes=len(sources),nominal_root_gate_units=cases.groupby(KEY).ngroups,
       contracts_per_cell=cases.groupby(KEY+['force']).ngroups,atomic_intervals=len(cases),
       target='Per map/root/gate choose one entry for all three forces. A/B original, C/D expanded; A/C old filter; B/D rolling.',
       method='For each policy separately enumerate attainable old-subset and added-subset maxima in the missing-binary-outcome box. Preserve nesting when differencing. Then add bounds over nominal map/root/gate units.',
       qualifications='These are logical identification bounds on finite hindsight planning potential under the stated observation model, not sampling confidence intervals. Extremes need not be jointly physically realizable across gates or policies. Pending batches remain in the denominator.',
       strict='Every unpaired or unobserved task remains unknown.',
       observed_terminal='For an estimand anchored on any of the seven actually archived pre-exit histories only: if all ended without a handback and none emitted or applied an exit decision, successful completion requiring handback is impossible on each history. Every other unmatched group remains unknown. A lack of publication alone never declares failure.',
       display='Scores are percentages; differences are percentage points. No deployed expanded entry selector is evaluated.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('prefix_counts','contracts_per_cell','atomic_intervals')}),
          totals[(totals.layout=='all')&totals.measure.isin(['A','B','C','D','B-A','C-A','D-B','D-C','D-B-C+A'])].to_string(index=False),
          sep='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.run.resolve()
    calculate(root,args.output.resolve() if args.output else root/'analysis/partial_identification')
