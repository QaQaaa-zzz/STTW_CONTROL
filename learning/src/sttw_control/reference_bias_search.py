"""Nominal-only selection and fixed event expansion; never selects per force."""
import copy
import numpy as np


def batch_slices(n):
    if type(n) is not int or n<=0 or n%8:
        raise ValueError('positive multiples of eight real cases required')
    return [slice(i,min(i+16,n)) for i in range(0,n,16)]


def freeze_selected(search, outcome, rows):
    vector=np.asarray(outcome['vector'],float)
    if vector.shape!=(6,) or not np.isfinite(vector).all() or np.any(abs(vector)>1):raise ValueError('invalid selected vector')
    if len(rows)!=64 or any(r['force_N']!=0 or r['context']!=search['context'] or r['alpha']!=search['alpha'] for r in rows):
        raise ValueError('all64 matching nominal-only candidates required')
    if any(r['physical_outcome']=='unknown' for r in rows):raise ValueError('unknown candidate cannot select a process')
    matches=[r for r in rows if r['nominal_feasible'] and r['objective']==outcome['cost'] and np.array_equal(np.asarray(r['vector']),vector)]
    if not matches:raise ValueError('selected process has no actually scored feasible witness')
    witness=matches[0]
    return dict(context=search['context'],alpha=search['alpha'],method=search['method'],vector=vector.tolist(),
        nominal_cost=float(outcome['cost']),nominal_case_id=witness['case_id'],nominal_raw_file=witness['raw_file'],
        nominal_raw_sha256=witness['raw_sha256'],search_id=search['search_id'],seed=search['seed'],
        information='Selected only from matching zero-force whole-task objective; one vector for ALL event directions/amplitudes/onsets.')


def evaluation_schedule(events,selected):
    if len(events)!=60 or len(selected)!=24:raise ValueError('complete60 original events and24 nominal decisions required')
    table={(r['context'],r['alpha'],r['method']):r for r in selected}
    if len(table)!=24:raise ValueError('duplicate or missing task decisions')
    if len({(e['context'],e['alpha'],e['event_id']) for e in events})!=60:raise ValueError('duplicate original events')
    cases=[];vectors=[]
    for method in ['cem','random']:
        for event in events:
            decision=table[(event['context'],event['alpha'],method)]
            c=copy.deepcopy(event);c.update(role=method,case_id=event['case_id'].rsplit('/',1)[0]+'/bias_'+method,
                selected_nominal_case_id=decision['nominal_case_id'],selected_nominal_raw_sha256=decision['nominal_raw_sha256'])
            cases.append(c);vectors.append(decision['vector'])
    return cases,np.asarray(vectors,float)
