#!/usr/bin/env python3
"""Pure selection helpers. Caller supplies fixed-protocol physical metrics.

This helper never selects on a stochastic rollout return and never writes an
unchecked checkpoint as qualified. Repository adapter must compute counts from
complete deterministic traces and atomically persist the selected checkpoint.
"""
from __future__ import annotations
import math

def selection_key(summary:dict)->tuple:
    count_names=('physical_failure_cases','working_limit_failure_cases',
                 'primary_failure_cases','joint_final_hold_failure_cases')
    for name in count_names:
        v=summary[name]
        if not isinstance(v,int) or v<0: raise ValueError(name)
    q=float(summary['physical_quality_score'])
    if not math.isfinite(q) or q<0: raise ValueError('invalid quality')
    if not summary.get('all_declared_traces_complete',False): raise ValueError('partial validation cannot select best')
    return tuple(summary[x] for x in count_names)+(q,)

def is_better(candidate:dict,incumbent:dict|None)->bool:
    key=selection_key(candidate)
    if incumbent is None:return True
    old=selection_key(incumbent)
    if key[:4]!=old[:4]:return key[:4]<old[:4]
    return key[4]<old[4]-max(1e-6,.005*old[4])

def qualified(summary:dict)->bool:
    return selection_key(summary)[:4]==(0,0,0,0)

if __name__=='__main__':
    base=dict(physical_failure_cases=0,working_limit_failure_cases=1,primary_failure_cases=0,joint_final_hold_failure_cases=1,physical_quality_score=2.,all_declared_traces_complete=True)
    good={**base,'working_limit_failure_cases':0,'physical_quality_score':4.}
    assert is_better(good,base) and not is_better(base,good)
    assert not qualified(base)
    try: selection_key({**base,'all_declared_traces_complete':False})
    except ValueError: pass
    else: raise AssertionError('partial not rejected')
    print('best-selection mathematical tests passed')
