"""Finite directional pulse evidence; never certifies a continuous safe interval."""
from __future__ import annotations
import math
from typing import Iterable

def summarize_direction(trials: Iterable[tuple[float, bool | None]]) -> dict:
    """Summarize one full-state/endpoint/direction/pulse/deadline identity.

    Inputs are nonnegative force magnitudes in N and True/False/None labels.
    None means unknown (e.g. execution fault), not physical task failure.
    The caller must verify shared identity before combining trials. Repeated
    amplitudes need a separate probabilistic protocol and are rejected here.
    """
    rows=[]
    for force,label in trials:
        force=float(force)
        if not math.isfinite(force) or force<0 or (label is not None and type(label) is not bool):
            raise ValueError('finite nonnegative magnitude and bool/None label required')
        rows.append((force,label))
    rows.sort(key=lambda x:x[0])
    if len({f for f,_ in rows})!=len(rows):
        raise ValueError('duplicate amplitudes require separate repeated-trial protocol')
    unknown=[f for f,label in rows if label is None]
    first_failure=next((f for f,label in rows if label is False),None)
    later_success=[f for f,label in rows if label is True and first_failure is not None and f>first_failure]
    result=dict(trials=[dict(force_N=f,success=label) for f,label in rows],
                last_prefix_grid_success_N=None,first_grid_failure_N=first_failure,
                success_after_failure_N=later_success,unknown_amplitudes_N=unknown,
                continuous_margin_N=None)
    nominal=next((label for f,label in rows if f==0),None)
    if nominal is None:return dict(result,status='UNKNOWN_NOMINAL')
    if nominal is False:return dict(result,status='NOMINAL_FAILURE')
    for f,label in rows:
        if label is not True:break
        result['last_prefix_grid_success_N']=f
    status=('UNTESTED_DIRECTION' if len(rows)==1 else
            'INCOMPLETE_GRID' if unknown else
            'SAMPLED_TRANSITION_NONMONOTONE' if later_success else
            'SAMPLED_TRANSITION' if first_failure is not None else 'RIGHT_CENSORED')
    return dict(result,status=status)
