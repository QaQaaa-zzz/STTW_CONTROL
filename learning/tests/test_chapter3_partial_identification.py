"""Compare finite-box bounds with direct enumeration of every missing binary outcome."""
from itertools import product
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
from bound_chapter3_p0_partial_identification import MEASURES, evidence_class, shared_score_bounds


def direct(z, original):
    # One entry is chosen for all pushes, separately for each post-handoff policy.
    by_entry=z.sum(axis=2)
    a,b=by_entry[:,original].max(axis=1)
    c,d=by_entry.max(axis=1)
    return [a,b,c,d,b-a,c-a,d-b,d-c,d-a,d-b-c+a]


def test_binary_completions_and_paired_contrasts():
    rng=np.random.default_rng(20261123)
    original=np.array([True,True,False])
    for _ in range(15):
        lower=rng.integers(0,2,size=(2,3,3))
        missing=rng.choice(lower.size,size=6,replace=False)
        lower.flat[missing]=0
        upper=lower.copy()
        upper.flat[missing]=1
        outcomes=[]
        for choices in product((0,1),repeat=len(missing)):
            z=lower.copy()
            z.flat[missing]=choices
            outcomes.append(direct(z,original))
        bounds=shared_score_bounds(lower,upper,original)
        np.testing.assert_array_equal(bounds[0],np.min(outcomes,axis=0))
        np.testing.assert_array_equal(bounds[1],np.max(outcomes,axis=0))


def test_no_information_and_nesting():
    lo=np.zeros((2,3,3),dtype=int)
    bounds=shared_score_bounds(lo,np.ones_like(lo),np.array([False,True,True]))
    records=dict(zip(MEASURES,zip(*bounds)))
    assert records['A']==(0,3)
    assert records['B-A']==(-3,3)
    assert records['C-A']==records['D-B']==(0,3)
    assert records['D-B-C+A']==(-3,3)


def test_known_shared_choice_cannot_see_push():
    z=np.array([[[1,0,0],[0,1,1],[1,1,1]], [[1,1,0],[0,0,1],[1,0,1]]])
    bounds=shared_score_bounds(z,z,np.array([True,True,False]))
    np.testing.assert_array_equal(bounds[0],bounds[1])
    assert dict(zip(MEASURES,bounds[0]))['A']==2
    assert dict(zip(MEASURES,bounds[0]))['C']==3


def test_failure_requires_no_handback_and_no_evidence_of_exit_use():
    assert evidence_class(True,True,True,True)=='matched'
    assert evidence_class(False,True,True,True)=='unmatched_no_handback_before_any_exit'
    for facts in [(False,True,True),(True,False,True),(True,True,False)]:
        assert evidence_class(False,*facts)=='unmatched_other'
