"""Permission factor must not expand the existing candidate or NN domain."""
import itertools
import jax
import jax.numpy as j
import numpy as np

from sttw_control.anticipatory_reference import apply_reference_permission


def test_all_permissions_and_profiles_under_real_jit_vmap():
    rows=list(itertools.product([False,True],repeat=4))
    cases=np.array([(*r,p) for r in rows for p in range(8)],dtype=np.int32)
    action=j.array([1.23,-.78])
    def apply(row):
        independent,disabled,stopped,zero,profile=row
        return apply_reference_permission(action,disabled=disabled.astype(bool),
            stopped=stopped.astype(bool),zero_mode=zero.astype(bool),
            profile=profile,independent_support=independent.astype(bool))
    values,flags=jax.jit(jax.vmap(apply))(j.asarray(cases))
    for row,value,flag in zip(cases,np.asarray(values),np.asarray(flags)):
        independent,disabled,stopped,zero,profile=map(int,row)
        override=bool(independent and disabled and not stopped and not zero and profile==7)
        assert bool(flag)==override
        expected=[1.23,0.] if override else ([0.,0.] if disabled else [1.23,-.78])
        np.testing.assert_array_equal(value,np.asarray(expected,dtype=np.float32))


def test_default_retains_original_mask_and_enabled_nonfinite_signal():
    source=j.array([j.nan,1.])
    action,flag=apply_reference_permission(source,disabled=False,stopped=False,
        zero_mode=False,profile=7)
    assert np.isnan(action[0]) and action[1]==1 and not flag
    action,flag=apply_reference_permission(source,disabled=True,stopped=False,
        zero_mode=False,profile=7)
    np.testing.assert_array_equal(action,[0.,0.]);assert not flag
