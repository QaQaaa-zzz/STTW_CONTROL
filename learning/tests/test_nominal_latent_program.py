import numpy as np
import pytest
from sttw_control.nominal_latent_program import extract_nominal_program


def example():
    return dict(active_tick=np.array([True]*8+[False]*4), time=np.r_[np.arange(8)*.005, [0]*4],
                original_policy_latent=np.vstack([np.repeat([[.1,.2],[.3,.4]],4,axis=0),np.full((4,2),np.nan)]),
                upper_active=np.array([True]*8+[False]*4), fallback_active=np.zeros(12,bool))


def test_extraction_uses_only_observed_full_publications():
    program, receipt = extract_nominal_program(example(),lower_ticks=8)
    np.testing.assert_array_equal(program,[[.1,.2],[.3,.4]])
    assert receipt['eligible'] and receipt['upper_publications']==2


def test_domain_exit_missing_actions_are_not_filled_with_zeros_or_last_value():
    log=example();log['upper_active'][4:8]=False;log['fallback_active'][4:8]=True
    log['original_policy_latent'][4:8]=np.nan
    program, receipt=extract_nominal_program(log,lower_ticks=8)
    assert program is None and not receipt['eligible']
    assert 'missing_nominal_bare_latents_no_fill_allowed' in receipt['reasons']


def test_intra_publication_change_is_not_silently_downsampled():
    log=example();log['original_policy_latent'][1,0]+=.01
    with pytest.raises(ValueError,match='within'):
        extract_nominal_program(log,lower_ticks=8)
