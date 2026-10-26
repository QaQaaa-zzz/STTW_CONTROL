import numpy as np
from review_passive_sensor import first_legal_latch


def test_first_latch_ignores_disallowed_or_repeated_rows_without_future_labels():
    raw=dict(active_tick=np.ones(16,bool),physical_tick=np.arange(16),
        actor_action_valid=np.ones(16,bool),probe_eso_history_valid=np.ones(16,bool),
        probe_eso_signal=np.full(16,.02))
    raw['actor_action_valid'][:4]=False
    raw['probe_eso_history_valid'][4:8]=False
    # No force, onset, alpha, future or outcome fields are available here.
    assert first_legal_latch(raw)==(8,2,3,2)
    raw['probe_eso_signal'][8:]=0.
    assert first_legal_latch(raw)==(None,2,3,0)
