import numpy as np
import pytest

from sttw_control.lineage_inference import embed_legacy_actor


def legacy():
    widths = [351,128,128,64,2]
    return {'params': {f'Dense_{i}': {'kernel': np.ones((widths[i],widths[i+1]),np.float32),
                                    'bias': np.zeros(widths[i+1],np.float32)} for i in range(4)}}


def test_embedding_does_not_change_original_and_cannot_erase_learned_modulation():
    original = legacy()
    embedded = embed_legacy_actor(original)
    assert set(original['params']) == {f'Dense_{i}' for i in range(4)}
    for name in ['gain_2','shift_2','gain_3','shift_3']:
        assert not embedded['params'][name].any()
    embedded['params']['Dense_0']['kernel'][0,0] = 19.
    assert original['params']['Dense_0']['kernel'][0,0] == 1.
    embedded['params']['gain_2'][0] = .5
    with pytest.raises(ValueError,match='unmodulated'):
        embed_legacy_actor(embedded)


def test_embedding_rejects_wrong_input_shape_instead_of_silently_padding():
    original = legacy()
    original['params']['Dense_0']['kernel'] = np.zeros((345,128),np.float32)
    with pytest.raises(ValueError,match='Dense_0 shape'):
        embed_legacy_actor(original)
