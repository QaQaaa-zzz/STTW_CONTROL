import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.tolerance_predictor_pilot import apply_feature_mask,predict,make_model,validate_user_mix_config
from sttw_control.tolerance_coordination import require_unmasked_online_inputs

KEEP=[4]+list(range(34,41))


def test_removes33_features_keeps8_and_preserves_original_array():
    x=np.arange(123,dtype=np.float32).reshape(3,41);original=x.copy();masked=apply_feature_mask(x,KEEP)
    np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(masked[:,KEEP],x[:,KEEP])
    assert np.all(masked[:,[i for i in range(41) if i not in KEEP]]==0)
    assert apply_feature_mask(x,None) is x


def test_prediction_mask_matches_manual_and_removals_cannot_change_prediction():
    import torch
    torch.manual_seed(3);model=make_model(41)
    rng=np.random.default_rng(3);x=rng.normal(size=(5,41)).astype(np.float32);altered=x.copy();altered[:,[i for i in range(41) if i not in KEEP]]+=100
    masked=predict(model,x,feature_mask_keep_indices=KEEP)
    np.testing.assert_array_equal(masked,predict(model,apply_feature_mask(x,KEEP)))
    np.testing.assert_array_equal(masked,predict(model,altered,feature_mask_keep_indices=KEEP))
    with torch.no_grad():raw=torch.sigmoid(model(torch.as_tensor(x)).squeeze(-1)).numpy()
    np.testing.assert_array_equal(predict(model,x),raw)
    assert not np.array_equal(masked,predict(model,x))


def test_config_marks_offline_same_capacity_and_export_rejects_mask():
    c=json.loads((Path(__file__).parents[1]/'configs/h75_tolerance_speed_query_ablation.json').read_text());validate_user_mix_config(c)
    assert c['architecture']['hidden']==[16,16] and c['features']['full_dim']==41
    with pytest.raises(ValueError):require_unmasked_online_inputs(c)
    require_unmasked_online_inputs({})
    c['feature_mask_keep_indices']=[4]
    with pytest.raises(ValueError):validate_user_mix_config(c)


def test_fixed_seed_initial_tensors_remain_identical_for_masked_predictor():
    import torch
    torch.manual_seed(0);a=make_model(41).state_dict()
    torch.manual_seed(0);b=make_model(41).state_dict()
    assert all(torch.equal(a[k],b[k]) for k in a)
