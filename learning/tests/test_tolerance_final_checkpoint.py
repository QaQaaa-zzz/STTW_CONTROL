"""Final learner boundary must survive non-multiple-of-five run lengths."""
import pytest
from sttw_control.tolerance_training import checkpoint_due


@pytest.mark.parametrize('total,expected',[(27,[5,10,15,20,25,27]),(25,[5,10,15,20,25]),(2,[2])])
def test_final_checkpoint_is_always_due(total,expected):
    assert [i for i in range(1,total+1) if checkpoint_due(i,total)]==expected
    assert checkpoint_due(3,27,stop=True)


@pytest.mark.parametrize('legacy',[False,True])
def test_authenticated_recovery_restores_anchor_and_counters(tmp_path,legacy):
    import copy
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from tensordict import TensorDict
    from sttw_control import path_command_training as training
    from sttw_control.tolerance_training import restore_recovery_boundary
    from sttw_control.preference_best import digest
    cfg=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Coverage_V2.json').read_text())
    obs=TensorDict({'policy':torch.zeros(4,351),'critic':torch.zeros(4,352)},batch_size=[4])
    algo=training.make_algorithm(obs,2,cfg,'cpu')
    algo.configure_parent_anchor(torch.zeros(17,351),torch.zeros(17,2),torch.zeros(2))
    algo.parent_anchor_cursor=12;identity={'schema':training.SCHEMA,'experiment':'short_recovery_v1'}
    weights=copy.deepcopy(algo.policy.state_dict())
    cp=training.save_checkpoint(tmp_path,algo,25,identity,route_counters=np.arange(4))
    record=dict(checkpoint=str(cp),checkpoint_sha256=digest(cp),update=25,parent_anchor_cursor=12)
    if legacy:
        saved_legacy=torch.load(cp,weights_only=False);saved_legacy.pop('runtime_state');torch.save(saved_legacy,cp)
        metrics=tmp_path/'metrics.jsonl'
        metrics.write_text(''.join(json.dumps(dict(update=i,optimizer=dict(accepted_minibatches=1 if i>2 else 0)))+'\n' for i in range(1,26)))
        record.update(checkpoint_sha256=digest(cp),cursor_source_metrics=str(metrics),cursor_source_metrics_sha256=digest(metrics))
    algo.parent_anchor_cursor=0
    with torch.no_grad():next(algo.policy.actor.parameters()).add_(1)
    saved=restore_recovery_boundary(algo,record,identity)
    assert algo.parent_anchor_cursor==12
    for k,v in weights.items():assert torch.equal(algo.policy.state_dict()[k],v)
    np.testing.assert_array_equal(saved['route_counters'],np.arange(4))
    with pytest.raises(ValueError,match='cursor mismatch'):
        restore_recovery_boundary(algo,dict(record,parent_anchor_cursor=13),identity)
    with pytest.raises(ValueError,match='hash mismatch'):
        restore_recovery_boundary(algo,dict(record,checkpoint_sha256='wrong'),identity)
