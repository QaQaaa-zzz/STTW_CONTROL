import pytest
from sttw_control.smooth_performance import measure_repeated


def test_measurements_restore_before_every_warmup_and_sample():
    state = [99]
    starts = []
    def restore(): state[0] = 7
    def operation():
        starts.append(state[0]); state[0] += 1
    result = measure_repeated(operation, restore, lambda: None, 2, 3)
    assert starts == [7] * 5
    assert len(result['warmup_seconds']) == 2
    assert len(result['samples_seconds']) == 3


def test_cannot_publish_under_sampled_benchmark():
    with pytest.raises(ValueError):
        measure_repeated(lambda: None, lambda: None, lambda: None, 1, 3)


def test_completed_samples_are_retained_if_later_iteration_fails():
    import copy
    receipts=[];calls=[0]
    def operation():
        calls[0]+=1
        if calls[0]==4:raise RuntimeError('later failure')
    with pytest.raises(RuntimeError,match='later failure'):
        measure_repeated(operation,lambda:None,lambda:None,on_sample=lambda row:receipts.append(copy.deepcopy(row)))
    assert len(receipts)==3
    assert len(receipts[-1]['warmup_seconds'])==2
    assert len(receipts[-1]['samples_seconds'])==1
    assert len(receipts[-1]['intervals'])==3
