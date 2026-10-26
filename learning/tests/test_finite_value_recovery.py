"""Resume only retained complete boundaries, using original global clocks."""
import copy
import pytest
from sttw_control.finite_value_recovery import validate_boundary, remaining_collections


def state():
    return dict(schema='sttw_finite_value_boundary_v1',exact_boundary_resume=True,
                next_collection=17,actor_steps=60,critic_steps=68,optimizer_steps=[60,68,68],
                settings=dict(epochs=4))


def test_boundary17_retains_adam_counters_and_skips_completed17_collections():
    validate_boundary(state(),17)
    schedule=[[dict(collection_index=i)] for i in range(64)]
    selected=remaining_collections(schedule,17)
    assert len(selected)==47 and selected[0]==(17,[dict(collection_index=17)])
    assert selected[-1]==(63,[dict(collection_index=63)])
    assert all(i>=17 for i,_ in selected)


@pytest.mark.parametrize('change',[dict(next_collection=16),dict(actor_steps=64),
    dict(optimizer_steps=[60,72,68]),dict(exact_boundary_resume=False),dict(schema='diagnostic_only')])
def test_partial_or_mismatched_boundary_is_not_resumed(change):
    s=state();s.update(change)
    with pytest.raises(ValueError):validate_boundary(s,17)


def test_missing_original_collection_clock_prevents_reseeded_resume():
    schedule=[[dict(collection_index=i)] for i in range(64)]
    schedule[18][0]['collection_index']=0
    with pytest.raises(ValueError):remaining_collections(schedule,17)
    with pytest.raises(ValueError):remaining_collections(schedule[:63],17)


def test_recovery_uses_latest_completed_boundary_instead_of_selecting_an_older_actor():
    from sttw_control.finite_value_recovery import validate_resume_source
    source=dict(phase='error',failure_kind='host_reboot',completed_updates=17,declared_updates=64)
    validate_resume_source(source,17)
    for boundary in [16,18]:
        with pytest.raises(ValueError):validate_resume_source(source,boundary)
    source['phase']='running'
    with pytest.raises(ValueError):validate_resume_source(source,17)
