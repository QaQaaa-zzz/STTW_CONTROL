"""TRAIN coverage must preserve ordinary and unselected complete routes."""
import copy
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.path_command_scenarios import sample_curriculum_route,validate_recovery_coverage

CFG={'path':{'spacing_m':.02,'curvature_transition_length_m':.5,'straight_prefix_m':3.,'goal_section_offset_after_turn_m':4.},'curriculum':{'previous_fraction':.25,'stages':[{'speed_m_s':[2.,2.4],'radius_m':[2.2,2.6]},{'speed_m_s':[2.4,2.6],'radius_m':[2.,2.2]},{'speed_m_s':[2.6,2.8],'radius_m':[1.8,2.]}]}}
COVERAGE={'turn_fraction':.5,'seed':918401,'speed_m_s':[2.2,2.45],'radius_m':[2.3,2.7],'angle_deg':[105.,130.]}

def batch(coverage):
    ids=j.asarray(np.linspace(0,511,64,dtype=np.int32))
    return jax.jit(jax.vmap(lambda eid:sample_curriculum_route(918201,eid,741,j.array([2.,1.,.1]),CFG,2,coverage)))(ids)

def test_default_and_disabled_preserve_full_routes():
    before=copy.deepcopy(CFG)
    default=batch(None);disabled=batch(dict(COVERAGE,turn_fraction=0.))
    for a,b in zip(jax.tree.leaves(default),jax.tree.leaves(disabled)):
        np.testing.assert_array_equal(a,b)
    assert CFG==before


def test_mixture_changes_only_selected_turn_geometry_and_speed():
    original=batch(None);mixed=batch(COVERAGE)
    selected=np.rad2deg(np.abs(np.asarray(mixed.angle)))>100
    turns=np.asarray(original.family)!=0
    assert 0<selected.sum()<turns.sum()
    assert not selected[~turns].any()
    for a,b in zip(jax.tree.leaves(original),jax.tree.leaves(mixed)):
        np.testing.assert_array_equal(np.asarray(a)[~selected],np.asarray(b)[~selected])
    for name,field in [('speed_m_s','speed'),('radius_m','radius'),('angle_deg','angle')]:
        values=np.asarray(getattr(mixed,field))[selected]
        if field=='angle':values=np.rad2deg(np.abs(values))
        lo,hi=COVERAGE[name];assert np.all((values>=lo)&(values<=hi))
    for field in ['family','sample_group','lateral_offset','heading_offset']:
        np.testing.assert_array_equal(getattr(original,field),getattr(mixed,field))
    np.testing.assert_array_equal(np.sign(original.angle),np.sign(mixed.angle))
    for a,b in zip(jax.tree.leaves(mixed),jax.tree.leaves(batch(COVERAGE))):
        np.testing.assert_array_equal(a,b)
    # Geometry and completion cross-section actually move with the declared angle.
    assert np.any(np.asarray(mixed.path.xy)[selected]!=np.asarray(original.path.xy)[selected])
    assert np.any(np.asarray(mixed.path.goal)[selected]!=np.asarray(original.path.goal)[selected])


@pytest.mark.parametrize('override',[{'turn_fraction':1.1},{'speed_m_s':[2.4,2.2]},{'radius_m':[0.,1.]},{'angle_deg':[105.,float('nan')]},{'seed':-1},{'typo':1}])
def test_invalid_coverage_rejected(override):
    validate_recovery_coverage(None);validate_recovery_coverage(COVERAGE)
    with pytest.raises(ValueError):validate_recovery_coverage(dict(COVERAGE,**override))
