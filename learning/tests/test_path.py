import numpy as np
import pytest
import jax.numpy as jp
from sttw_control.path import CircleConfig, circle_command, circle_reference


def test_circle_geometry_and_feedback_direction():
    c=CircleConfig(radius=3.,lookahead=1.)
    xy=circle_reference(c,101)
    np.testing.assert_allclose(np.linalg.norm(xy-np.array([0,3]),axis=1),3.,atol=1e-6)
    np.testing.assert_allclose(xy[[0,-1]],[[0,0],[0,0]],atol=1e-6)
    nominal=float(circle_command(jp.array([0.,0.,0.]),c,.408,25*np.pi/180))
    outward=float(circle_command(jp.array([0.,-.2,0.]),c,.408,25*np.pi/180))
    inward=float(circle_command(jp.array([0.,.2,0.]),c,.408,25*np.pi/180))
    assert inward<nominal<outward<c.max_steer
    assert nominal>0


def test_circle_config_rejects_invalid_dimensions():
    with pytest.raises(ValueError): CircleConfig(radius=0.)
    with pytest.raises(ValueError): CircleConfig(lookahead=float('nan'))
