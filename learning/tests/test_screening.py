import json
import pytest
from sttw_control.env import load_config
from sttw_control.screening import scenarios


def test_screening_preserves_physics_and_initial_circle_tangent():
    c=load_config('learning/configs/disturbance_learning.json')
    p=json.load(open('learning/configs/authority_screening.json'))
    items=list(scenarios(c,p))
    assert len(items)==18 and len({name for name,_ in items})==18
    for _,x in items:
        assert x.circle.center_y==x.circle.direction*x.circle.radius
        assert x.actuator==c.actuator and x.controller==c.controller
        assert x.random_events is None and x.action_mapping is None
    with pytest.raises(ValueError):list(scenarios(c,{**p,'speeds':[-1]}))
