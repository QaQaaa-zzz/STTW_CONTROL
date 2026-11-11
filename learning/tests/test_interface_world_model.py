import numpy as np
import torch
import pytest
from sttw_control.interface_world_model import InterfaceWorldModel


@pytest.mark.parametrize("independent_ground", [False, True])
def test_composition_uses_conditional_goal_once_and_separate_ground_branch(independent_ground):
    class Map:
        def crop(self, root):
            return np.zeros((len(root),128),np.float32)
    class Member:
        def __call__(self, features):
            return torch.zeros((len(features),80))
    class Downstream:
        def predict(self, root, observation, time):
            return {'conditional_goal_probability':np.full(len(root),.4)}
    model=InterfaceWorldModel.__new__(InterfaceWorldModel)
    model.ground=[];model.map=Map();model.members=[Member()]*3;model.downstream=Downstream()
    mean=np.zeros(39,np.float32);mean[6]=1
    model.c=dict(input_mean=np.zeros(582,np.float32),input_std=np.ones(582,np.float32),target_mean=mean,target_std=np.zeros(39,np.float32))
    if independent_ground:
        ground=torch.nn.Linear(582,1)
        torch.nn.init.zeros_(ground.weight)
        torch.nn.init.constant_(ground.bias,float(np.log(.8/.2)))
        model.ground=[(ground,np.zeros(582,np.float32),np.ones(582,np.float32))]*3
    root=np.zeros((2,13),np.float32);obs=np.zeros((2,80),np.float32);history=np.zeros((2,352),np.float32);params=np.zeros((2,8),np.float32);params[0,-1]=1
    result=model.score(root,obs,history,np.zeros(2),params,particles=4)
    np.testing.assert_allclose(result['score'],[.2,.8 if independent_ground else .5],atol=1e-7)
    np.testing.assert_allclose(result['valid_handoff_probability'],.5)
