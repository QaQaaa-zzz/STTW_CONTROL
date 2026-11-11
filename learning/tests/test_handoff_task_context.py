import numpy as np
from sttw_control.handoff_task_context import route_context


def test_same_state_distinguishes_mirrored_downstream_geometry():
    x=np.linspace(0,20,801);phase=np.clip((x-12)/4,0,1)
    y=.6*np.sin(np.pi*phase)**2
    heading=np.arctan(np.gradient(y,x));curvature=np.gradient(heading,x)
    root=np.zeros((1,13),np.float32);root[0,0]=11.;root[0,6]=1.
    positive=dict(xy=np.column_stack((x,y)),heading=heading,curvature=curvature)
    negative=dict(xy=np.column_stack((x,-y)),heading=-heading,curvature=-curvature)
    left=route_context(root,positive,.6,[1.,0,0,0])
    right=route_context(root,negative,.6,[1.,0,0,0])
    np.testing.assert_allclose(left[:,:18],-right[:,:18],atol=1e-7)
    np.testing.assert_allclose(left[:,-1],right[:,-1])
    assert left.shape==(1,19) and np.max(left[:,:6])>.5


def test_runtime_requires_and_uses_declared_task_context(tmp_path):
    import torch
    import pytest
    from torch import nn
    from sttw_control.handoff_model import HandoffContinuationModel
    net=nn.Sequential(nn.Linear(58,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
    with torch.no_grad():
        for parameter in net.parameters():parameter.zero_()
        net[0].weight[0,39]=1
        net[2].weight[0,0]=1
        net[4].weight[0,0]=1
    for seed in range(3):
        torch.save(dict(state_dict=net.state_dict(),columns=list(range(58)),mean=np.zeros(58,np.float32),std=np.ones(58,np.float32),target='goal'),tmp_path/f'task_context_{seed}.pt')
    model=HandoffContinuationModel(tmp_path,feature_set='task_context')
    root=np.zeros((1,13),np.float32);obs=np.zeros((1,80),np.float32)
    with pytest.raises(ValueError,match='task-context schema'):model.predict(root,obs,[0.])
    context=np.zeros((1,19),np.float32);before=model.predict(root,obs,[0.],context)['conditional_goal_probability']
    context[0,0]=1.;after=model.predict(root,obs,[0.],context)['conditional_goal_probability']
    assert after[0]>before[0]
