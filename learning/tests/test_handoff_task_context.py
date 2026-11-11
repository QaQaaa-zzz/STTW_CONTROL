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


def test_frenet_motion_is_invariant_to_world_rotation():
    from sttw_control.handoff_task_context import route_context
    xy=np.column_stack((np.linspace(-5,15,81),np.zeros(81)))
    route=dict(xy=xy,heading=np.zeros(81),curvature=np.zeros(81))
    root=np.zeros((1,13),np.float32);root[0,0:2]=[2.,.2];root[0,6]=1.;root[0,7:9]=[2.,.1]
    first=route_context(root,route,.6,[1,0,0,0],representation='frenet')
    angle=.7;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    turned=root.copy();turned[:,:2]=root[:,:2]@rot.T;turned[:,7:9]=root[:,7:9]@rot.T;turned[:,5]=np.sin(angle/2);turned[:,6]=np.cos(angle/2)
    route2=dict(xy=xy@rot.T,heading=np.full(81,angle),curvature=np.zeros(81))
    second=route_context(turned,route2,.6,[1,0,0,0],representation='frenet')
    np.testing.assert_allclose(first,second,atol=2e-7)
    np.testing.assert_allclose(first[0,2:6],[2.,.1,.36,.84],atol=2e-7)


def test_endpoint_search_matches_exhaustive_polyline_projection():
    from sttw_control.handoff_task_context import project_polyline
    rng=np.random.default_rng(7109);xy=np.column_stack((np.linspace(0,20,301),np.sin(np.linspace(0,12,301))))
    points=rng.uniform([-2,-6],[22,6],(300,2));route=dict(xy=xy)
    progress,lateral,tangent,arc=project_polyline(points,route)
    segment=np.diff(xy,axis=0);length=np.linalg.norm(segment,axis=1);delta=points[:,None]-xy[None,:-1]
    t=((delta*segment[None]).sum(-1)/length[None]**2).clip(0,1);residual=delta-t[:,:,None]*segment[None];i=(residual**2).sum(-1).argmin(1);k=np.arange(len(points))
    np.testing.assert_allclose(progress,arc[i]+t[k,i]*length[i],atol=1e-12)
    np.testing.assert_allclose(abs(lateral),abs(-tangent[:,1]*residual[k,i,0]+tangent[:,0]*residual[k,i,1]),atol=1e-12)
