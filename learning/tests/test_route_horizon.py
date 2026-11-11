import numpy as np
from sttw_control.route_horizon import nominal_horizon,remaining_route_length

def test_remaining_length_uses_goal_entry_and_projects_current_position():
    roads=np.array([[[0.,0.],[5.,0.],[10.,0.],[15.,0.]]]*2)
    root=np.zeros((1,13));root[0,0]=2.
    goal=dict(x_min_m=10.,x_max_m=12.,abs_y_max_m=.5)
    np.testing.assert_allclose(remaining_route_length(root,roads,goal),[[8.,8.]])
    p=np.zeros((2,8));p[:,5]=[1.,2.]
    allowed,time=nominal_horizon(root,roads,p,goal,1.,6.)
    assert allowed.tolist()==[[False,True]]
    p[0,-1]=1
    assert nominal_horizon(root,roads,p,goal,1.,6.)[0].tolist()==[[True,True]]
