import numpy as np
import pytest
from sttw_control.command_comparison import reference_pose, generate_comparison


def test_exact_circle_and_straight_reference():
    circle=reference_pose([[0,2,1]], [0,np.pi,2*np.pi], [0,0,0])
    np.testing.assert_allclose(circle[-1], [0,0,2*np.pi],atol=1e-12)
    np.testing.assert_allclose(circle[1,:2], [0,4],atol=1e-12)
    straight=reference_pose([[0,2,0]], [0,3], [1,2,np.pi/2])
    np.testing.assert_allclose(straight[-1], [1,8,np.pi/2])


def test_switch_inside_sample_and_nonzero_initial_heading():
    actual=reference_pose([[0,1,0],[.3,2,0]], [0,1], [0,0,0])
    assert actual[-1,0]==pytest.approx(1.7)
    with pytest.raises(ValueError):reference_pose([[1,1,0]], [0,1], [0,0,0])


def test_incomplete_panels_rejected(tmp_path):
    with pytest.raises(ValueError,match='coverage'):
        generate_comparison({'one':tmp_path/'one','two':tmp_path/'two'},tmp_path/'plots')


def test_saved_comparison_preserves_failure_and_rewards(tmp_path):
    import json
    runs={}
    for label,failed in [('first',False),('second',True)]:
        root=tmp_path/label;runs[label]=root
        for policy in ('baseline','residual'):
            d=root/'evaluation/alpha_0/seed_1/case'/policy;d.mkdir(parents=True)
            failure=failed and policy=='residual'
            t=np.array([0.,.5]) if failure else np.array([0.,.5,1.])
            reward=np.zeros(len(t));reward[1:]=-.1
            if failure:reward[-1]=-100
            np.savez(d/'trace.npz',time=t,pose=np.column_stack([t,t*0,t*0]),reward=reward,
                     priority_alpha=t*0,terminated=np.array([False]*(len(t)-1)+[failure]))
            (d/'commands.json').write_text(json.dumps({'schedule':[[0,1,0,0]]}))
            (d/'declaration.json').write_text(json.dumps({'config':{'horizon_seconds':1,'controller':{'dt':.5}}}))
    out=tmp_path/'plots';entries=generate_comparison(runs,out)
    assert entries[0]['endpoints'][-1]['end_seconds']==.5
    assert entries[0]['endpoints'][-1]['observed_return']==-100
    data=np.load(out/'case_seed_1.npz')
    assert len(data['alpha_0_reference'])==3
    assert len(data['alpha_0_method_2_reward'])==2
    assert (out/'case_seed_1_reward_steps.pdf').exists()
