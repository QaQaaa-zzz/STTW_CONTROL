import numpy as np
import pytest
from sttw_control.rejoin_comparison import prepare_cases, geometric_task_trace


def test_prepare_cases_preserves_zero_uniqueness_and_pairs_only_tail_rule():
    source=[]
    for role in ['parent','task_only','physical_credit','parent_speed_floor']:
        for context in [1,3]:
            for alpha in [0.,.25,.5,.75,1.]:
                source.append(dict(case_id=f'{role}/{context}/{alpha}',role=role,
                    panel='physical_deterministic',force_N=32,context_index=context,alpha=alpha))
    for context in [1,3]:
        source.append(dict(case_id=f'zero/{context}',role='zero_middle',
            panel='physical_deterministic',force_N=32,context_index=context,alpha=None))
    result=prepare_cases(source)
    assert len(result)==36
    assert len({x['case_id'] for x in result})==36
    assert sum(x['role']=='zero_middle' for x in result)==4
    for i in range(0,36,2):
        a,b=result[i:i+2]
        assert a['tail_rule']=='held' and b['tail_rule']=='live_zero'
        assert {k:v for k,v in a.items() if k not in ('tail_rule','case_id')}=={
            k:v for k,v in b.items() if k not in ('tail_rule','case_id')}


def test_shadow_errors_use_actual_heading_and_keep_real_xy_off_path():
    path=dict(s=np.arange(6.),xy=np.column_stack([np.arange(6.),np.zeros(6)]),
              heading=np.zeros(6),goal=np.array(4.))
    cfg=dict(path=dict(projection_backward_m=.2,projection_forward_floor_m=2.,
                       projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1))
    data=dict(active_tick=np.array([True,True,False]),actual_xy=np.array([[.5,1.],[1.5,-2.],[99.,99.]]),
              actual_yaw=np.array([.3,-.2,99.]))
    result=geometric_task_trace(data,path,cfg,initial_xy=np.zeros(2))
    np.testing.assert_allclose(result['path_cross_track'][:2],[1.,-2.])
    np.testing.assert_allclose(result['path_heading_error'][:2],[-.3,.2])
    assert np.isnan(result['path_cross_track'][2])
    np.testing.assert_allclose(result['path_progress'][:2],[.5,1.5])


def test_missing_or_nonfinite_active_heading_cannot_be_scored_as_recovery():
    p=dict(s=np.arange(3.),xy=np.column_stack([np.arange(3.),np.zeros(3)]),heading=np.zeros(3),goal=2.)
    c=dict(path=dict(projection_backward_m=.2,projection_forward_floor_m=2.,
                    projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1))
    with pytest.raises(ValueError):
        geometric_task_trace(dict(active_tick=np.array([True]),actual_xy=np.zeros((1,2)),actual_yaw=np.array([np.nan])),p,c,initial_xy=np.zeros(2))


def test_shadow_projection_matches_existing_project_with_backward_motion():
    import jax.numpy as j
    from sttw_control.geometric_path import Path,project
    s=np.linspace(0,5,101,dtype=np.float32)
    xy=np.column_stack([s,.1*s*s]).astype(np.float32)
    heading=np.arctan(.2*s).astype(np.float32)
    path=dict(s=s,xy=xy,heading=heading,goal=np.array(4.,np.float32))
    c=dict(path=dict(projection_backward_m=.2,projection_forward_floor_m=.5,
                    projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1))
    points=np.array([[.1,.2],[.4,.2],[.3,-.2],[.7,.3],[1.,.4]],np.float32)
    result=geometric_task_trace(dict(active_tick=np.ones(5,bool),actual_xy=points,
                                   actual_yaw=np.zeros(5)),path,c,initial_xy=np.zeros(2))
    reference=Path(j.array(s),j.array(xy),j.array(heading),j.zeros_like(j.array(s)),j.float32(3),j.float32(4))
    previous=j.float32(0);old=np.zeros(2)
    for tick,point in enumerate(points):
        p=project(reference,j.array(point),previous,j.float32(np.linalg.norm(point-old)),c)
        np.testing.assert_allclose(result['path_progress'][tick],p.progress,atol=2e-6)
        np.testing.assert_allclose(result['path_cross_track'][tick],p.cross_track,atol=2e-6)
        np.testing.assert_allclose(result['path_heading_error'][tick],p.heading,atol=2e-6)
        previous=p.progress;old=point


def test_compiled_domain_dispatch_uses_matching_branch_arguments(monkeypatch):
    import jax
    import jax.numpy as j
    from flax import struct
    from test_live_path_continuation import Env,initial
    from sttw_control.rejoin_comparison import make_comparison_step
    from sttw_control.survival_collection import SurvivalCollectionState
    import sttw_control.physical_continuation as physical_continuation

    @struct.dataclass
    class Governor:
        current_reference: object
    @struct.dataclass
    class Physical:
        failed: object
        controller_memory: object
        yaw_unwrapped: object
        governor: object

    class DispatchEnv(Env):
        def __init__(self):
            super().__init__()
            for k in ['actual_forward_speed','wheel_speed_proxy','phi','phi_dot','actual_delta',
                      'peak_roll','time','yaw_unwrapped']:
                self.zero_log[k]=j.float32(0)
            for k in ['contact_failure','nonfinite','lower_fault']:
                self.zero_log[k]=j.bool_(False)
            self.zero_log.update(actual_xy=j.zeros(2),applied_force_impulse=j.zeros(3),
                nominal=j.zeros(2),offsets=j.zeros(2),filtered_offset=j.zeros(2),governed=j.zeros(2))
        def observation(self,*_):return j.zeros(351),j.zeros(352),j.bool_(False)
        def policy_step(self,s,*_):
            logs=jax.tree.map(lambda v:j.repeat(v[None],4,axis=0),self.zero_log)
            logs.update(scored_tick_reward=j.zeros(4),active_tick=j.ones(4,bool))
            return s.replace(tick=s.tick+4),logs

    env=DispatchEnv()
    def held_factory(_lower_tick,*,dt):
        def advance(s,pulse,time,reference):
            return s.replace(tick=s.tick+1,lower=s.lower+100),dict(
                env.zero_log,active_tick=j.bool_(True),reference=reference)
        return advance
    monkeypatch.setattr(physical_continuation,'make_hold_last_tick',held_factory)
    s=initial(physical=Physical(j.bool_(False),j.float32(17),j.float32(0),Governor(j.zeros(2))))
    carry=SurvivalCollectionState(s,jax.random.PRNGKey(0),j.zeros(2),j.bool_(True))
    step=jax.jit(make_comparison_step(env,None,horizon_s=.04,mode='zero'))
    held,lh=step(carry,None,None,j.float32(1.5),None,j.bool_(False))
    live,ll=step(carry,None,None,j.float32(1.5),None,j.bool_(True))
    assert float(held.state.lower)==423 and float(live.state.lower)==27
    assert int(held.state.tick)==int(live.state.tick)==4
    assert not np.any(lh['live_tracking_active']) and np.all(ll['live_tracking_active'])
