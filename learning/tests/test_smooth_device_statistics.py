import numpy as np
import jax
import jax.numpy as jp
from sttw_control.smooth_command_training import SmoothCampaign
from sttw_control.smooth_device_statistics import initialize, accumulate


def test_episode_events_dwell_and_cross_rollout_match_reference():
    n=2;nf=16
    ref=object.__new__(SmoothCampaign);ref.n=n;ref.training_stage=51;ref.endpoints={0:{}}
    carry=initialize(n,nf,2);fn=jax.jit(accumulate)
    events=[]
    for step in range(24):
        r=np.array([.1,.2],np.float32);d=np.array([step in (10,12,23),step==23]);failed=d & (step==12)
        stat=np.ones((n,4,nf),np.float32);stat[:,:,0]=4
        diag={'all_proposal_dv':np.full(n,-.4,np.float32),'all_governed_dv':np.full((n,4),-.4,np.float32),'all_valid_ticks':np.ones((n,4),bool)}
        ref.record_episode_statistics(0,r,d,failed,stat,diag)
        carry,event=fn(carry,jp.asarray(r),jp.asarray(d),jp.asarray(failed),jp.asarray(stat),jp.array([0,3]),jp.array([1,2]),jp.int32(8),*[jp.asarray(diag[k]) for k in ('all_proposal_dv','all_governed_dv','all_valid_ticks')])
        event=jax.device_get(event)
        for i in np.flatnonzero(event['done']):events.append((event['return_sum'][i],event['steps'][i],event['failed'][i]))
        if step==15: # batch boundary must not discard unfinished episodes
            assert int(carry['episode_steps'][1])==16
    ep=ref.endpoints[0];out=jax.device_get(carry)
    assert len(events)==4
    for got,want in zip(events,ep['completed_episode_batch']):
        np.testing.assert_allclose(got[0],want['return_sum'],rtol=1e-6)
        assert got[1:]==(want['policy_steps'],want['physical_failure'])
    for k in ('episode_return','episode_steps','episode_stats','negative_dwell','negative_seen','negative_covered','coverage_completed'):
        np.testing.assert_allclose(out[k],ep[k],rtol=1e-6)
    assert out['family_sums'][0,0]==24*4 and out['family_sums'][3,0]==24*4


def test_delayed_reset_events_keep_multiple_resets_and_true_initial_pose(tmp_path):
    ref=object.__new__(SmoothCampaign);ref.out=tmp_path;ref.seen_cases=set()
    from types import SimpleNamespace
    ref.env=SimpleNamespace(lower=object())
    events=dict(done=np.ones((3,2),bool),env_id=np.tile([0,1],(3,1)),episode_index=np.repeat(np.arange(1,4)[:,None],2,axis=1),
                alpha=np.zeros((3,2)),family=np.ones((3,2)),initial_heading_error=np.zeros((3,2)),
                reference_pose=np.arange(18).reshape(3,2,3),actual_yaw=np.arange(6).reshape(3,2),
                slew=np.ones((3,2,2)),rows=np.zeros((3,2,16,3)))
    ref.write_reset_events(events,'alpha0')
    import json
    rows=[json.loads(x) for x in (tmp_path/'alpha0/case_manifest.jsonl').read_text().splitlines()]
    assert len(rows)==6
    assert rows[0]['initial_reference_pose']==[0,1,2]
    assert rows[-1]['episode_index']==3
    ref.write_reset_events(events,'alpha0')
    assert len((tmp_path/'alpha0/case_manifest.jsonl').read_text().splitlines())==6


def test_float32_threshold_boundary_matches_original_float64_comparison():
    # Original NumPy comparison promotes the thresholds to float64. Both
    # float32 -.15 and -.30 round below the exact decimal threshold.
    for threshold in (-.15,-.30):
        carry=initialize(1,16,1)
        fn=jax.jit(accumulate)
        for step in range(10):
            carry,_=fn(carry,jp.ones(1),jp.zeros(1,bool),jp.zeros(1,bool),jp.ones((1,4,16)),jp.zeros(1,jp.int32),jp.zeros(1,jp.int32),jp.int32(4),jp.full(1,threshold),jp.full((1,4),threshold),jp.ones((1,4),bool))
        expected=np.float32(threshold)<np.array([-.15,-.30])
        np.testing.assert_array_equal(np.asarray(carry['negative_seen'])[0],np.broadcast_to(expected,(2,2)))


def test_gpu_family_reduction_conserves_float32_statistics():
    import pytest
    if jax.default_backend()!='gpu':pytest.skip('requires actual GPU dot precision')
    rng=np.random.default_rng(87);n=512;fields=46
    stat=rng.uniform(.01,400,(n,4,fields)).astype(np.float32);stat[:,:,0]=4
    family=rng.integers(0,4,n,dtype=np.int32)
    carry,_=jax.jit(accumulate)(initialize(n,fields,16),jp.zeros(n),jp.zeros(n,bool),jp.zeros(n,bool),
        jp.asarray(stat),jp.asarray(family),jp.zeros(16,jp.int32),jp.int32(n*4),jp.zeros(n),jp.zeros((n,4)),jp.ones((n,4),bool))
    expected=np.stack([stat[family==i,0].astype(np.float64).sum(0) for i in range(4)])
    np.testing.assert_allclose(np.asarray(carry['family_sums']),expected,rtol=2e-6,atol=1e-5)
