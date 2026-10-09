"""Device accumulators; episode events are emitted for every policy step."""
import jax
import jax.numpy as jp


def initialize(n, fields, components):
    return dict(episode_return=jp.zeros(n),episode_steps=jp.zeros(n,jp.int32),
                episode_stats=jp.zeros((n,fields)),negative_dwell=jp.zeros((n,2,2),jp.int32),
                negative_seen=jp.zeros((n,2,2),bool),negative_covered=jp.zeros((2,2),jp.int32),
                coverage_completed=jp.int32(0),sums=jp.zeros((4,fields)),family_sums=jp.zeros((4,fields)),
                reward_sum=jp.float32(0),fails=jp.int32(0),ends=jp.int32(0),
                capcounts=jp.zeros(components,jp.int32),capden=jp.int32(0))


def begin_rollout(carry):
    """Clear batch summaries while retaining all unfinished episode history."""
    carry=dict(carry)
    for k in ('sums','family_sums','reward_sum','fails','ends','capcounts','capden'):
        carry[k]=jp.zeros_like(carry[k])
    return carry


def accumulate(carry,reward,done,failed,stat,family,capcounts,active,proposal,governed,valid):
    c=dict(carry)
    returns=c['episode_return']+reward;steps=c['episode_steps']+1;stats=c['episode_stats']+stat[:,0]
    dwell,seen=c['negative_dwell'],c['negative_seen']
    for tick in range(4):
        values=jp.stack((proposal,governed[:,tick]),axis=1)
        # Preserve NumPy's former float32-to-float64 strict comparison:
        # these two decimal boundaries round downward in float32.
        mask=(values[:,:,None]<=jp.array([-.15,-.30])) & valid[:,tick,None,None]
        dwell=jp.where(mask,dwell+1,0);seen=seen | (dwell>=40)
    event=dict(done=done,return_sum=returns,steps=steps,failed=failed,stats=stats)
    # Both family and episode attribution are from before reset.
    c['sums']=c['sums']+stat.sum(axis=0)
    # DEFAULT GPU dot precision can round inputs below float32 accuracy.
    # Statistics must conserve the same values as the ordinary sum reduction.
    c['family_sums']=c['family_sums']+jp.matmul(jax.nn.one_hot(family,4).T,stat[:,0],precision=jax.lax.Precision.HIGHEST)
    c['reward_sum']=c['reward_sum']+reward.sum();c['fails']=c['fails']+failed.sum();c['ends']=c['ends']+done.sum()
    c['capcounts']=c['capcounts']+capcounts;c['capden']=c['capden']+active
    c['negative_covered']=c['negative_covered']+(seen & done[:,None,None]).sum(axis=0)
    c['coverage_completed']=c['coverage_completed']+done.sum()
    c['episode_return']=jp.where(done,0.,returns);c['episode_steps']=jp.where(done,0,steps)
    c['episode_stats']=jp.where(done[:,None],0.,stats)
    c['negative_dwell']=jp.where(done[:,None,None],0,dwell)
    c['negative_seen']=seen & ~done[:,None,None]
    return c,event
