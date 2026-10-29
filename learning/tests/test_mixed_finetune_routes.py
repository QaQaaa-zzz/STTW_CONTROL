"""New coverage and retained route geometry are sampled without simulation."""
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_scenarios import sample_mixed_finetune_route,validate_mixed_finetune_distribution


def test_mixture_covers_both_directions_and_retained_tasks():
    cfg=json.loads((Path(__file__).parents[1]/'configs/h75_low_speed_mixed_finetune.json').read_text())
    validate_mixed_finetune_distribution(cfg)
    generate=jax.jit(jax.vmap(lambda eid:sample_mixed_finetune_route(919101,eid,j.int32(0),j.zeros(3),cfg)))
    r=jax.device_get(generate(j.arange(256)))
    low=r.sample_group==0
    assert .35<low.mean()<.65
    for mask in (low,~low):
        assert set(r.family[mask].tolist())=={0,1,2}
        assert np.any(r.angle[mask]<0) and np.any(r.angle[mask]>0)
    assert r.speed[low].min()>=1.6 and r.speed[low].max()<=2.2
    tight=low&(r.family!=0)
    assert r.radius[tight].min()>=.9 and r.radius[tight].max()<=1.5
    # Named original130-degree turns were outside the old90..100 sampler.
    assert np.any((~low)&np.isclose(abs(r.angle),np.deg2rad(130)))
    assert np.any((~low)&np.isclose(r.radius,2.7))
    assert np.all(np.diff(r.path.s,axis=1)>0)
    assert np.all(np.isfinite(r.path.xy))
    assert np.all(r.path.s[:,-1]>r.path.goal)
    # The adopted S tasks retain their +3m exit-goal extension.
    s=(~low)&(r.family==2)&np.isclose(r.radius,1.8)&np.isclose(r.speed,2.8)
    assert np.any(s)
    np.testing.assert_allclose(r.path.goal[s]-r.path.turn_end[s],7.,atol=2e-6)
