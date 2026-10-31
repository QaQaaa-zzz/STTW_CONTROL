"""Pulse clock/reset/mixture contracts and a twelve-tick CPU wiring check."""
import copy,json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.lower_command import LowerCommandEnv,episode_force_pulse,observation_from_history,command_rows


def config():return json.loads((Path(__file__).parents[1]/'configs/lower_pulse_finetune.json').read_text())


def test_sampler_is_reproducible_signed_mixture_and_uses_physical_origin():
    c=config();ids=j.arange(1024)
    f=jax.jit(jax.vmap(lambda i:episode_force_pulse(i,0,3.5,c)))(ids)
    again=jax.vmap(lambda i:episode_force_pulse(i,0,3.5,c))(ids)
    np.testing.assert_allclose(f,again,rtol=1e-6,atol=2e-6)
    a=np.asarray(f);assert np.all((a[:,0]>=4.5)&(a[:,0]<=9.5))
    np.testing.assert_allclose(a[:,1]-a[:,0],.15,atol=1e-6)
    assert 400<np.sum(a[:,3]==0)<624 and np.any(a[:,3]<0) and np.any(a[:,3]>0)
    assert np.all((abs(a[:,3])==0)|((abs(a[:,3])>=24)&(abs(a[:,3])<=36)))
    np.testing.assert_array_equal(a[:,[2,4]],0)
    assert not np.array_equal(a,np.asarray(jax.vmap(lambda i:episode_force_pulse(i,1,3.5,c))(ids)))
    shifted=episode_force_pulse(3,2,7.,c);old=episode_force_pulse(3,2,3.5,c)
    np.testing.assert_allclose(np.asarray(shifted[:2]-old[:2]),3.5,atol=1e-6)
    np.testing.assert_array_equal(shifted[2:],old[2:])


def test_no_schedule_leaks_into_observation_or_changes_command_rng():
    c=config();old=copy.deepcopy(c);old.pop('episode_disturbance')
    for a,b in zip(command_rows(1,2,2.3,c),command_rows(1,2,2.3,old)):np.testing.assert_array_equal(a,b)
    for a,b in zip(observation_from_history(j.ones((10,20)),j.ones(10),0,c),observation_from_history(j.ones((10,20)),j.ones(10),0,old)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('key,value',[('probability',1.1),('duration_s',float('nan')),('onset_s',[9.9,10.]),('magnitude_N',[-24.,36.]),('frame','bodyY')])
def test_invalid_configuration_rejected_before_model_construction(key,value):
    c=config();c['episode_disturbance'][key]=value
    with pytest.raises(ValueError,match='disturbance'):LowerCommandEnv(c)


def test_cpu_step_clears_force_and_reset_resamples_without_history_leak():
    c=config();env=LowerCommandEnv(c,backend='cpu');snap=env.physics.initial().replace(physical_tick=j.int32(700))
    baseline=None
    for force in [0.,24.,-24.,None]:
        s=env.reset(snap,j.int32(0),j.int32(0));old_s=s
        pulse=None if force is None else j.array([3.505,3.510,0.,force,0.])
        s=s.replace(force_pulse=pulse);impulses=[]
        for _ in range(3):
            s,_,_,log=env.step(s,j.zeros(2));impulses.append(np.asarray(log['applied_force_impulse']))
        np.testing.assert_allclose(np.asarray(impulses)[:,1],[0.,0. if force is None else force*.005,0.],atol=2e-5)
        if force==0:baseline=np.asarray(s.physical.data.qpos).copy()
        if force is None:np.testing.assert_allclose(s.physical.data.qpos,baseline,atol=1e-7)
        fresh=env.reset(snap,j.int32(0),j.int32(1))
        assert int(fresh.tick)==0 and int(fresh.physical.physical_tick)==700
        assert int(fresh.mask.sum())==1 and np.all(np.asarray(fresh.frames[:-1])==0)
        assert not np.array_equal(fresh.force_pulse,old_s.force_pulse)
