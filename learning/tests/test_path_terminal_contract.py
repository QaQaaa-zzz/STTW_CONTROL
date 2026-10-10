"""Only terminal semantics, no vehicle simulation."""
import json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from flax import struct
from sttw_control.path_command_training import termination_flags,reset_finished_episode
from sttw_control.path_command_env import PathCommandEnv
from sttw_control.path_command_reward import failure_reward
CFG=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())

@struct.dataclass
class Physical:
 failed: object
@struct.dataclass
class State:
 physical: object
 fault: object
 domain_exit: object
 offsets: object
 tick: object
 trigger: object
@struct.dataclass
class Episode:
 state: object
 env_id: object
 episode: object

def minimal_env():
 env=PathCommandEnv.__new__(PathCommandEnv);env.config=CFG;env.cc=SimpleNamespace(dt=.005);env.terminal_contract='terminal_contract_v2';env.record_frame=lambda s:s
 env.zero_log=dict(active_tick=j.bool_(False),tick_reward=j.array(0.),raw_components={'offset_rate':j.array(0.)},effective_components={'offset_rate':j.array(0.)},component_capped={'offset_rate':j.bool_(False)})
 def tick(s,z,path,speed):
  return s.replace(tick=s.tick+1,domain_exit=s.trigger),{**env.zero_log,'active_tick':j.bool_(True),'tick_reward':j.array(-.01)}
 env.tick=tick;return env

def test_two_environment_domain_terminal_resets_only_self_and_penalty_once():
 env=minimal_env()
 def run(eid):
  s=State(Physical(j.bool_(False)),j.bool_(False),j.bool_(False),j.zeros(2),j.int32(100),eid==0)
  end,logs=env.policy_step(s,j.zeros(2),None,2.,20.)
  flags=termination_flags(end.tick,end.physical.failed,end.fault,end.domain_exit,20.,CFG)
  ep=Episode(end,eid,j.int32(3))
  reset=lambda eid,k:Episode(s.replace(tick=j.int32(0),domain_exit=j.bool_(False)),eid,k)
  return reset_finished_episode(ep,flags,reset),flags,logs
 ep,flags,logs=jax.jit(jax.vmap(run))(j.arange(2))
 np.testing.assert_array_equal(ep.state.tick,[0,104]);np.testing.assert_array_equal(ep.episode,[4,3])
 np.testing.assert_array_equal(flags['tracking_domain_failure'],[True,False]);np.testing.assert_array_equal(flags['bootstrap'],[False,True])
 np.testing.assert_array_equal(logs['active_tick'].sum(1),[1,4])
 np.testing.assert_allclose(logs['scored_tick_reward'][0].sum(),failure_reward(975,CFG),rtol=2e-6)
 assert np.count_nonzero(logs['scored_tick_reward'][0])==1
 np.testing.assert_allclose(logs['scored_tick_reward'][1].sum(),-.04)
 both=termination_flags(101,True,False,True,20.,CFG);assert int(both['end_code'])==1
 # Nonfinite upper action is still engineering fault, never a domain reset.
 s=State(Physical(j.bool_(False)),j.bool_(False),j.bool_(False),j.zeros(2),j.int32(0),j.bool_(False))
 end,_=env.policy_step(s,j.array([j.nan,0.]),None,2.,20.)
 fault=termination_flags(end.tick,False,end.fault,end.domain_exit,20.,CFG)
 assert bool(fault['engineering_fault']) and not bool(fault['tracking_domain_failure'])
 assert int(end.tick)==0
