"""Opt-in goal continuation around an explicitly supplied native environment.

This module can be loaded by file without importing the modern control package.
The supplied native methods retain physics, histories, reference filters and
controller evolution. Only the legacy stop reasons and target publication
permission change. This is a new task contract, not a native replay.
"""
import jax
import jax.numpy as j
from flax import struct


@struct.dataclass
class ContinuationCarry:
    state: object
    ever_domain_exit: object
    clean_blocks: object


def target_permission(carry, *, zero_mode=False):
    """Same existing sixteen-complete-clean-block handback convention."""
    s=carry.state
    stopped=s.physical.failed|s.fault|s.follower.numerical_fault
    return ((~carry.ever_domain_exit|(carry.clean_blocks>=16))
            &~s.follower.invalid&~stopped&~j.asarray(zero_mode,bool))


class NativeGoalContinuation:
    """Install only on a freshly loaded, dedicated native env instance.

    `contact_only_physics`, `project` and `pursuit` are explicit dependencies:
    the caller must bind their source identities rather than mix package paths.
    Native `domain_exit` is no longer a stopping latch. Its cumulative meaning
    lives in the carry and logs; the actual instantaneous flags remain intact.
    """
    def __init__(self,env,*,contact_only_physics,project,pursuit):
        if getattr(env,'_goal_continuation_installed',False):
            raise ValueError('dedicated native environment already adapted')
        self.env=env
        self._native_tick=env.tick
        self._project=project
        self._pursuit=pursuit
        env.physics=contact_only_physics(env.physics)
        env.tick=self.tick
        env._goal_continuation_installed=True

    def __getattr__(self,name):
        return getattr(self.env,name)

    def initial(self,state):
        # Initial numerical faults are NOT cleared. Invalid geometry may be
        # observed, but cannot authorize a target or masquerade as a new reset.
        ever=state.domain_exit|state.follower.invalid
        return ContinuationCarry(state.replace(domain_exit=j.bool_(False)),
                                 j.asarray(ever,bool),j.int32(0))

    def tick(self,state,z,path,v_user,force_pulse=None,horizon=None):
        # Native policy_step stops on this latch. Move its meaning to telemetry
        # at the point of state production, not by relabeling frozen trajectories.
        end,log=self._native_tick(state.replace(domain_exit=j.bool_(False)),z,path,v_user,
                                  force_pulse=force_pulse,horizon=horizon)
        # Native pursuit COMMANDS update only every four lower ticks. Query
        # validity read-only at each actual post-tick pose; never publish this
        # extra query or change native follower/nominal/filter evolution.
        _,pose,speed=self.env.physics.observe(end.physical.data)
        validity=self._pursuit(path,end.projection.progress,pose,speed,v_user,self.env.config)
        log=dict(log,continuation_geometry_invalid=validity.invalid,
                 continuation_geometry_fault=validity.numerical_fault)
        return end.replace(domain_exit=j.bool_(False),
                           fault=end.fault|validity.numerical_fault),log

    def policy_step(self,carry,path,v_user,horizon,target,force_pulse=None,*,zero_mode=False):
        """One original four-lower-tick block with a common publication mask.

        `target(observation, state)` runs only on the permitted JAX branch.
        Current invalidity blocks the next upper publication. A held target
        within an already issued block remains held, as in native control.
        Observation validity is checked even when target inference is skipped.
        """
        state=carry.state
        obs,_,observation_fault=self.env.observation(state,path,v_user,horizon)
        state=state.replace(fault=state.fault|observation_fault|state.follower.numerical_fault,
                            domain_exit=j.bool_(False))
        current=carry.replace(state=state)
        allowed=target_permission(current,zero_mode=zero_mode)
        z=jax.lax.cond(allowed,lambda _:target(obs,state),
                       lambda _:j.zeros_like(state.filtered),None)
        end,logs=self.env.policy_step(state,z,path,v_user,horizon,force_pulse)
        active=logs['active_tick']
        invalid=logs['continuation_geometry_invalid']
        initial_invalid=state.follower.invalid
        ever=carry.ever_domain_exit|initial_invalid|j.any(active&invalid)|end.follower.invalid
        clean=(j.all(active)&~initial_invalid&~j.any(active&invalid)
               &~end.follower.invalid&~end.physical.failed&~end.fault)
        count=j.where(clean,j.minimum(carry.clean_blocks+1,16),0).astype(j.int32)
        cumulative=carry.ever_domain_exit|initial_invalid|(j.cumsum(active&invalid)>0)
        logs=dict(logs,reference_target_permitted=j.full_like(active,allowed),
                  cumulative_domain_exit=cumulative,
                  clean_blocks_at_publication=j.full(active.shape,carry.clean_blocks,j.int32),
                  clean_blocks_at_end=j.full(active.shape,count,j.int32))
        return ContinuationCarry(end,ever,count),logs

    def rebind_at_upper_boundary(self,carry,path,v_user):
        """Host-side geometry update between complete upper blocks only.

        Preserve all sidecar history, including clean counts from COMPLETE
        earlier blocks. A rebind contributes zero new clean blocks and never
        clears an exit record. Current validity is rechecked before publication.
        No mid-block geometry switch is permitted by this adapter.
        """
        s=carry.state
        if int(s.tick)%4:
            raise ValueError('geometry rebind requires an upper-block boundary')
        return self.rebind_geometry(carry,path,v_user)

    def rebind_geometry(self,carry,path,v_user):
        """JAX-compatible binding; caller must certify the upper boundary.

        This supports batched rebinding after a host-side boundary/mask check,
        without dispatching one series of small GPU kernels per changed lane.
        """
        s=carry.state
        _,pose,speed=self.env.physics.observe(s.physical.data)
        projection=self._project(path,pose[:2],j.float32(0),j.float32(0),self.env.config)
        follower=self._pursuit(path,projection.progress,pose,speed,v_user,self.env.config)
        # Numerical invalidity must latch even if no physics tick is performed.
        rebound=s.replace(projection=projection,follower=follower,
                          fault=s.fault|follower.numerical_fault)
        return carry.replace(state=rebound)
