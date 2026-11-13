"""Frozen pendulum-bicycle skill adapter; uses the original JIT runtime.

Requires the read-only DVGC/JIT/src reference on PYTHONPATH. The caller owns
the physical state and observation FIFO; loading or changing the policy does
not reset either. This is not an STTW-to-jump state converter.
"""
from pathlib import Path
import json
import hashlib
from types import SimpleNamespace



class PendulumSkill:
    def __init__(self, registry: Path, skill_id: str):
        import jax
        from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
        from jit_dvgc.config import load_config
        from jit_dvgc.constants import ACTOR_OBSERVATION_SIZE, PRIVILEGED_OBSERVATION_SIZE
        from jit_dvgc.model import load_host_model
        from jit_dvgc.ppo import make_checkpoint_policy

        self.record = json.loads(Path(registry).read_text())["records"][skill_id]["policy"]
        raw = Path(self.record["formal_config"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.record["formal_config_file_sha256"]:
            raise ValueError("Frozen skill configuration changed")
        inputs = json.loads(raw)["inputs"]
        config = load_config(Path(inputs["up_config_path"]), runtime_only=True)
        if config.config_sha256 != inputs["up_config_sha256"]:
            raise ValueError("Frozen actuator configuration changed")
        self.bundle = load_host_model(config)
        identity = CheckpointIdentity(
            self.record["formal_config_sha256"], self.bundle.xml_sha256,
            tuple(self.record["actor_frame_fields"]), tuple(self.record["actor_task_fields"]),
            tuple(self.record["action_order"]),
        )
        payload = load_checkpoint(
            Path(self.record["checkpoint"]), expected=identity
        )
        # Inference only needs dimensions, not the training reset pool or PPO budget.
        dimensions = SimpleNamespace(actor_observation_size=ACTOR_OBSERVATION_SIZE,
            privileged_observation_size=PRIVILEGED_OBSERVATION_SIZE, action_size=4)
        self.payload=payload
        self.policy = jax.jit(make_checkpoint_policy(dimensions, payload, deterministic=True))

    def load_adapted_actor(self, bundle):
        """Load a declared task-adapted export; keep source physics/action mapping."""
        import numpy as np
        import jax
        import jax.numpy as jp
        path=Path(bundle);identity=json.loads((path/'identity.json').read_text())
        raw=(path/'actor.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=identity['artifact_sha256']:
            raise ValueError('Adapted actor export identity mismatch')
        arrays={key:jp.asarray(value) for key,value in dict(np.load(path/'actor.npz')).items()}
        if arrays['mean'].shape!=(76,):raise ValueError('Native Playground actor requires the original 76-D input')
        def policy(observation,key):
            x=(observation['state']-arrays['mean'])/arrays['std']
            for i in range(4):
                x=x@arrays[f'kernel_{i}']+arrays[f'bias_{i}']
                if i<3:x=jax.nn.silu(x)
            return jp.tanh(x[:4]),{}
        self.policy=jax.jit(policy)
        self.record=dict(source_frozen_policy=self.record,adaptation=identity)

    def infer(self, actor_observation, knee_position):
        """Return normalized action and native actuator targets, without stepping.

        Observation is the original 76-D FIFO plus jump signal. Targets are
        steering rad, rear-wheel rad/s, hip rad, knee rad, in that order.
        """
        import jax
        from jax import numpy as jp
        from jit_dvgc.action_mapping import map_action

        observation = jp.asarray(actor_observation, dtype=jp.float32)
        if observation.shape != (76,):
            raise ValueError("Expected the original 76-D skill observation")
        action, _ = self.policy({"state": observation}, jax.random.PRNGKey(0))
        ctrl = map_action(action, jp.asarray(knee_position), self.bundle.action_mapping)
        return action, ctrl

class TorchPendulumObservation:
    """76-D RB observation on the common chassis, sampled at 50 Hz.

    Inputs are physical root-link state (xyzw, world velocities), joint state,
    and root-origin world acceleration. The common chassis COM is at its root
    origin. The virtual RB frame uses the declared pendulum mounting rotation.
    No donor mass, wheel radius, or height is substituted into physical signals.
    Caller supplies previous normalized RB action; ground-mode action history
    needs an explicit convention because the ground controller is a different skill.
    """
    def __init__(self, joint_names, mount_wxyz, count, device='cpu'):
        import torch
        self.torch=torch
        self.indices=[joint_names.index(n) for n in ('steering_joint','hip_joint','knee_joint')]
        self.wheels=[joint_names.index(n) for n in ('frontwheel_joint','rearwheel_joint')]
        self.critic_indices=[joint_names.index(n) for n in ('rearwheel_joint','steering_joint','frontwheel_joint','hip_joint','knee_joint')]
        self.mount=torch.tensor(mount_wxyz[1:]+mount_wxyz[:1],device=device).expand(count,4)
        self.frames=torch.zeros(count,3,25,device=device)
        self.valid_count=torch.zeros(count,device=device,dtype=torch.long)
        self.gravity=torch.tensor([0.,0.,-1.],device=device).expand(count,3)

    def make_frame(self, root, position, velocity, root_acceleration_world,
                   obstacle_relative_x, last_action):
        from .torch_observation import rotate_inverse
        torch=self.torch
        q=root[:,3:7]
        body=lambda v:rotate_inverse(self.mount,rotate_inverse(q,v))
        frame=torch.cat((body(self.gravity),body(root[:,10:13]),
            body(root_acceleration_world-9.81*self.gravity),
            position[:,self.indices],velocity[:,self.indices],
            -velocity[:,self.wheels],last_action,
            root[:,7:8],obstacle_relative_x[:,None],root[:,2:3],
            torch.ones_like(root[:,:1])),dim=-1)
        return frame

    def advance_frame(self, frame, jump_signal):
        torch=self.torch
        self.frames=torch.cat((self.frames[:,1:],frame[:,None]),dim=1)
        self.valid_count=(self.valid_count+1).clamp_max(3)
        self.frames[:,:,-1]=(torch.arange(3,device=frame.device)[None]>=3-self.valid_count[:,None])
        return torch.cat((self.frames.flatten(1),jump_signal[:,None]),dim=-1)

    def advance(self, root, position, velocity, root_acceleration_world,
                obstacle_relative_x, last_action, jump_signal):
        frame=self.make_frame(root,position,velocity,root_acceleration_world,obstacle_relative_x,last_action)
        return self.advance_frame(frame,jump_signal)

    def privileged(self, actor_observation, root, position, velocity, illegal_contact):
        """Original 106-D critic order on a virtual upright root frame.

        The common car's extra passive suspension DOF remains physical but is
        not appended to this fixed-size source critic. Wheel coordinates flip
        with wheel axes, for position as well as velocity. No physics is reset.
        """
        from .torch_observation import rotate_inverse
        torch=self.torch
        q=root[:,3:7];a=q[:,:3];b=self.mount[:,:3]
        xyz=q[:,3:]*b+self.mount[:,3:]*a+torch.cross(a,b,dim=-1)
        w=q[:,3:]*self.mount[:,3:]-(a*b).sum(-1,keepdim=True)
        virtual=torch.cat((xyz,w),-1)
        x,y,z=xyz.unbind(-1);w=w[:,0]
        angles=torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            torch.asin((2*(w*y-z*x)).clamp(-1,1)),
            torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),-1)
        sign=torch.tensor([-1.,1.,-1.,1.,1.],device=root.device)
        qpos=torch.cat((root[:,:3],virtual[:,[3,0,1,2]],position[:,self.critic_indices]*sign),-1)
        qvel=torch.cat((root[:,7:10],rotate_inverse(virtual,root[:,10:13]),velocity[:,self.critic_indices]*sign),-1)
        return torch.cat((actor_observation,qpos,qvel,angles,root[:,7:10],illegal_contact[:,None].float()),-1)

class TorchPendulumSkill:
    """Exact deterministic Brax actor arrays exported for Isaac Lab inference."""
    def __init__(self, bundle, device='cuda:0'):
        import numpy as np
        import torch
        bundle=Path(bundle);self.identity=json.loads((bundle/'identity.json').read_text())
        path=bundle/'actor.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.identity['artifact_sha256']:
            raise ValueError('Pendulum actor identity mismatch')
        data=np.load(path);to=lambda x:torch.as_tensor(x,device=device)
        self.weights=[(to(data[f'kernel_{i}']),to(data[f'bias_{i}'])) for i in range(4)]
        self.mean=to(data['mean']);self.std=to(data['std']);self.mapping=self.identity['mapping']
        self.lo=to(np.asarray(self.mapping['ctrl_min'],dtype=np.float32));self.hi=to(np.asarray(self.mapping['ctrl_max'],dtype=np.float32))

    def infer(self, observation, knee_position, heading=None):
        import torch
        with torch.no_grad():
            if self.identity.get('heading_observation'):
                observation=torch.cat((observation,heading),-1)
            x=(observation-self.mean)/self.std
            for weight,bias in self.weights[:-1]:x=torch.nn.functional.silu(x@weight+bias)
            weight,bias=self.weights[-1];action=torch.tanh((x@weight+bias)[...,:4])
            m=self.mapping
            absolute=lambda a,initial,i:torch.where(a>=0,initial+a*(self.hi[i]-initial),initial+a*(initial-self.lo[i]))
            hip=absolute(action[:,2],m['hip_initial'],2)
            knee=absolute(action[:,3],m['knee_initial'],3) if m['joint_target_semantics']=='keyframe_centered_absolute' else knee_position-action[:,3]*m['knee_target_delta']
            target=torch.stack((action[:,0]*self.hi[0],m['base_rear_speed']+action[:,1]*m['rear_speed_delta'],hip,knee),dim=-1)
            return action,torch.maximum(self.lo,torch.minimum(self.hi,target))

class TorchPendulumGeometry:
    """World-x support of the actual primitive collision geometry (MuJoCo types)."""
    def __init__(self, geometry, body_names, device='cuda:0'):
        import torch
        self.indices=[body_names.index(g['body']) for g in geometry]
        self.kind=torch.tensor([g['type'] for g in geometry],device=device)
        self.size=torch.tensor([g['size'] for g in geometry],device=device)
        self.pos=torch.tensor([g['pos'] for g in geometry],device=device)
        self.quat=torch.tensor([g['quat'][1:]+g['quat'][:1] for g in geometry],device=device)

    def frontmost_x(self, bodies):
        import torch
        from .torch_observation import rotate_inverse
        state=bodies[:,self.indices];q=state[...,3:7]
        direction=torch.zeros_like(state[...,:3]);direction[...,0]=1.
        local=rotate_inverse(q,direction)
        center=state[...,0]+(local*self.pos).sum(-1)
        axis=rotate_inverse(self.quat.expand_as(q),local)
        # Common model collision types: box=6, ellipsoid=4, cylinder=5.
        box=(self.size*axis.abs()).sum(-1)
        ellipsoid=((self.size*axis).square().sum(-1)).sqrt()
        cylinder=self.size[:,1]*axis[...,2].abs()+self.size[:,0]*axis[...,:2].square().sum(-1).sqrt()
        radius=torch.where(self.kind==6,box,torch.where(self.kind==4,ellipsoid,cylinder))
        return (center+radius).amax(-1)
