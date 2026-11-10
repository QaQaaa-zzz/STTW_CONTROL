"""Optional measured landing handback to the existing H75 ground stack."""
import json
from pathlib import Path
import numpy as np
import torch


class DitchGroundHandoff:
    def __init__(self, env, spec, output):
        from .isaac_path_rollout import IsaacPathRollout
        from .torch_path_policy import TorchPathPolicy
        from .torch_path_reference import TorchPathReference
        if env.num_envs != 1:
            raise ValueError('Ditch handback diagnostic requires one lane')
        self.env,self.spec,self.output=env,spec,Path(output)
        settings=json.loads(Path(spec['settings']).read_text())
        settings.update(num_envs=1,alphas=[spec['alpha']],zero_middle=[False],zero_lower=[True],
                        terrains={'ditch':'shared'},episode_s=env.max_episode_length*env.dt,
                        initial_state=None,user_speed_m_s=spec['speed_m_s'],auto_reset_horizon=False,
                        record_control_commands=True)
        bundle=Path(spec['bundle']);identity=json.loads((bundle/'identity.json').read_text())
        config=json.loads((bundle/'config.json').read_text())
        actor=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],config,1,env.device)
        self.ground=IsaacPathRollout(bundle,settings,actor,plant=env.plant)
        ss=np.arange(0,30.01,.02)
        route=dict(s=ss,xy=np.column_stack((ss-5.,np.zeros_like(ss))),heading=np.zeros_like(ss),curvature=np.zeros_like(ss))
        self.ground.begin_connection(TorchPathReference(route,config,env.device),self.ground.speed.clone(),self.ground.alpha.clone())
        self.active=False;self.commands=[]
        (self.output/'ground_identity.json').write_text(json.dumps(dict(upper_sha256=self.ground.upper_sha256,
            lower_sha256=self.ground.lower_sha256,zero_lower=True,controller='H75 + ECBC/ESO; lower residual zero',
            reference='Original world-local y=0, no reanchoring at landing',control_dt_s=.005,policy_dt_s=.02,
            feedback='Existing ground clean-state interface; frozen randomized physical properties retained; jump noise/delay only before handback',
            pendulum='Existing joint-brake model at current posture'),indent=2))

    def step(self, latent):
        e=self.env;g=self.ground
        if not self.active and int(e.verified_jump.scorer.stage[0])>=3:
            before=e.physical()[0].clone()
            e.plant.set_pendulum_locked(True)
            g.resume_ground_lanes(e.all_ids,self.spec['speed_m_s'],self.spec['alpha'])
            g.physical_tick.fill_(round(float(e.episode_length_buf[0])*e.dt/.005))
            self.active=True
            (self.output/'handoff.json').write_text(json.dumps(dict(time_s=float(e.episode_length_buf[0])*e.dt,
                root=before.cpu().tolist(),stage=3,physical_reset=False,
                unchanged_root=bool(torch.equal(before,e.physical()[0]))),indent=2))
        if not self.active:return e.step(latent)
        reward=torch.zeros(1,device=e.device);code=torch.zeros(1,device=e.device,dtype=torch.long);tick=0
        def physics(command):
            nonlocal reward,code,tick
            result=e.plant.step(command,refresh_diagnostics=True)
            inc,done,success,stage=e.verified_jump.sample(e,torch.zeros(1,4,device=e.device),tick==e.physics_substeps-1)
            code=torch.where(done&(code==0),torch.where(success,13,e.verified_jump.end_code),code)
            reward+=inc;tick+=1
            return result
        g.physics_step_override=physics
        result=g.step();self.commands.append(result['executed_commands'].cpu().numpy())
        scorer=e.verified_jump.scorer
        supervisor_failure=result['failure']&~scorer.done
        # A ground supervisor stop is a task failure, not a physical-success exit.
        # Close the same finite-horizon balance once, cancelling residual potential.
        reward-=torch.where(supervisor_failure,100.+scorer.phi,0.)
        scorer.phi.masked_fill_(supervisor_failure,0.)
        scorer.done|=supervisor_failure
        if bool(result['failure'].any()):
            (self.output/'ground_stop.json').write_text(json.dumps(dict(
                time_s=float(e.episode_length_buf[0]+1)*e.dt,
                domain_exit=g.middle.domain_exit.cpu().tolist(),numerical_fault=g.middle.fault.cpu().tolist(),
                ground_failure=result['failure'].cpu().tolist(),root=e.physical()[0].cpu().tolist(),
                tracking_error=result['tracking_error'].cpu().tolist()),indent=2))
        e.platform_held.copy_(scorer.success);e.episode_length_buf+=1;e.episode_return+=reward
        done=scorer.done.clone()|result['failure']
        reason=torch.where(g.middle.domain_exit,15,torch.where(g.middle.fault,16,17))
        code=torch.where(result['failure']&(code==0),reason,code)
        info=dict(unscaled_reward=reward,end_code=code,reward_parts={'verified_jump':reward,'episode_return_override':torch.zeros_like(reward)},
                  jump_stage=scorer.stage.clone(),jump_success=scorer.success.clone())
        return e.obs_buf,e.privileged_obs_buf,reward*e.task['ppo']['reward_scaling'],done,info
