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
        self.leg_hold=spec.get('leg_hold','joint_brake')
        self.handoff_stage=spec.get('handoff_stage',3)
        self.handoff_event=spec.get('handoff_event','stage')
        if self.handoff_event not in ('stage','first_far_contact'):
            raise ValueError('Unknown landing handoff event')
        self.task_continuation=spec.get('continue_to_horizon',False)
        if self.handoff_stage not in (3,4):
            raise ValueError('Handoff stage must be capture3 or completed hold4')
        if self.leg_hold not in ('joint_brake','finite_motor'):
            raise ValueError('Unknown handoff leg hold mode')
        settings=json.loads(Path(spec['settings']).read_text())
        settings.update(num_envs=1,alphas=[spec['alpha']],zero_middle=[False],zero_lower=[True],
                        terrains={'ditch':'shared'},episode_s=env.max_episode_length*env.dt,
                        initial_state=None,user_speed_m_s=spec['speed_m_s'],auto_reset_horizon=False,
                        record_control_commands=True)
        bundle=Path(spec['bundle']);identity=json.loads((bundle/'identity.json').read_text())
        config=json.loads((bundle/'config.json').read_text())
        actor=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],config,1,env.device)
        self.ground=IsaacPathRollout(bundle,settings,actor,plant=env.plant)
        reference_path=spec.get("reference_path")
        if reference_path:
            with np.load(reference_path) as saved:
                route={key:saved[key] for key in ("s","xy","heading","curvature")}
        else:
            ss=np.arange(0,30.01,.02)
            route=dict(s=ss,xy=np.column_stack((ss-5.,np.zeros_like(ss))),heading=np.zeros_like(ss),curvature=np.zeros_like(ss))
        self.ground.begin_connection(TorchPathReference(route,config,env.device),self.ground.speed.clone(),self.ground.alpha.clone())
        self.active=False;self.commands=[];self.command_times=[];self.command_phases=[]
        self.history=[];self.record_history=spec.get("record_controller_history",False)
        self.approach=spec.get('approach')
        self.approaching=self.approach is not None
        self.approach_initialized=False
        (self.output/'ground_identity.json').write_text(json.dumps(dict(upper_sha256=self.ground.upper_sha256,
            lower_sha256=self.ground.lower_sha256,zero_lower=True,controller='H75 + ECBC/ESO; lower residual zero',
            reference=reference_path or 'Original world-local y=0, no reanchoring at landing',control_dt_s=.005,policy_dt_s=.02,
            feedback='Existing ground clean-state interface; frozen randomized physical properties retained; jump noise/delay only before handback',
            pendulum=self.leg_hold),indent=2))

    def step(self, latent):
        e=self.env;g=self.ground
        if self.approaching:
            if not self.approach_initialized:
                g.resume_ground_lanes(e.all_ids,self.approach['speed_m_s'],self.approach['alpha'])
                self.approach_initialized=True
            if float(e.physical()[0][0,0])>=self.approach['jump_start_x_m']:
                self.approaching=False
                np.savez_compressed(self.output/'approach_entry.npz',root=e.physical()[0].cpu().numpy(),
                    observation=e.obs_buf.cpu().numpy(),joint_state=e.plant.dofs.cpu().numpy(),
                    time_s=float(e.episode_length_buf[0])*e.dt)
        ready=(bool(e.verified_jump.observed_far_contact[0]) if self.handoff_event=='first_far_contact'
               else int(e.verified_jump.scorer.stage[0])>=self.handoff_stage)
        if not self.approaching and not self.active and ready:
            before=e.physical()[0].clone()
            if self.leg_hold=='joint_brake':
                e.plant.set_pendulum_locked(True)
            else:
                # Preserve native limits and use existing finite-torque position servos.
                indices=e.plant.pendulum_servos.indices
                e.plant.pendulum_target=e.plant.dofs[:,indices,0].clone()
                e.plant.pendulum_target=torch.maximum(e.lo[2:],torch.minimum(e.hi[2:],e.plant.pendulum_target))
            g.resume_ground_lanes(e.all_ids,self.spec['speed_m_s'],self.spec['alpha'])
            g.physical_tick.fill_(round(float(e.episode_length_buf[0])*e.dt/.005))
            self.active=True
            (self.output/'handoff.json').write_text(json.dumps(dict(time_s=float(e.episode_length_buf[0])*e.dt,
                root=before.cpu().tolist(),stage=int(e.verified_jump.scorer.stage[0]),event=self.handoff_event,physical_reset=False,leg_hold=self.leg_hold,
                leg_target=e.plant.pendulum_target.cpu().tolist(),
                unchanged_root=bool(torch.equal(before,e.physical()[0]))),indent=2))
        if not self.active and not self.approaching:
            transition=e.step(latent)
            if self.handoff_stage==4 and bool(transition[4]['jump_success'].all()) and int(e.episode_length_buf[0])<e.max_episode_length:
                obs,priv,reward,done,info=transition
                return obs,priv,reward,torch.zeros_like(done),info
            return transition
        reward=torch.zeros(1,device=e.device);code=torch.zeros(1,device=e.device,dtype=torch.long);tick=0
        def physics(command):
            nonlocal reward,code,tick
            old_velocity=e.plant.roots[:,7:10].clone()
            result=e.plant.step(command,refresh_diagnostics=True)
            inc,done,success,stage=e.verified_jump.sample(e,torch.zeros(1,4,device=e.device),tick==e.physics_substeps-1)
            terminal=e.verified_jump.physical_failure if self.task_continuation or self.handoff_stage==4 else done
            reason=e.verified_jump.end_code if self.task_continuation or self.handoff_stage==4 else torch.where(success,13,e.verified_jump.end_code)
            code=torch.where(terminal&(code==0),reason,code)
            reward+=inc;tick+=1
            if self.approaching and tick==e.physics_substeps:
                from .phase_u_events import advance_events
                from .jump_heading import heading_features
                root=e.physical()[0]
                acceleration=(root[:,7:10]-old_velocity)/e.plant.physics_dt
                reference=e.cfg.get('policy_obstacle_reference_x_m',e.ditch['front_x_m'])
                distance=reference-(e.geometry.frontmost_x(e.plant.rigid)-e.plant.origin[:,0])
                # Existing chain convention: zero jump action history during ground approach.
                frame=e.history.make_frame(root,e.plant.dofs[:,:,0],e.plant.dofs[:,:,1],acceleration,distance,e.last_action)
                e.events=advance_events(e.events,dict(x=root[:,0],z=root[:,2],vertical_velocity=root[:,9],physical_failure=e.verified_jump.physical_failure,forward_velocity=root[:,7]),e.task)
                e.obs_buf=torch.cat((e.history.advance_frame(e.randomization.sensor_frame(frame),e.events['jump_signal'].float()),heading_features(root)),-1)
                e.privileged_obs_buf=e.physical()[2]
            return result
        g.physics_step_override=physics
        if self.record_history:
            # Observation-only trace; no solver snapshot, restoration, or control mutation.
            groups={
                'controller':(g.controller,('eso','disturbance','gains')),
                'lower':(g.lower,('frames','mask','previous_applied','yaw_rate')),
                'policy':(g.policy,('filtered','offsets','execution_mode','hold_channels')),
                'middle':(g.middle,('frames','mask','pose','yaw_rate','nominal','rates','fault','domain_exit')),
                'rollout':(g,('previous_reference','previous_command','measurement','alpha','speed','task_tick','physical_tick')),
            }
            row={group+'_'+key:getattr(obj,key).detach().cpu().numpy().copy()
                 for group,(obj,keys) in groups.items() for key in keys}
            row.update(root_xyzw=e.physical()[0].detach().cpu().numpy().copy(),
                       joint_state=e.plant.dofs.detach().cpu().numpy().copy())
            self.history.append(row)
        result=g.step();self.commands.append(result['executed_commands'].cpu().numpy())
        self.command_times.append(float(e.episode_length_buf[0])*e.dt)
        self.command_phases.append('approach' if self.approaching else 'continuation')
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
        if self.task_continuation or self.handoff_stage==4:
            done=(code!=0)|result['failure']|(e.episode_length_buf>=e.max_episode_length)
            code=torch.where((e.episode_length_buf>=e.max_episode_length)&(code==0)&~result['failure'],9,code)
        else:
            done=scorer.done.clone()|result['failure']
        reason=torch.where(g.middle.domain_exit,15,torch.where(g.middle.fault,16,17))
        code=torch.where(result['failure']&(code==0),reason,code)
        info=dict(unscaled_reward=reward,end_code=code,reward_parts={'verified_jump':reward,'episode_return_override':torch.zeros_like(reward)},
                  jump_stage=scorer.stage.clone(),jump_success=scorer.success.clone())
        return e.obs_buf,e.privileged_obs_buf,reward*e.task['ppo']['reward_scaling'],done,info
