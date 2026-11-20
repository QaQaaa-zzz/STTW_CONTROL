"""Fixed-budget raised-platform departure queries on the live GPU state."""
import numpy as np
import torch
from .playground_history_model import GroundPredictionState
from .playground_terrain_features import TerrainFeatures
from .playground_prediction import BodyEnvelope, euler, rotate


def kinematic(model, initial, speed, request, route, steps):
    follower = model.nominal.follower
    wheelbase = follower['wheelbase_m_expected']
    x = initial.clone()
    yaw = euler(x[:,3:7])[:,2]
    zero = torch.zeros_like(yaw)
    x[:,3:7] = torch.stack(((yaw/2).cos(), zero, zero, (yaw/2).sin()), 1)
    # Locked joints are common geometry. This baseline observes no live steering,
    # roll, yaw rate, or controller history. Use geometry and the actual chord.
    x[:,7] = 0
    u = torch.zeros((len(x),9), device=x.device)
    u[:,0] = speed
    poses, speeds, domains = [], [], []
    for _ in range(steps+1):
        f = model.nominal.features(x, u, *route)
        poses.append(x.clone())
        speeds.append(u.clone())
        domains.append(f[:,9:12])
        chord2 = f[:,11:13].square().sum(1).clamp_min(1e-8)
        steer = torch.atan(2*wheelbase*f[:,12]/chord2).clamp(-follower['steer_reference_max_rad'], follower['steer_reference_max_rad'])
        dyaw = .02*u[:,0]*steer.tan()/wheelbase
        x[:,0] += .02*u[:,0]*(yaw+dyaw/2).cos()
        x[:,1] += .02*u[:,0]*(yaw+dyaw/2).sin()
        yaw = yaw+dyaw
        x[:,3:7] = torch.stack(((yaw/2).cos(), zero, zero, (yaw/2).sin()),1)
        x[:,7] = steer
        dv = .02*follower['nominal_speed_slew_m_s2']
        u[:,0] += (request-u[:,0]).clamp(-dv,dv)
    return tuple(torch.stack(values, 1) for values in (poses, speeds, domains))


def velocity_state(q, v):
    inverse = torch.cat((q[:, 3:4], -q[:, 4:7]), 1)
    return torch.cat((rotate(inverse, v[:, :3]), v[:, 3:6], v[:, [7, 8, 6]]), 1)


class PlatformDepartureQuery:
    def __init__(self, physics, nominal, goal, margin, halfwidth, relative=None):
        self.model=nominal
        self.relative = relative
        if relative:
            terrain = dict(np.load(relative['raster']))
            terrain['height'] -= relative['height']
            nominal.terrain = TerrainFeatures(**terrain, device='cuda')
            with np.load(relative['initializer']) as init:
                self.mean, self.scale, self.weights = [torch.as_tensor(init[k], device='cuda', dtype=torch.float32) for k in ('mu','scale','weight')]
        self.envelope=BodyEnvelope(physics,dimensions=3)
        walls=[i for i in range(physics.ngeom) if physics.geom(i).name.startswith('planning_obstacle_')]
        self.center=torch.as_tensor(physics.geom_pos[walls],device='cuda',dtype=torch.float32)
        self.half=torch.as_tensor(physics.geom_size[walls],device='cuda',dtype=torch.float32)
        self.goal=goal
        self.margin=margin
        self.halfwidth=halfwidth

    @torch.inference_mode()
    def choose(self,q,body_speed,request,route,branch_ids,section,side,old,seconds,velocity=None,previous=None):
        # The caller has the first physical state after relock; measure three feasible continuations.
        pose=q[:,[0,1,2,3,4,5,6,8,9,7,10,11]].repeat_interleave(3,0)
        refs=branch_ids.flatten()
        published=(route.s,route.xy[refs],route.heading[refs],route.curvature[refs])
        req=request.repeat_interleave(3)
        if self.relative:
            u, prev = velocity.repeat_interleave(3,0), previous.repeat_interleave(3,0)
            inputs = torch.cat((pose[:,:8],pose[:,10:12],u,prev),1)
            hidden = ((inputs-self.mean)/self.scale) @ self.weights[1:] + self.weights[0]
            pose[:,2] -= self.relative['height']
            f = self.model.rollout_state(GroundPredictionState(pose,u,prev,hidden), *published, int(seconds/.02),req)
            poses,v,domain = f['pose'],f['velocity'],f['domain']
            poses[:,:,2] += self.relative['height']
        else:
            poses,v,domain=kinematic(self.model,pose,body_speed.repeat_interleave(3),req,published,int(seconds/.02))
        lo,hi=self.envelope(poses)
        dist=torch.maximum(lo[:,:,:,None]-self.center-self.half,self.center-self.half-hi[:,:,:,None]).amax(-1).amin((2,3))
        yaw=euler(poses[:,:,3:7].reshape(-1,4))[:,2].reshape(poses.shape[:2])
        speed=v[:,:,0]
        vx = speed*yaw.cos()
        if self.relative:
            world_v=rotate(poses[:,:,3:7].reshape(-1,4),v[:,:,:3].reshape(-1,3)).reshape(*yaw.shape,3)
            vx=world_v[:,:,0]
            speed=vx*yaw.cos()+world_v[:,:,1]*yaw.sin()
        goal=self.goal
        hit=(poses[:,:,0]>=goal['x_min_m'])&(poses[:,:,0]<=goal['x_max_m'])&(poses[:,:,1].abs()<=goal['abs_y_max_m'])&(vx>=goal['min_vx_m_s'])&(domain[:,:,0].abs()<=goal['path_tolerance_m'])&(domain[:,:,1].abs()<=goal['heading_tolerance_rad'])&((speed-req[:,None]).abs()<=goal['speed_tolerance_m_s'])
        steps=hit.shape[1]
        if steps<26:return old,torch.zeros_like(old,dtype=torch.bool),torch.zeros_like(old,dtype=torch.bool)
        held=hit.unfold(1,26,1).all(-1)
        end=torch.where(held.any(1),held.long().argmax(1)+25,steps-1)
        window=torch.arange(steps,device=q.device)<=end[:,None]
        clearance=dist.masked_fill(~window,float('inf')).amin(1)
        x,y=poses[:,:,0],poses[:,:,1];target=section.repeat_interleave(3)[:,None]
        cross=(x[:,:-1]<target)&(x[:,1:]>=target)&window[:,1:]
        idx=cross.long().argmax(1);i=torch.arange(len(idx),device=q.device)
        at=y[i,idx]+(target[:,0]-x[i,idx])*(y[i,idx+1]-y[i,idx])/(x[i,idx+1]-x[i,idx]).clamp_min(1e-7)
        gate=cross.any(1)&(at*side.repeat_interleave(3)>=.06)&(at.abs()<=3.9)
        bad=(domain[:,:,0].abs()>self.halfwidth)|(domain[:,:,1].abs()>=torch.pi/2)|(domain[:,:,2]<self.model.nominal.follower['preview_body_x_min_m'])
        if self.relative:
            attitude=euler(poses[:,:,3:7].reshape(-1,4)).reshape(*yaw.shape,3)
            bad |= (attitude[:,:,0].abs()>.7)|(attitude[:,:,1].abs()>1.309)
        safe=clearance>self.margin
        pass_goal=held.any(1)&~(bad&window).any(1)
        feasible=safe&pass_goal&gate
        score=(16*feasible.float()+4*safe.float()+2*gate.float()+pass_goal.float()+.01*clearance.clamp(-1,1)).reshape(-1,3)
        # Keep C's command when it is predicted feasible; otherwise prefer safer alternatives.
        lane=torch.arange(len(old),device=q.device)
        choice=torch.where(feasible.reshape(-1,3)[lane,old],old,score.argmax(1))
        return choice,feasible.reshape(-1,3)[lane,choice],feasible.reshape(-1,3)[lane,old]
