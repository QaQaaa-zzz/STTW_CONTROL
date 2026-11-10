"""Physics-step inputs for JumpReward on the committed rough-bank mesh.

The map has a single terrain contact filter, rough center |y| <= 1, and two
separate banks. Shape boundaries and filtered solver contact points decide
which bank supports a wheel. No ground plane is permitted under the trench.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


class BankEnvelope:
    """Conservative mesh upper bound under the swept footprint of one wheel."""

    def __init__(self, vertices, spec, radius, half_width, device, slices=21):
        self.banks = []
        self.minimum_height = min(0., spec.get("far_bank_height_m",0.))
        width = math.hypot(radius, half_width)
        self.bound = width
        # IsaacLabVectorPlant creates these cylinders along the wheel link's
        # local Z axis. Keep the exact vertical support function of their union.
        if slices < 1 or slices % 2 != 1:
            raise ValueError('Expected positive odd cylinder slice count')
        step = 2*half_width/slices
        self.slice_half = step/2
        centers = torch.arange(slices,device=device)*step-((slices-1)/2)*step
        self.slice_offset = centers.abs()
        self.slice_radius = radius*torch.sqrt((1-(centers/half_width).square()).clamp_min(0))
        for left, right in ((2., spec['front_x_m']),
                            (spec['front_x_m'] + spec['width_m'], 8.)):
            # Build from the actual triangle vertices issued to the simulator.
            # Ditch skirts are flat. Their duplicate seam vertices share the bank elevation.
            keep = ((vertices[:, 0] >= np.float32(left)) &
                    (vertices[:, 0] <= np.float32(right)) &
                    (np.abs(vertices[:, 1]) <= 1.))
            xyz = vertices[keep]
            xs, ix = np.unique(xyz[:, 0], return_inverse=True)
            ys, iy = np.unique(xyz[:, 1], return_inverse=True)
            if len(xs) < 2 or len(ys) != 81 or len(np.unique(ix*len(ys)+iy)) != len(xs)*len(ys):
                raise ValueError('Expected the pinned rectangular rough-bank meshes')
            z = np.full((len(xs), len(ys)), -np.inf, dtype=np.float32)
            np.maximum.at(z, (ix, iy), xyz[:, 2])
            dx, dy = (xs[-1]-xs[0])/(len(xs)-1), (ys[-1]-ys[0])/(len(ys)-1)
            if not np.allclose(np.diff(xs), dx, atol=1e-5) or not np.allclose(np.diff(ys), dy, atol=1e-5):
                raise ValueError('Expected uniform rough-bank mesh grid')
            # Every triangle value lies below its greatest vertex. The extra cell
            # encloses triangle vertices at both edges of the bounding footprint.
            kx, ky = math.ceil(width/dx)+1, math.ceil(width/dy)+1
            raw = torch.as_tensor(z, device=device)[None, None]
            high = F.max_pool2d(raw,(2*kx+1,2*ky+1),1,(kx,ky))[0,0]
            self.banks.append((float(xs[0]),float(xs[-1]),float(dx),float(dy),high))

    def clearance(self, wheels):
        x, y, z = wheels.unbind(-1)
        height = torch.full_like(z,self.minimum_height)
        for left, right, dx, dy, grid in self.banks:
            ix = ((x-left)/dx).round().long().clamp(0,grid.shape[0]-1)
            iy = ((y+1.)/dy).round().long().clamp(0,grid.shape[1]-1)
            reach = (x>=left-self.bound)&(x<=right+self.bound)&(y.abs()<=1.+self.bound)
            height = torch.maximum(height,torch.where(reach,grid[ix,iy],self.minimum_height))
        # Bounding sphere encloses every slice of the pinned profiled cylinders
        # at every camber. The open gap uses the lower bank as virtual envelope.
        return z-self.bound-height

    def clearance_upper(self, wheels, wheel_quat):
        # q maps wheel-local axes to world, in plant.rigid's documented xyzw
        # order. A vertical line is resolved into cylinder axial and radial
        # components. Subtracting the lowest bank elevation upper-bounds
        # clearance for both raised and lowered landing banks.
        unit = torch.nn.functional.normalize(wheel_quat,dim=-1)
        axis_z = (1-2*unit[:,:,:2].square().sum(-1)).abs().clamp(max=1)[...,None]
        tangent = torch.sqrt((1-axis_z.square()).clamp_min(0))
        extent = ((self.slice_offset+self.slice_half)*axis_z+self.slice_radius*tangent).amax(-1)
        return wheels[:,:,2]-extent-self.minimum_height


def classify_patches(raw, origin, wheel_z, near_lip, far_lip, target_end, lane_halfwidth, capacity):
    """Use *all* loaded terrain patches, not their average across a border."""
    magnitude, point, normal, _, count, start = raw
    n = len(origin)
    count, start = count.reshape(n,-1), start.reshape(n,-1)
    # Depending on PhysX replication, the terrain filter may expose more than
    # one shape column. Each wheel must touch only the terrain of its own lane.
    single_filter = (count>0).sum(1)==1
    count, selected = count.max(dim=1,keepdim=True)
    start = start.gather(1,selected)
    slots = torch.arange(capacity,device=origin.device)[None,:]
    index = start.long()+slots
    if len(point) == 0:
        false = torch.zeros(n,dtype=torch.bool,device=origin.device)
        return false, false
    valid = (slots<count)&(index>=0)&(index<len(point))
    safe_index = index.clamp(0,len(point)-1)
    loaded_patch = valid & (magnitude.reshape(-1)[safe_index].abs()>1.e-6)
    local = point.reshape(-1,3)[safe_index]-origin[:,None,:]
    xy = local[:,:,:2]
    lane = (xy[:,:,1].abs()<=lane_halfwidth) & (wheel_z[:,None]>=local[:,:,2]+.02)
    # Wheel center must be above the support and the terrain-filtered normal
    # forces must carry measurable vertical load. Reject saturated buffers.
    vertical = (normal.reshape(-1,3)[safe_index][:,:,2]*magnitude.reshape(-1)[safe_index]).abs()
    good_index = single_filter&(count.squeeze(1)<capacity)&((slots>=count)|valid).all(1)
    observed = good_index & (torch.where(loaded_patch,vertical,0.).sum(1)>1.)
    def all_on(patch):
        return observed & (~loaded_patch | (lane & patch)).all(1)
    return all_on((xy[:,:,0] >= -5.) & (xy[:,:,0] <= near_lip)), all_on((xy[:,:,0] >= far_lip) & (xy[:,:,0] <= target_end))


class DitchJumpAdapter:
    def __init__(self, env, vertices):
        from .offroad_jump_reward import JumpReward
        plant, spec = env.plant, env.ditch
        if (not spec or env.cfg['plant_options'].get('terrain_base_plane') or
                len(plant.wheel_sensors) != 2 or env.cfg['plant_options']['wheel_collision'] != 'cylinder'):
            raise ValueError('Verified ditch scoring requires the pinned open rough-bank map and filtered wheel contacts')
        self.near_lip = spec['front_x_m']
        self.far_lip = self.near_lip+spec['width_m']
        self.end = min(8., spec['ground_end_x_m'])
        self.halfwidth = 1.
        # Saturation is a buffer-validity check, not a physical contact limit.
        self.contact_capacity = env.cfg['plant_options'].get('wheel_contact_buffer_capacity') or env.cfg['plant_options']['wheel_patch_capacity']
        self.bounds = BankEnvelope(vertices,spec,plant.wheel_radius_m,
                                   env.cfg['plant_options']['wheel_width_m']/2,env.device,
                                   env.cfg['plant_options']['wheel_profile_slices'])
        mode = env.cfg['verified_ditch_support_mode']
        if mode not in ('rough','strict'):
            raise ValueError('Choose verified_ditch_support_mode rough or strict')
        self.scorer = JumpReward(env.num_envs,env.device,dt=plant.physics_dt,
                                 width=spec['width_m'],hold_s=1.,
                                 approach_distance=self.near_lip-1.5,rough_support=(mode=='rough'))
        self.record_diagnostics = env.cfg.get('record_verified_diagnostics',False)
        self.diagnostics = []

    def reset(self, ids):
        self.scorer.reset(ids)
        if self.record_diagnostics:
            self.diagnostics = []

    def sample(self, env, action, final_tick):
        p = env.plant
        root, body_contact, critic = env.physical()
        body = p.rigid[:,env.wheel_ids,:7]
        wheels = body[:,:,:3]-p.origin[:,None,:]
        wheel_quat = body[:,:,3:7]
        raw_contact = [sensor.contact_physx_view.get_contact_data(dt=p.physics_dt) for sensor in p.wheel_sensors]
        material = [classify_patches(raw,
                                     p.origin,wheels[:,i,2],self.near_lip,self.far_lip,
                                     self.end,self.halfwidth,self.contact_capacity)
                    for i,raw in enumerate(raw_contact)]
        force = p.contact[:,env.wheel_ids]
        finite = (torch.isfinite(root).all(1)&torch.isfinite(p.dofs).all(dim=(1,2))&
                  torch.isfinite(body).all(dim=(1,2))&(wheel_quat.norm(dim=-1)>.5).all(1)&
                  torch.isfinite(force).all(dim=(1,2))&
                  torch.isfinite(action).all(1))
        loaded = force.norm(dim=-1)>1.
        near, target = (torch.stack(q,dim=1)&(force[:,:,2]>1.) for q in zip(*material))
        roll, pitch = critic[:,99], critic[:,100]
        # Privileged qvel[3:6] contains angular velocity in the virtual,
        # upright body frame. Euler rates retain the correct sign when yawed.
        wx, wy, wz = critic[:,91:94].unbind(-1)
        roll_rate = wx+(wy*roll.sin()+wz*roll.cos())*pitch.clamp(-1.4,1.4).tan()
        pitch_rate = wy*roll.cos()-wz*roll.sin()
        limit = env.task['physical_limits']
        failed = ((~finite) | body_contact | (root[:,2]<-.15) |
                  (root[:,1].abs()>env.ditch['lateral_limit_m']) |
                  (root[:,0]<1.5-limit['max_backward_distance']) |
                  (roll.abs()>limit['max_abs_roll']) | (pitch.abs()>limit['max_abs_pitch']))
        timeout = (env.episode_length_buf+1>=env.max_episode_length)&final_tick
        code = torch.full_like(env.episode_length_buf,14)  # core event rejection
        code = torch.where(timeout,9,code)
        code = torch.where(failed,12,code)
        code = torch.where(root[:,0]<1.5-limit['max_backward_distance'],7,code)
        code = torch.where(pitch.abs()>limit['max_abs_pitch'],4,code)
        code = torch.where(roll.abs()>limit['max_abs_roll'],3,code)
        self.end_code = torch.where(finite,code,2)
        wheels = torch.nan_to_num(wheels,nan=0.,posinf=0.,neginf=0.)
        def safe(value):
            return torch.nan_to_num(value,nan=0.,posinf=0.,neginf=0.)
        result = self.scorer.step(s=wheels[:,:,0]-self.near_lip,
                                clearance=self.bounds.clearance(wheels),
                                clearance_upper=safe(self.bounds.clearance_upper(wheels,wheel_quat)),
                                loaded=loaded,
                                near=near,target=target,lane=wheels[:,:,1].abs()<=self.halfwidth,
                                v=safe(root[:,7]),vn=safe(root[:,9]),roll=safe(roll),
                                pitch=safe(pitch),roll_rate=safe(roll_rate),pitch_rate=safe(pitch_rate),
                                failed=failed,timeout=timeout)
        if self.record_diagnostics:
            self.diagnostics.append(dict(
                wheel_x=wheels[:,:,0].clone(),near=near.clone(),target=target.clone(),loaded=loaded.clone(),
                clearance=self.bounds.clearance(wheels),clearance_upper=self.bounds.clearance_upper(wheels,wheel_quat),
                roll=roll.clone(),pitch=pitch.clone(),armed=self.scorer.armed.clone(),
                attempt=self.scorer.attempt.clone(),stage=result[3].clone(),failed=failed.clone(),
                patch_count=torch.stack([raw[4].reshape(env.num_envs,-1).sum(1) for raw in raw_contact],-1),
                patch_out_of_bounds=torch.stack([((raw[4]>0)&((raw[5]<0)|(raw[5]+raw[4]>len(raw[1])))).reshape(env.num_envs,-1).any(1) for raw in raw_contact],-1)))
        return result
