"""Shared observable jump-connection features for collection and execution."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sttw_control.empirical_capability import ConnectionCapability, select_connection_estimate, connection_selection_status

fields=[('speed','m/s'),('roll','rad'),('roll_rate','rad/s'),('cross_track','m'),('heading_error','rad'),('steer','rad'),('steer_rate','rad/s'),('previous_speed_reference','m/s'),('lateral_velocity_heading_frame','m/s'),('world_yaw_rate','rad/s'),('previous_alpha','1'),('previous_alpha_duration','s'),('remaining_path_to_release','m'),('path_curvature','1/m'),('corridor_half_width','m'),('pitch','rad'),('body_angular_y','rad/s'),('hip','rad'),('knee','rad'),('hip_velocity','rad/s'),('knee_velocity','rad/s'),('platform_height','m'),('remaining_path_to_platform','m')]

def decision_features(decision,run,protocol,spec):
 root=np.asarray(decision['root'])[0];measurement=np.asarray(decision['measurement'])[0]
 qx,qy,qz,qw=root[3:7];yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
 route=np.load(run/'ground_route.npz');xy=route['xy'];edges=np.diff(xy,axis=0);u=np.clip(((root[:2]-xy[:-1])*edges).sum(-1)/(edges*edges).sum(-1),0,1);points=xy[:-1]+u[:,None]*edges;i=int(np.argmin(((points-root[:2])**2).sum(-1)))
 progress=float(route['s'][i]+u[i]*(route['s'][i+1]-route['s'][i]));heading=float(np.interp(progress,route['s'],route['heading']));delta=root[:2]-points[i]
 release_index=int(np.argmin(np.linalg.norm(xy-[spec['handoff_x'],0.],axis=1)));edge_index=int(np.argmin(np.linalg.norm(xy-[spec['platform_front_x'],0.],axis=1)))
 model=json.loads(Path(spec['model_config']).read_text());mount=np.asarray(model['mount_rotation_wxyz'])[[1,2,3,0]];rotation=Rotation.from_quat(root[3:7])*Rotation.from_quat(mount)
 # Only recorded live articulation ordering can bind joint velocity slots.
 live=protocol.get('joint_names')
 if live is None:
  trajectory=np.load(run/'trajectory.npz');trace=trajectory['trace'][:,0];sample=trace[np.argmin(abs(trace[:,0]-decision['time']))]
  hip=float(sample[9]);knee=float(sample[10]);hip_v=knee_v=None
 else:
  h=live.index('hip_joint');k=live.index('knee_joint');hip=decision['position'][0][h];knee=decision['position'][0][k];hip_v=decision['velocity'][0][h];knee_v=decision['velocity'][0][k]
 values=[float(np.cos(yaw)*root[7]+np.sin(yaw)*root[8]),float(measurement[0]),float(measurement[1]),float(-np.sin(heading)*delta[0]+np.cos(heading)*delta[1]),float(np.arctan2(np.sin(heading-yaw),np.cos(heading-yaw))),float(measurement[2]),float(measurement[3]),decision['previous_reference'][0][0],float(-np.sin(yaw)*root[7]+np.cos(yaw)*root[8]),float(root[12]),decision['previous_alpha'][0],decision.get('previous_alpha_duration_s',decision['time']),float(route['s'][release_index]-progress),float(np.interp(progress,route['s'],route['curvature'])),spec['completion']['corridor_half_width_m'],float(rotation.as_euler('xyz')[1]),float(rotation.inv().apply(root[10:13])[1]),hip,knee,hip_v,knee_v,.16,float(route['s'][edge_index]-progress)]
 return values,dict(zip([n for n,_ in fields],values))


class JumpConnectionSelector:
    """Frozen Q_hold evidence; exclude the whole queried origin before lookup."""
    def __init__(self, records, config, *, origin_id, policy_sha256, model_contract_sha256, physics_dt, zero_lower):
        self.config = config
        self.candidates = {f"v{c['speed_m_s']}_a{c['alpha']}": c for c in config['candidates']}
        accepted = [r for r in records if r['origin_id'] != origin_id
            and r['target_semantics'] == 'Q_hold' and r['supervised_input_complete']
            and r['policy_sha256'] == policy_sha256 and r['model_contract_sha256'] == model_contract_sha256
            and r['physics_dt'] == physics_dt and r['zero_lower'] == zero_lower
            and (r.get('decision_prefix') or {}).get('corridor_intact_at_decision') is True]
        kernel = [dict(state=r['state'], task_id=r['decision_stage'], skill_id='matched_runtime',
            candidate_id=f"v{r['candidate']['speed_m_s']}_a{r['candidate']['alpha']}",
            complete=r['Y'], success_cost=r['T_after_decision_s']) for r in accepted]
        self.capability = ConnectionCapability(kernel, config['feature_scales'],
            config['bandwidth'], config['minimum_support'])
        self.evidence_rows = len(kernel)
        self.excluded_origin_id = origin_id

    def choose(self, state, stage, current):
        query = self.capability.query(state, stage, 'matched_runtime', list(self.candidates))
        previous = next((key for key, value in self.candidates.items()
            if abs(value['speed_m_s']-current['speed_m_s']) < 1e-6 and abs(value['alpha']-current['alpha']) < 1e-6), None)
        choice = select_connection_estimate(query['estimates'], self.config['minimum_completion'],
            self.config['probability_tolerance'], previous_candidate=previous,
            time_tolerance=self.config['time_tolerance_s'])
        selected = self.candidates[choice['candidate_id']] if choice is not None else current
        return dict(status=connection_selection_status(query['estimates'], choice),
            candidate=dict(selected), estimates=query['estimates'], evidence_rows=self.evidence_rows,
            excluded_origin_id=self.excluded_origin_id,
            fallback='retain_current_reference' if choice is None else None,
            scope='Empirical Q_hold recommendation, not a calibrated or guaranteed success probability')


class ForwardRouteSelector:
    """Known-map route choice using the shared task manager and Q_route data."""
    def __init__(self, records, config, *, origin_id, policy_sha256, model_contract_sha256, physics_dt, zero_lower):
        from .continuous_task import ContinuousTask
        accepted = [r for r in records if r['origin_id'] != origin_id
            and r['target_semantics'] == 'Q_route' and r['supervised_input_complete']
            and r['policy_sha256'] == policy_sha256 and r['model_contract_sha256'] == model_contract_sha256
            and r['physics_dt'] == physics_dt and r['zero_lower'] == zero_lower
            and r['map_id'] == config['map_id']]
        capability = ConnectionCapability(accepted, config['feature_scales'], config['bandwidth'], config['minimum_support'])
        self.task = ContinuousTask(config['program'], config['routes'], config['candidates'], capability)
        self.config = config
        self.evidence_rows = len(accepted)
        self.excluded_origin_id = origin_id

    def choose(self, state, available_routes):
        result = self.task.choose(state, available_routes,
            self.config['minimum_completion'], self.config['probability_tolerance'], self.config['time_tolerance_s'])
        return dict(result, evidence_rows=self.evidence_rows, excluded_origin_id=self.excluded_origin_id,
            scope='Empirical route recommendation; unavailable support retains current execution, not proof of safety')
