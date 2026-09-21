"""256/128 LeakyReLU residual Actor and identity-bound inference export.

This is an Actor architecture, not an implementation of the paper's TD3.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from flax import linen as nn, serialization
import jax
import jax.numpy as jp
from .observation import FIELDS, PATH_FIELDS, TRACKING_FIELDS, TIMED_FIELDS, PRIORITY_V2_FIELDS
from .action_mapping import ACTION_FIELDS


def make_policy_identity(model_identity,config,history_steps):
    """Bind mesh assets, loader transform and runtime as well as XML/config."""
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    config=dict(config)
    if config.get("forward_speed_source","wheel")=="wheel":config.pop("forward_speed_source",None)
    if config.get('timed_reference') is not None:
        config['timed_reference']=dict(config['timed_reference'])
        if config['timed_reference'].get('mode','time')=='time':
            for k in ('mode','geometry_stride','projection_margin','extension_seconds'):
                config['timed_reference'].pop(k,None)
        for key,default in (('training_mix',False),('priority_v2_screen',False),('fixed_scenario',None),
                            ('fast_speed_slew',1.),('fast_yaw_slew',2.4),
                            ('gentle_yaw_rate',.35)):
            if config['timed_reference'].get(key,default)==default:
                config['timed_reference'].pop(key,None)
    if config.get('tracking') is not None:
        config['tracking']=dict(config['tracking'])
        if config['tracking'].get('objective')!='soft_budget_v1':
            for key in list(config['tracking']):
                if key.startswith('soft_'):config['tracking'].pop(key)
        if not config['tracking'].get('precision_reward',False):
            for key in list(config['tracking']):
                if key.startswith('precision_'):config['tracking'].pop(key)
        for k,v in (('objective','legacy'),('deadline_penalty',0.),('overdue_rate',0.)):
            if config['tracking'].get(k,v)==v:config['tracking'].pop(k,None)
        for key,default in (('final_overspeed_tolerance',.2),('overspeed_band',.2)):
            if config['tracking'].get(key,default)==default:
                config['tracking'].pop(key,None)
    if config.get("timed_reference") is None:config.pop("timed_reference",None)
    else:
        config['timed_reference']=dict(config['timed_reference'])
        if config['timed_reference'].get('recovery_probability',0.)==0.:
            for key in ('recovery_probability','recovery_start','conflict_start_window'):
                config['timed_reference'].pop(key,None)
    if config.get('tracking') is not None:
        config['tracking']=dict(config['tracking'])
        for key,default in [('geometric',False),('reward_mode','gaussian'),('shrink_tolerances',False),('deadline_penalty',0.),('over_deadline_rate',0.)]:
            if config['tracking'].get(key,default)==default:config['tracking'].pop(key,None)

    if config.get("tracking") is not None and not config["tracking"].get("timed",False):
        config["tracking"]=dict(config["tracking"])
        for k in list(config["tracking"]):
            if k=="timed" or k.startswith(("longitudinal_","yaw_rate_","final_longitudinal_","final_yaw_rate_")):config["tracking"].pop(k)
    if config.get("reference_paths") is None:config.pop("reference_paths",None)
    else:
        config['reference_paths']=dict(config['reference_paths'])
        if not config['reference_paths'].get('continuous_projection',False):
            config['reference_paths'].pop('continuous_projection',None)
            config['reference_paths'].pop('projection_margin',None)
    if config.get("speed_schedule") is None:config.pop("speed_schedule",None)
    if config.get("tracking") is None:config.pop("tracking",None)
    if "actuator" in config:
        config["actuator"]=dict(config["actuator"])
        if config["actuator"].get("base_output_scale",1.)==1.:
            config["actuator"].pop("base_output_scale",None)
        if not config["actuator"].get("project_base",False):config["actuator"].pop("project_base",None)
        if config["actuator"].get("composition","additive")=="additive":
            config["actuator"].pop("composition",None)
    if config.get("motion_commands") is None:config.pop("motion_commands",None)
    else:
        config['motion_commands']=dict(config['motion_commands'])
        if config['motion_commands'].get('residual_gate_min') is None:config['motion_commands'].pop('residual_gate_min',None)
        if config['motion_commands'].get('yaw_tracking_reward_rate',0.)==0.:
            for key in ('yaw_tracking_reward_rate','yaw_tracking_reward_scale'):
                config['motion_commands'].pop(key,None)
        if config['motion_commands'].get('tolerance_penalty_rate',0.)==0.:
            for key in ('tolerance_penalty_rate','speed_tolerance','yaw_tolerance','speed_excess_scale','yaw_excess_scale'):
                config['motion_commands'].pop(key,None)
        if config['motion_commands'].get('tracking_priority_ratio')==100.:
            config['motion_commands'].pop('tracking_priority_ratio')
    if config.get("learning_roll_reference") is None:config.pop("learning_roll_reference",None)
    if config.get("alive_reward_rate")==1.:config.pop("alive_reward_rate")
    if config.get("preparation_seconds",0.)==0.:
        config.pop("preparation_seconds",None);config.pop("preparation_base_output_scale",None)
    if config.get('priority') is None:config.pop('priority',None)
    else:
        config['priority']=dict(config['priority'])
        if config['priority'].get('training_alphas') is None:config['priority'].pop('training_alphas',None)
        for key,value in [('weight_schedule','linear'),('weight_base',10.),('tracking_weight_scale',1.),('risk_gate',True),('speed_cost_scale',1.),('path_cost_scale',1.)]:
            if config['priority'].get(key)==value:config['priority'].pop(key)
    if 'observation' in config:
        config['observation']=dict(config['observation'])
        if not config['observation'].get('include_priority_v2',False):config['observation'].pop('include_priority_v2',None)
        if not config['observation'].get('include_timed',False):config['observation'].pop('include_timed',None)
        if not config['observation'].get('include_tracking',False):config['observation'].pop('include_tracking',None)
        if not config['observation'].get('include_motion',False):config['observation'].pop('include_motion',None)
        if config['observation'].get('include_attitude_risk') is True:config['observation'].pop('include_attitude_risk')
        if config['observation'].get('include_priority') is False:config['observation'].pop('include_priority')
    if config.get('rear_disturbance_mode','torque')=='torque':config.pop('rear_disturbance_mode',None)
    if config.get('bend') is None:config.pop('bend',None)
    if config.get('disturbance_rear_torque',0.)==0:config.pop('disturbance_rear_torque',None)
    if config.get('random_events') is not None:
        config['random_events']=dict(config['random_events'])
        for k,v in [('rear_probability',0.),('rear_min',.02),('rear_max',.08)]:
            if config['random_events'].get(k)==v:config['random_events'].pop(k)
    if config.get("figure_eight") is None:config.pop("figure_eight",None)
    # New optional behavior must not invalidate existing direct-action models.
    if config.get('action_mapping') is None:config.pop('action_mapping',None)
    if config.get('action_mapping') is not None:
        mapping=dict(config['action_mapping'])
        for key,value in [('authority_aware',False),('lateral_weight',1.),('speed_weight',1.),('regularization',1e-6)]:
            if mapping.get(key)==value:mapping.pop(key)
        config['action_mapping']=mapping
    identity={'model_sha256':digest(model_identity),'config_sha256':digest(config),'history_steps':history_steps}
    if config.get('observation',{}).get('include_path',False):
        identity['observation_fields']=list(FIELDS+PATH_FIELDS)
    if (config.get('figure_eight') is not None or config.get('bend') is not None) and config.get('observation',{}).get('include_path',False):
        identity['observation_fields']=list(FIELDS)+['path_right_error','heading_error','path_curvature']
    if config.get('observation',{}).get('include_motion',False):identity['observation_fields']=list(FIELDS)+['yaw_rate_reference','yaw_rate_world']
    if config.get('priority') is not None:
        identity['observation_fields']=identity.get('observation_fields',list(FIELDS))+['speed_priority']+(['attitude_risk'] if config.get('observation',{}).get('include_attitude_risk',True) else [])
    if config.get('observation',{}).get('include_tracking',False):
        identity['observation_fields'][15]='path_lateral_error'
        identity['observation_fields'] += list(TRACKING_FIELDS)
    if config.get('observation',{}).get('include_timed',False):identity['observation_fields'] += list(TIMED_FIELDS)
    if config.get('observation',{}).get('include_priority_v2',False):identity['observation_fields'] += list(PRIORITY_V2_FIELDS)
    if config.get('action_mapping') is not None:identity['action_fields']=ACTION_FIELDS
    if config.get("actuator",{}).get("composition")=="full_range":
        identity["action_fields"]=["steer_available_range_fraction","rear_available_range_fraction"]
    return identity


class ResidualActor(nn.Module):
    hidden_sizes: tuple=(256,128)
    negative_slope: float=.01
    activation: str="leaky_relu"

    @nn.compact
    def __call__(self,observation,return_logits=False):
        x=observation
        for width in self.hidden_sizes:
            x=nn.Dense(width)(x)
            # Match torch.nn.ELU used by the RSL actor export. Flax's ELU uses
            # a numerically different expm1 path that drifts by ~1e-5 per deep
            # network and breaks the bound checkpoint inference contract.
            if self.activation=="elu":x=jp.where(x>0,x,jp.exp(x)-1.)
            elif self.activation=="leaky_relu":x=nn.leaky_relu(x,negative_slope=self.negative_slope)
            else:raise ValueError("unsupported actor activation")
        logits=nn.Dense(2)(x)
        return logits if return_logits else jp.tanh(logits)


def save_policy(path,params,mean,std,identity,*,hidden_sizes=(256,128),negative_slope=.01,activation="leaky_relu"):
    path=Path(path)
    mean,std=np.asarray(mean),np.asarray(std)
    fields=identity.get('observation_fields',list(FIELDS))
    size=int(identity['history_steps'])*(len(fields)+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid observation normalization')
    payload=serialization.to_bytes(params)
    meta={'schema':'sttw_actor_v1','identity':identity,'fields':fields,
          'actions':identity.get('action_fields',['steer_rate_residual','rear_rate_residual']),'hidden_sizes':list(hidden_sizes),
          'negative_slope':negative_slope,'activation':activation,'mean':mean.tolist(),'std':std.tolist(),
          'payload_sha256':hashlib.sha256(payload).hexdigest()}
    # Validate dimensions before publishing a checkpoint directory.
    ResidualActor(tuple(hidden_sizes),negative_slope,activation).apply(params,jp.zeros(size))
    path.mkdir(parents=True,exist_ok=False)
    (path/'actor.msgpack').write_bytes(payload)
    (path/'identity.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


def load_policy(path,*,expected,return_logits=False):
    path=Path(path)
    meta=json.loads((path/'identity.json').read_text())
    if meta.get('schema')!='sttw_actor_v1' or meta['identity']!=expected or meta['fields']!=expected.get('observation_fields',list(FIELDS)) or meta['actions']!=expected.get('action_fields',['steer_rate_residual','rear_rate_residual']):
        raise ValueError('policy observation/action/identity mismatch')
    payload=(path/'actor.msgpack').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=meta['payload_sha256']:
        raise ValueError('policy payload hash mismatch')
    mean,std=np.asarray(meta['mean']),np.asarray(meta['std'])
    size=int(expected['history_steps'])*(len(expected.get('observation_fields',FIELDS))+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid normalization')
    actor=ResidualActor(tuple(meta['hidden_sizes']),meta['negative_slope'],meta.get('activation','leaky_relu'))
    template=actor.init(jax.random.PRNGKey(0),jp.zeros(size))
    params=serialization.from_bytes(template,payload)
    return jax.jit(lambda obs: actor.apply(params,(obs-jp.asarray(mean))/jp.asarray(std),return_logits=return_logits))


_MODE_PRIORITY_KEYS = ('randomize_alpha', 'fixed_alpha', 'training_alphas', 'validation_alphas')

def check_mode_transfer_contract(source_task, target_task):
    """Only sampling/selection of alpha may change for the isolated V1 experiment.

    Does NOT permit reward, observation, vehicle, actuator, reference, preparation,
    controller or timing changes. Fixed alpha still enters the original Actor.
    """
    source=json.loads(json.dumps(source_task));target=json.loads(json.dumps(target_task))
    if source.get('tracking',{}).get('objective')!='soft_budget_v1':
        raise ValueError('independent-mode transfer currently requires the frozen V1 reward')
    priority=target.get('priority') or {}
    alpha=priority.get('fixed_alpha')
    if priority.get('randomize_alpha') is not False or alpha not in (0.,.5,1.):
        raise ValueError('target must declare a fixed discrete alpha 0/.5/1')
    if list(priority.get('training_alphas') or []) != [alpha]:
        raise ValueError('target training_alphas must contain ONLY its fixed alpha')
    for task in (source,target):
        task['priority']=dict(task.get('priority') or {})
        for key in _MODE_PRIORITY_KEYS:task['priority'].pop(key,None)
    if source!=target:
        changed=sorted(key for key in set(source)|set(target) if source.get(key)!=target.get(key))
        raise ValueError('Actor transfer would silently change task fields: '+','.join(changed))
    return float(alpha)


def verified_actor_initialization(checkpoint, source_training, target_task, model_identity,
                                  mean, std, hidden_sizes, activation):
    """Read a safe msgpack Actor, validate original identity then allow alpha-only transfer.

    No torch.load/pickle, Critic, optimizer, RNG or simulated state is imported.
    """
    cp=Path(checkpoint).resolve();source=Path(source_training).resolve()
    declaration=json.loads((source/'declaration.json').read_text())
    if not json.loads((source/'status.json').read_text()).get('complete'):
        raise ValueError('source training is not complete')
    if not cp.is_relative_to(source/'checkpoints'):
        raise ValueError('initial Actor must be a checkpoint from the declared source training')
    alpha=check_mode_transfer_contract(declaration['task'],target_task)
    expected=make_policy_identity(model_identity,declaration['task'],target_task['observation']['history_steps'])
    side=json.loads((cp/'identity.json').read_text())
    if declaration['policy_identity']!=expected or side['identity']!=expected:
        raise ValueError('source model/runtime/task identity does not match; do not bypass this guard')
    if tuple(side['hidden_sizes'])!=tuple(hidden_sizes) or side.get('activation','leaky_relu')!=activation:
        raise ValueError('source and target Actor architectures differ')
    if not np.array_equal(np.asarray(side['mean']),np.asarray(mean)) or not np.array_equal(np.asarray(side['std']),np.asarray(std)):
        raise ValueError('observation normalization differs')
    load_policy(cp,expected=expected)  # validates finite normalization and payload hash
    payload=(cp/'actor.msgpack').read_bytes()
    params=serialization.msgpack_restore(payload)
    record={'checkpoint':str(cp),'source_training':str(source),'fixed_alpha':alpha,
            'actor_payload_sha256':hashlib.sha256(payload).hexdigest(),
            'source_declaration_sha256':hashlib.sha256((source/'declaration.json').read_bytes()).hexdigest(),
            'scope':'Actor weights only; fresh Critic, Adam, exploration std, RNG and physical preparation'}
    return params,record



def load_independent_mode_bundle(path, source_task, model_identity):
    """Read three declared full Actors; do not blend actions or swap private heads."""
    path=Path(path).resolve();bundle=json.loads(path.read_text())
    if bundle.get('schema')!='sttw_independent_modes_v1' or not bundle.get('complete'):
        raise ValueError('incomplete independent-mode bundle')
    if set(bundle.get('modes',{}))!={'0.0','0.5','1.0'}:raise ValueError('bundle must have exactly 0/.5/1')
    policies={};metadata={}
    for key,item in bundle['modes'].items():
        cp=Path(item['checkpoint']);training=Path(item['training'])
        if not cp.is_absolute():cp=(path.parent/cp).resolve()
        if not training.is_absolute():training=(path.parent/training).resolve()
        if not cp.is_relative_to(training/'checkpoints'):raise ValueError('bundle checkpoint outside declared training')
        if not json.loads((training/'status.json').read_text()).get('complete'):raise ValueError('mode training incomplete')
        d=json.loads((training/'declaration.json').read_text())
        alpha=check_mode_transfer_contract(source_task,d['task'])
        if alpha!=float(key):raise ValueError('mode label and fixed alpha disagree')
        payload=(cp/'actor.msgpack').read_bytes()
        if hashlib.sha256(payload).hexdigest()!=item['actor_payload_sha256']:raise ValueError('mode Actor changed')
        if hashlib.sha256((training/'declaration.json').read_bytes()).hexdigest()!=item['declaration_sha256']:raise ValueError('mode declaration changed')
        init=json.loads((training/'actor_initialization.json').read_text())
        if init.get('actor_payload_sha256')!=bundle.get('source_actor_sha256'):
            raise ValueError('bundle modes did not share the declared source Actor')
        expected=make_policy_identity(model_identity,d['task'],d['task']['observation']['history_steps'])
        if expected!=d['policy_identity']:raise ValueError('mode runtime/task identity differs')
        policies[alpha]=load_policy(cp,expected=expected)
        metadata[key]={**item,'checkpoint':str(cp),'training':str(training),'scope':'independent full Actor selected by alpha; no action averaging'}
    return policies,metadata
