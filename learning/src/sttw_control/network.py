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
from .observation import FIELDS, PATH_FIELDS, TRACKING_FIELDS, TIMED_FIELDS
from .action_mapping import ACTION_FIELDS


def make_policy_identity(model_identity,config,history_steps):
    """Bind mesh assets, loader transform and runtime as well as XML/config."""
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    config=dict(config)
    if config.get('timed_reference') is not None:
        config['timed_reference']=dict(config['timed_reference'])
        if config['timed_reference'].get('mode','time')=='time':
            for k in ('mode','geometry_stride','projection_margin','extension_seconds'):
                config['timed_reference'].pop(k,None)
        for key,default in (('training_mix',False),('fixed_scenario',None),
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


def load_policy(path,*,expected):
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
    return jax.jit(lambda obs: actor.apply(params,(obs-jp.asarray(mean))/jp.asarray(std)))


def alpha_sampling_compatible(source, target):
    """Only fixed/random alpha sampling may change during Actor-only transfer.

    Full rewards, path generation, history layout, baseline, actuator and physics
    must match. In particular this is NOT permission to change a failed task.
    """
    import copy
    a,b=copy.deepcopy(source),copy.deepcopy(target)
    for x in (a,b):
        if not isinstance(x.get('priority'),dict):return False
        for key in ('fixed_alpha','randomize_alpha','training_alphas','validation_alphas'):
            x['priority'].pop(key,None)
    return a==b


def resolve_actor_checkpoint(value):
    """Accept a checkpoint or completed training run. Never silently choose last."""
    if value is None:raise ValueError('Actor initialization path is required')
    path=Path(value).resolve()
    if (path/'identity.json').is_file() and (path/'actor.msgpack').is_file():return path
    if (path/'best_model.json').is_file():
        best=json.loads((path/'best_model.json').read_text())
        candidate=Path(best['checkpoint'])
        if candidate.is_dir():return candidate.resolve()
        # A moved training directory may retain absolute old checkpoint paths.
        moved=path/'checkpoints'/candidate.name
        if moved.is_dir():return moved.resolve()
    raise ValueError('explicit Actor checkpoint or training best is missing; never substitute last')


def actor_transfer_data(value,target_config,model_identity,mean,std,hidden_sizes,activation):
    from dataclasses import asdict
    from .env import config_from_dict
    path=resolve_actor_checkpoint(value)
    declaration=path.parent.parent/'declaration.json'
    if not declaration.is_file():raise ValueError('initializer needs its source training declaration')
    raw=json.loads(declaration.read_text());source=asdict(config_from_dict(raw['task']))
    target=asdict(target_config)
    if not alpha_sampling_compatible(source,target):
        raise ValueError('initializer changed reward/model/control/reference, not merely alpha sampling')
    expected=make_policy_identity(model_identity,source,target_config.observation.history_steps)
    load_policy(path,expected=expected)  # checks source payload, shape and immutable identity
    meta=json.loads((path/'identity.json').read_text())
    if tuple(meta['hidden_sizes'])!=tuple(hidden_sizes) or meta.get('activation','leaky_relu')!=activation:
        raise ValueError('initializer architecture mismatch')
    if not np.array_equal(np.asarray(meta['mean']),mean) or not np.array_equal(np.asarray(meta['std']),std):
        raise ValueError('initializer normalization mismatch')
    params=serialization.msgpack_restore((path/'actor.msgpack').read_bytes())
    return params,{'checkpoint':str(path),'actor_sha256':hashlib.sha256((path/'actor.msgpack').read_bytes()).hexdigest(),
                   'source_identity':expected,'transfer':'Actor weights only; unchanged task except explicit alpha sampling'}


def load_expert_bundle(root,model_identity,*,endpoint='best'):
    """Load one independently trained Actor per alpha; never average actions."""
    from dataclasses import asdict
    from .env import config_from_dict
    root=Path(root).resolve()
    bundle=json.loads((root/'experts.json').read_text())
    if bundle.get('schema')!='sttw_expert_bundle_v1' or not bundle.get('complete'):
        raise ValueError('incomplete or unknown expert bundle')
    if endpoint not in ('best','last','initial'):raise ValueError('invalid expert endpoint')
    task=asdict(config_from_dict(bundle['task']));members=bundle['members']
    if not json.loads((root/'status.json').read_text()).get('complete'):raise ValueError('expert orchestration incomplete')
    if sorted(m['alpha'] for m in members)!=[0.,.5,1.]:raise ValueError('bundle must contain each exact alpha once')
    policies={};identities={}
    for member in members:
        alpha=member['alpha'];run=Path(member['training'])
        # Relocation preserves the explicit mode directory and never guesses checkpoints.
        if not run.exists():run=root/f'alpha_{alpha:g}'/'training'
        stage=json.loads((run/'declaration.json').read_text())
        cfg=config_from_dict(stage['task'])
        if cfg.priority.randomize_alpha or cfg.priority.fixed_alpha!=alpha or not alpha_sampling_compatible(task,asdict(cfg)):
            raise ValueError('expert task or fixed-alpha contract differs')
        if not json.loads((run/'status.json').read_text()).get('complete'):
            raise ValueError('unfinished expert')
        chosen=(Path(member['sampled_best']) if endpoint=='best' else Path(member['last_checkpoint']) if endpoint=='last'
                else run/'checkpoints/update_0000')
        if not chosen.exists():chosen=run/'checkpoints'/chosen.name
        if endpoint in ('best','last') and member.get(endpoint+'_actor_sha256') is not None:
            if hashlib.sha256((chosen/'actor.msgpack').read_bytes()).hexdigest()!=member[endpoint+'_actor_sha256']:
                raise ValueError('expert Actor changed after bundle publication')
        expected=make_policy_identity(model_identity,asdict(cfg),cfg.observation.history_steps)
        policies[alpha]=load_policy(chosen,expected=expected)
        identities[alpha]={**expected,'checkpoint':str(chosen.resolve()),
            'actor_sha256':hashlib.sha256((chosen/'actor.msgpack').read_bytes()).hexdigest(),
            'selection':endpoint,'fixed_training_alpha':alpha,
            'scope':'independent expert; evaluation overrides alpha sampling only'}
    return task,policies,identities
