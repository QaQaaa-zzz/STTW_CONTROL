"""Observable risk and externally conditioned priorities; not a safety guarantee."""
from dataclasses import dataclass
import math
import jax.numpy as jp

@dataclass(frozen=True)
class PriorityConfig:
    risk_gate: bool=True
    speed_cost_scale: float=1.
    path_cost_scale: float=1.
    fixed_alpha: float=.5
    randomize_alpha: bool=True
    training_alphas: tuple | None=None
    error_warning: float=.15
    error_critical: float=.35
    roll_warning: float=.35
    roll_critical: float=.6
    rate_warning: float=.8
    rate_critical: float=1.8
    weight_schedule: str="linear"
    weight_base: float=10.
    tracking_weight_scale: float=1.
    objective_floor: float=.1
    motion_floor: float=.1
    attitude_boost: float=4.
    validation_alphas: tuple=(0.,.5,1.)

    def __post_init__(self):
        if self.training_alphas is not None:
            values=tuple(self.training_alphas)
            if not values or len(set(values))!=len(values) or any(not math.isfinite(x) or not 0<=x<=1 for x in values):
                raise ValueError('invalid training alphas')
            object.__setattr__(self,'training_alphas',values)
        if self.weight_schedule not in ("linear","exponential") or not math.isfinite(self.weight_base) or self.weight_base<=1 or not math.isfinite(self.tracking_weight_scale) or self.tracking_weight_scale<=0:
            raise ValueError("invalid priority weight schedule")
        if not self.validation_alphas or len(set(self.validation_alphas))!=len(self.validation_alphas) or any(not math.isfinite(x) or not 0<=x<=1 for x in self.validation_alphas):raise ValueError("invalid validation alphas")
        if not isinstance(self.risk_gate,bool) or any(not math.isfinite(x) or x<=0 for x in (self.speed_cost_scale,self.path_cost_scale)):raise ValueError('invalid priority cost normalization')
        if not math.isfinite(self.fixed_alpha) or not 0<=self.fixed_alpha<=1 or not isinstance(self.randomize_alpha,bool):raise ValueError('invalid priority alpha')
        for a,b in [(self.error_warning,self.error_critical),(self.roll_warning,self.roll_critical),(self.rate_warning,self.rate_critical)]:
            if not math.isfinite(a+b) or not 0<=a<b:raise ValueError('invalid risk thresholds')
        if not 0<self.objective_floor<=1 or not 0<self.motion_floor<=1 or not math.isfinite(self.attitude_boost) or self.attitude_boost<0:raise ValueError('invalid priority weights')


def sample_alpha(key,c):
    import jax
    if not c.randomize_alpha:return jp.asarray(c.fixed_alpha)
    key=jax.random.fold_in(key,31)
    if c.training_alphas is None:return jax.random.uniform(key)
    return jp.asarray(c.training_alphas)[jax.random.randint(key,(),0,len(c.training_alphas))]


def priority_weights(alpha,error,roll,rate,c):
    components=jp.array([(jp.abs(error)-c.error_warning)/(c.error_critical-c.error_warning),
                         (jp.abs(roll)-c.roll_warning)/(c.roll_critical-c.roll_warning),
                         (jp.abs(rate)-c.rate_warning)/(c.rate_critical-c.rate_warning)])
    x=jp.clip(jp.max(components),0,1);risk=x*x*(3-2*x)
    risk=jp.where(c.risk_gate,risk,0.)
    alpha=jp.clip(alpha,0,1)
    motion=1-(1-c.motion_floor)*risk
    speed,path=objective_weights(alpha,c)
    speed=motion*speed;path=motion*path
    return jp.array([risk,speed,path,1+c.attitude_boost*risk])


def objective_weights(alpha,c,xp=jp):
    """Shared reward/reconstruction weights, including fixed global scaling."""
    a=xp.clip(alpha,0,1)
    if c.weight_schedule=='exponential':
        speed=xp.power(c.weight_base,2*a-1)
        path=xp.power(c.weight_base,1-2*a)
    else:
        speed=c.objective_floor+2*(1-c.objective_floor)*a
        path=c.objective_floor+2*(1-c.objective_floor)*(1-a)
    return c.tracking_weight_scale*speed,c.tracking_weight_scale*path
