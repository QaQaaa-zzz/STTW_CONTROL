"""Diagnostic-only frozen210D local residual, without task clocks or screening."""
import jax.numpy as jp
from flax import struct
from .lower_command import SCALES
from .direct_command_policy import make_frame
from .controller import _system
from .network import ResidualActor

@struct.dataclass
class LocalLowerState:
    frames:object
    mask:object
    previous_applied:object
    yaw_rate:object

class FrozenLocalLower:
    local_interface=True
    def __init__(self,params):
        self.params=params;self.net=ResidualActor((256,128),activation='elu')
        if params['params']['Dense_0']['kernel'].shape!=(210,256):raise ValueError('local210 frozen Actor required')
    def initial(self,pose):
        return LocalLowerState(jp.zeros((10,20)),jp.zeros(10),jp.zeros(2),jp.asarray(0.))
    def prepare_local(self,s,measurement,pose,true_speed,governed,physical,cc):
        previous=physical.governor.current_reference
        rates=(governed-previous)/cc.dt
        m=measurement.at[4].set(s.yaw_rate)
        _,_,a4,_,_=_system(m[5]*.1,physical.controller.gains,cc)
        frame=make_frame(measurement=m,forward_speed=true_speed,previous_governed=previous,
            previous_final_command=physical.actuator.previous,previous_bounded_residual=s.previous_applied,
            raw=governed,raw_rates=rates,eso_equilibrium_shift=-physical.controller.disturbance/a4,cc=cc)
        s=s.replace(frames=jp.concatenate([s.frames[1:],frame[None]]),mask=jp.concatenate([s.mask[1:],jp.ones(1)]))
        obs=jp.concatenate([jp.clip(s.frames/SCALES,-5.,5.).reshape(-1),s.mask])
        action=self.net.apply(self.params,obs)
        return s,action,obs,dict(finite=jp.all(jp.isfinite(obs))&jp.all(jp.isfinite(action)),
            path_features=jp.zeros(3),path_endpoint_extension=jp.bool_(False))
    def finish_local(self,s,applied_residual,yaw_rate):
        return s.replace(previous_applied=applied_residual,yaw_rate=yaw_rate)
