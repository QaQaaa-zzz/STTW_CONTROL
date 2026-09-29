"""200Hz control preview; unchanged ECBC equations and actuator composition."""
import jax.numpy as jp
from .controller import controller_step
from .actuator import apply_residual

def preview_controls(controller,measurement,raw,governed,enable_eso,cc):
    # Explicit [v,delta] -> legacy measurement [speed,steer,rate,roll,rate,delta].
    m=measurement
    row=jp.array([m[5]*.1,m[2],m[3],m[0],m[1],raw[1]])
    next_controller,nominal=controller_step(controller,row,enable_eso,cc)
    _,goal=controller_step(controller,row.at[5].set(governed[1]),enable_eso,cc)
    return next_controller,nominal,goal

def controls(controller,actuator,measurement,raw,governed,enable_eso,bypass,cc,ac,
             lower_action=None,previewed=None):
    m=measurement
    next_controller,nominal,goal=(preview_controls(controller,m,raw,governed,enable_eso,cc)
                                if previewed is None else previewed)
    u_nom=jp.array([nominal.steer_rate,raw[0]/.1])
    u_goal=jp.array([goal.steer_rate,governed[0]/.1])
    requested=jp.where(bypass,jp.zeros(2),u_goal-u_nom)
    scales=jp.array([ac.steer_residual_scale,ac.rear_residual_scale])
    if lower_action is not None:requested=requested+scales*lower_action
    action=jp.clip(requested/scales,-1,1)
    applied=scales*action
    _,zero_final=apply_residual(actuator,u_nom,jp.zeros(2),m[2],ac)
    actuator,final=apply_residual(actuator,u_nom,action,m[2],ac)
    return next_controller,actuator,dict(u_nom=u_nom,u_goal=u_goal,
        requested_residual=requested,applied_residual=applied,
        normalized_residual=action,actual_normalized_residual=(final-zero_final)/scales,
        zero_residual_final_command=zero_final,u_prelimit=u_nom+applied,final_command=final,
        # Divide/multiply roundoff is not a permission-limit event.
        residual_clipped=jp.any(jp.abs(requested)>scales),
        final_command_clipped=jp.any(final!=u_nom+applied))
