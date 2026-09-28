"""Kinematic user intent only. Never used as the physical predictor."""
import jax.numpy as jp

def integrate(pose,command,dt=.005,wheelbase=.408,caster=25*jp.pi/180):
    v,delta=command
    theta=v*jp.cos(caster)*jp.tan(delta)/wheelbase*dt
    distance=v*dt*jp.sinc(theta/(2*jp.pi))
    direction=pose[2]+theta/2
    return pose+jp.array([distance*jp.cos(direction),distance*jp.sin(direction),theta])

def unwrap(accumulated,previous_wrapped,current_wrapped):
    delta=current_wrapped-previous_wrapped
    return accumulated+jp.arctan2(jp.sin(delta),jp.cos(delta))
