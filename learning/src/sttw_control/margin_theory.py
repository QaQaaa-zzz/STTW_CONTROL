"""Nominal roll recovery and geometric tracking margins (SI units).

Fixed-speed, small-angle model; external contact inputs are not implicit in
these expressions. Rate/travel limits are explicit arguments, not alpha gains.
"""
import numpy as np


def coefficients(speed, config):
    v=np.maximum(np.asarray(speed),config['minimum_speed'])
    m=config['mass'];h=config['cg_height']
    inertia=m*h*h+config.get('roll_inertia_com',.043929*m/5.434)
    c=-m*config['cg_forward']*h*v*np.cos(config['caster'])/config['wheelbase']/inertia
    moment=-(m*v*v*h-m*config['cg_forward']*config['trail']*config['gravity'])*np.cos(config['caster'])/config['wheelbase']
    b=np.where(moment<0,-1,1)*np.maximum(abs(moment),.1)/inertia
    return c,b,m*config['gravity']*h/inertia


def gains(speed, config):
    c,b,a=coefficients(float(speed),config);w=config['wc']
    return np.linalg.solve([[1,0,c],[0,c,b],[-a,b,0]],[3*w,3*w*w+a,w**3])


def constant_rate_segment(state, time, c, b, a, rate):
    p,w,d=state;lam=np.sqrt(a);ch=np.cosh(lam*time);sh=np.sinh(lam*time)
    f0=b*d+c*rate;f1=b*rate
    return (p*ch+w*sh/lam+f0*(ch-1)/a+f1*(sh/lam-time)/a,
            p*lam*sh+w*ch+f0*sh/lam+f1*(ch-1)/a,d+rate*time)


def optimal_envelope(speed,state,time,config,rate_limit=3.,travel_limit=.8):
    """Most favorable roll in direction of current angular velocity.

Returned state is oriented so that initial angular velocity is nonnegative;
second result is the orientation sign. Domain: a>0, b<0, c<0, initial travel valid.
"""
    c,b,a=coefficients(speed,config);side=1 if state[1]>=0 else -1
    oriented=side*np.asarray(state);hit=(travel_limit-oriented[2])/rate_limit
    first=constant_rate_segment(oriented,np.minimum(time,hit),c,b,a,rate_limit)
    return constant_rate_segment(first,np.maximum(time-hit,0),c,b,a,0),side


def impulse_response(speed,time,config):
    """Roll, roll rate, steering angle and steering rate for unit roll-rate impulse."""
    c,b,a=coefficients(speed,config);kd,kphi,kp=gains(speed,config);w=config['wc']
    t=np.asarray(time);e=np.exp(-w*t);h=(kd-w)/2
    phi=e*(t+h*t*t)
    omega=e*(1+(2*h-w)*t-w*h*t*t)
    acc=e*(2*h-2*w+(w*w-4*w*h)*t+w*w*h*t*t)
    delta=(acc-(a-c*kphi)*phi+c*kp*omega)/(b-c*kd)
    rate=-kd*delta-kphi*phi-kp*omega
    return phi,omega,delta,rate


def impulse_capacity(speed,config,rate_limit=3.,horizon=3.,step=.0005):
    t=np.arange(round(horizon/step)+1)*step
    phi,omega,delta,rate=impulse_response(speed,t,config)
    delay=config['wheelbase']/speed
    p1=float(impulse_response(speed,np.array(delay),config)[1])
    return dict(J_sat_rad_s=float(rate_limit/np.max(abs(rate))),wheel_delay_s=delay,
                p1_roll_rate=p1,peak_roll_per_unit_impulse=float(np.max(abs(phi))))


def tracking_margin(target_forward,min_forward=.05):
    """Use the actual follower target in body coordinates, not a path approximation."""
    return np.asarray(target_forward)-min_forward


def local_tracking_margin(lookahead,cross_track_left,heading_error,min_forward=.05):
    """Straight local path; heading_error = vehicle heading minus path heading."""
    return lookahead*np.cos(heading_error)-cross_track_left*np.sin(heading_error)-min_forward


def tracking_domain_margins(target_forward,heading_error,min_forward=.05):
    """Separate forward-distance (m) and heading (rad) margins of the follower."""
    wrapped=(np.asarray(heading_error)+np.pi)%(2*np.pi)-np.pi
    return tracking_margin(target_forward,min_forward),np.pi/2-abs(wrapped)


def recovery_outcome(speed,state,config,rate_limit=3.,travel_limit=.8,roll_limit=.7,horizon=1.,step=.00025):
    t=np.arange(round(horizon/step)+1)*step
    (phi,omega,delta),side=optimal_envelope(speed,state,t,config,rate_limit,travel_limit)
    stopped=np.flatnonzero(omega<=0);crossed=np.flatnonzero(phi>roll_limit)
    stop=int(stopped[0]) if len(stopped) else len(t)
    cross=int(crossed[0]) if len(crossed) else len(t)
    outcome='braked' if stop<cross else 'unavoidable_nominal_limit' if cross<len(t) else 'unresolved'
    i=min(stop,cross,len(t)-1)
    return dict(outcome=outcome,event_s=float(t[i]),phi_event_deg=float(np.degrees(side*phi[i])),angle_certificate_excess_rad=float(phi[i]-roll_limit) if outcome=='unavoidable_nominal_limit' else None)
