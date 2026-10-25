"""Measured-rate relay between frozen reference Actors; no event-time query."""
import jax
import jax.numpy as j
from flax import struct
from .entry_recovery_factorial import factorial_inventory
from .reference_reserve_gate import initial_gate,gate_step

MODES=('AA','BB','causal_B','causal_support','constant_support')


@struct.dataclass
class CausalRelayState:
    state: object
    gate: object
    clean_blocks: object
    relay: object


def relay_choice(mode,active):
    return (mode==0)&active,j.where((mode==2)|((mode==1)&active),7,0)


def relay_inventory(world_direction):
    cases=[]
    for old in factorial_inventory(world_direction):
        if old['mode']!='AA':continue
        for mode,name in enumerate(MODES[2:]):
            c={k:v for k,v in old.items() if k not in ['mode','mode_index','handoff_tick','case_id','cache_candidate']}
            cases.append(dict(c,case_id=f'a{old["alpha"]:g}/{old["event_id"]}/{name}',mode=name,mode_index=mode,cache_candidate=False))
    return cases


def make_relay_step(env,actor,*,horizon_s):
    from .anticipatory_reference import make_reference_step
    advance=make_reference_step(env,actor,horizon_s=horizon_s,allow_reentry=True)
    lower_bound=j.float32(env.config['residual_interface']['speed_reference_bounds_m_s'][0])
    dt=env.config['timing']['upper_dt_s']
    def step(carry,parameters,path,speed,pulse,mode):
        # Observe the current physical state before advancing any new tick.
        measurement,_,_=env.physics.observe(carry.state.physical.data)
        roll,roll_rate,steer,steer_rate,yaw_rate,rear_omega,front_omega=measurement
        relay,_,receipt=gate_step(carry.relay,j.zeros(2),rho=j.float32(0),roll_rate=roll_rate,mode=2,dt=dt)
        use_b,profile=relay_choice(mode,receipt['active'])
        selected=jax.tree.map(lambda a,b:j.where(use_b,b,a),parameters['A'],parameters['B'])
        s=carry.state;stopped=s.physical.failed|s.fault|(s.tick>=round(horizon_s/.005))
        before=s.tick
        supplied=carry.replace(state=s.replace(fault=s.fault|(~stopped&receipt['fault'])),relay=relay)
        end,logs=advance(supplied,selected,path,speed,pulse,profile,lower_bound,j.bool_(False))
        active=logs['active_tick'];legal=logs['actor_action_valid']
        return end,dict(logs,desired_B=active&use_b,selected_B=active&legal&use_b,
            relay_active=active&receipt['active'],relay_trigger=active&receipt['trigger'],
            relay_remaining_s=j.where(active,relay.remaining_s,j.nan),
            relay_input_roll_rate=j.where(active,roll_rate,j.nan),
            relay_profile=j.repeat(profile,4),decision_task_tick=j.repeat(before,4))
    return step


def relay_contrasts(rows):
    if not rows or len({x['case_id'] for x in rows})!=len(rows):raise ValueError('unique nonempty rows required')
    groups={}
    for r in rows:
        g=groups.setdefault((r['alpha'],r['event_id']),{})
        if r['mode'] in g or r['mode'] not in MODES:raise ValueError('one case per relay cell required')
        g[r['mode']]=r
    if any(set(g)!=set(MODES) for g in groups.values()):raise ValueError('all five controls required')
    result={}
    for target,base in [('causal_B','AA'),('causal_support','AA'),('constant_support','AA'),('causal_B','causal_support'),('causal_B','constant_support'),('causal_B','BB')]:
        counts=dict(pairs=len(groups),known=0,unknown=0,physical_gains=0,physical_losses=0,task_gains=0,task_losses=0,details=[])
        for key,g in groups.items():
            a,b=g[target],g[base];unknown='unknown' in [a['category'],b['category']]
            p=None if unknown else int(a['category']!='contact')-int(b['category']!='contact')
            t=None if unknown else int(a['category']=='safe_task')-int(b['category']=='safe_task')
            counts['unknown' if unknown else 'known']+=1
            if not unknown:
                counts['physical_gains']+=int(p==1);counts['physical_losses']+=int(p==-1)
                counts['task_gains']+=int(t==1);counts['task_losses']+=int(t==-1)
            counts['details'].append(dict(alpha=key[0],event_id=key[1],target_case=a['case_id'],baseline_case=b['case_id'],physical_change=p,task_change=t,scored_cost_change=a['scored_cost']-b['scored_cost']))
        result[target+'_vs_'+base]=counts
    return result
