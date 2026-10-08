"""Resolve the supplied SmoothV4 endpoint specifications onto the real V3 interfaces."""
from pathlib import Path
import json

def resolve(alpha, *, fresh=False, train_wall=1800, total_wall=5400):
    root=Path(__file__).resolve().parents[3]
    s=json.loads((root/f'learning/configs/upper_STTW_R196_ALPHA1_alpha{alpha}.json').read_text())
    v=json.loads((root/f'docs/smooth_v4/attachment/STTW_R196_SmoothV4_alpha{alpha}.json').read_text())
    assert v['fixed_upper_alpha']==alpha
    s['smooth_v4']=v;s['reward']=v['reward'];s['actor_temporal_regularizer']=v['actor_temporal_regularizer']
    for k,x in v['action'].items():
        if k in s['action'] and isinstance(x,(float,int)):assert s['action'][k]==x,(k,s['action'][k],x)
    s['ppo'].update(v['ppo']);s['ppo'].update(seed=81,target_mean_kl_after_epoch=.01,hard_stop_mean_kl=.03,default_updates=60,future_total_updates_only_after_user_approval=60)
    c=s['commands'];t=v['training_commands']
    for k in ['nominal_fraction','conflict_fraction','random_fraction']:c[k]=t[k]
    c['random_seed']=81;c['nominal']['straight_probability']=t['nominal_straight_fraction']
    c['nominal']['steer_abs_max_rad']=t['nominal_steer_abs_max_rad']
    s['budget'].update(additional_compute_wall_seconds=5400,compile_wall_seconds=600,training_wall_seconds=1800,evaluation_wall_seconds=1200,engineering_wall_seconds=120,default_control_transitions_upper=2*62*512*128*4)
    if fresh:
        s['initialization_mode']='scratch'
        s['smooth_v4']['upper']['warm_start_actor_only']=False
        s['smooth_v4']['ppo']['policy_updates_per_endpoint']=150
        s['smooth_v4']['ppo']['validation_updates']=[60,150]
        s['smooth_v4']['evaluation']['stage150_cases']=list(v['evaluation']['stage60_cases'])
        s['ppo'].update(default_updates=150,future_total_updates_only_after_user_approval=150,policy_updates_per_endpoint=150,validation_updates=[60,150])
        s['smooth_v4']['budget'].update(train_wall_per_endpoint_s=train_wall,overall_new_compute_wall_s=total_wall)
        s['budget'].update(training_wall_seconds=train_wall,additional_compute_wall_seconds=total_wall,default_control_transitions_upper=2*152*512*128*4)
    return s
