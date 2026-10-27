"""Zero-physics audit of reference feasibility and geometric progress semantics.

Synthetic function arguments probe the interface only. They are not robot
states, trajectories, stopping experiments, or changes to task permissions.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path

import jax.numpy as j
import numpy as np

from sttw_control.geometric_path import Path as Route
from sttw_control.path_reference import pursuit
from sttw_control.path_command_policy import correction_tick
from sttw_control.goal_arrival_analysis import digest


def function_ast(path, name):
    tree=ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.worktree.resolve()
    native_plan=root/'runs/native_goal_confirmation_v1/plan.json'
    native=json.loads(native_plan.read_text())
    native_root=Path(native['native_runtime_root'])
    active_root=root/'learning/src/sttw_control'
    sources={}
    ast_checks=[]
    for filename,names in [('path_reference.py',['pursuit','publish']),
                           ('geometric_path.py',['project','at']),
                           ('path_command_policy.py',['map_latent','correction_tick'])]:
        a=active_root/filename;b=native_root/'sttw_control'/filename
        if digest(b)!=native['native_expected_sources'][str(b)]:
            raise ValueError('Archived native source differs from its execution plan: '+filename)
        sources[str(a)]=digest(a);sources[str(b)]=digest(b)
        for name in names:
            equal=function_ast(a,name)==function_ast(b,name)
            if not equal:
                raise ValueError('Current/native pure interface differs; do not silently transfer conclusion: '+name)
            ast_checks.append(dict(function=name,exact_function_ast_equal=equal))
    configs=[('capability90',root/'runs/preference_capability_grid_v1/evaluation/common_environment_config.json'),
             ('replanning72',root/'runs/goal_replanning_github_v1/common_environment_config.json'),
             ('native80',root/'runs/native_goal_confirmation_v1/effective_config.json')]
    probe=[]; at_goal=[]; profiles=[]
    s=j.array([0.,10.,20.]);xy=j.array([[0.,0.],[10.,0.],[20.,0.]])
    for label,file in configs:
        cfg=json.loads(file.read_text());sources[str(file)]=digest(file)
        plan_file=native_plan if label=='native80' else file.parent/'plan.json'
        declaration=json.loads(plan_file.read_text());sources[str(plan_file)]=digest(plan_file)
        if label=='native80':
            if cfg!=declaration['config']:
                raise ValueError('Native effective config differs from executed plan')
        else:
            bindings={**declaration.get('sealed_files',{}),**declaration.get('input_sha256',{})}
            matches=[sha for key,sha in bindings.items() if Path(key).resolve()==file.resolve()]
            if len(matches)!=1 or matches[0]!=digest(file):
                raise ValueError('Historical effective config not bound to its execution plan')
        lo,hi=cfg['residual_interface']['speed_reference_bounds_m_s']
        profiles.append(dict(profile=label,speed_reference_bounds_m_s=[lo,hi],
                             exact_zero_speed_reference_permitted=lo<=0<=hi))
        for nominal in [0.,.5,1.,1.5,3.]:
            for latent in [-100.,0.,100.]:
                _,_,issued,_=correction_tick(j.zeros(2),j.zeros(2),j.array([latent,0.]),j.array([nominal,0.]),cfg)
                issued=np.asarray(issued)
                if not np.isfinite(issued).all() or not lo-1e-6<=issued[0]<=hi+1e-6:
                    raise ValueError('Actual reference clip differs from configuration')
                probe.append(dict(profile=label,synthetic_nominal_speed_m_s=nominal,
                    latent_speed=latent,issued_speed_reference_m_s=float(issued[0]),
                    synthetic_function_probe_only=True))
        for marker in [5.,10.,15.]:
            path=Route(s,xy,j.zeros(3),j.zeros(3),j.float32(0.),j.float32(marker))
            # Keep real geometry, progress and pose identical; vary only goal.
            result=pursuit(path,j.float32(10.),j.array([10.,0.,0.]),j.float32(1.5),j.float32(1.5),cfg)
            command=np.asarray(result.command)
            at_goal.append(dict(profile=label,goal_marker_m=marker,pose_x_m=10.,
                projection_progress_m=10.,nominal_command=command.tolist(),
                follower_invalid=bool(result.invalid)))
        if not all(r['nominal_command']==[1.5,0.] for r in at_goal if r['profile']==label):
            raise ValueError('Follower goal-marker probe changed command unexpectedly')
    old=root/'runs/support_permission_factor_v1/analysis/path_governor_numerical_5/numerical.json'
    sources[str(old)]=digest(old);previous=json.loads(old.read_text())
    for filename in ['path_command_env.py','frozen_local_lower.py','path_reference_governor.py','native_goal_continuation.py']:
        file=active_root/filename;sources[str(file)]=digest(file)
    sources[str(native_plan)]=digest(native_plan);sources[str(Path(__file__))]=digest(__file__)
    output=dict(profiles=profiles,reference_probes=probe,goal_marker_probes=at_goal,
        current_native_function_ast_checks=ast_checks,
        pursuit_signature=str(inspect.signature(pursuit)),
        prior_preview=dict(actual_lower_ticks=previous['actual_lower_ticks'],
            allocated_lower_slots=previous['allocated_lower_slots'],execution_s=previous['execution_s'],
            governor_control_episodes=previous['governor_control_episodes']),
        interface_findings=dict(exact_zero_reference_in_declared_profiles=any(x['exact_zero_speed_reference_permitted'] for x in profiles),
            existing_projection_semantics='Geometric monotone projection based on measured XY; no separately integrated virtual path-time state in the inspected PathState.',
            stopping_or_static_point_stability='NOT_ESTABLISHED by this interface audit; no physical claim or stability proof.',
            finite_preview_safety='No terminal invariant set or certified backup in inspected path_reference_governor; previous proposal fallback explicitly uncertified.',
            implementation_reuse='Existing five-proposal predictive comparator already implemented; do not recreate or call it a new method.',
            user_goal='First entry into original target disk; stopping is not a task requirement. Do not add a stop test to make a governor relevant.'),
        source_sha256=sources,new_physics_steps=0,optimizer_updates=0,
        limits='45 pure correction function probes and9 synthetic goal-marker probes, not45 physical rollouts. Positive speed-command floor is not a lower bound on actual speed or proof of falling at rest. Generic predictive reference governance is not ruled out by a theorem-specific point-stability mismatch. No novelty, safety certificate, or runtime improvement claim.')
    args.output.mkdir(exist_ok=False)
    (args.output/'review.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(profiles=profiles,correction_probes=len(probe),goal_marker_probes=len(at_goal),prior_preview=output['prior_preview'],new_physics_steps=0),indent=2))


if __name__=='__main__':main()
