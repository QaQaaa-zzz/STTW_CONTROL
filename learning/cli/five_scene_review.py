"""Replay full ECBC + residual on the historical five-scene panel; no selection.

Run from the repository root with PYTHONPATH=learning/src. Existing completed
traces can be audited and reused by repeating the same arguments.
"""
import os
os.environ.update(JAX_PLATFORMS='cpu', MUJOCO_GL='egl', OMP_NUM_THREADS='1')
import argparse,json,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from sttw_control.env import RecoveryEnv,config_from_dict
from sttw_control.network import make_policy_identity,load_policy
from sttw_control.evaluation import evaluate
from sttw_control.tracking_diagnostics import audit_trace,rescore_trace,generate_panel

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=49001);p.add_argument('--scenes',nargs='+',help='Explicit subset; defaults to all five scenes');a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);(root/'frozen').mkdir(exist_ok=True)
    if not json.loads((a.training/'status.json').read_text()).get('complete'):raise ValueError('training must be complete before this review')
    source=json.loads((a.training/'declaration.json').read_text());cfg=config_from_dict(source['task']);best=None
    if a.checkpoint is None:
        best=json.loads((a.training/'best_model.json').read_text())
    ck=(a.checkpoint or Path(best['checkpoint'])).resolve();alphas=list(cfg.priority.validation_alphas)
    if cfg.actuator.base_output_scale!=1.:raise ValueError('this panel requires full ECBC scale 1; scaled and direct arms need separately labeled baseline configurations')
    if not cfg.tracking or not cfg.tracking.geometric:raise ValueError('geometric tracking configuration required')
    identity=make_policy_identity(RecoveryEnv(cfg).bundle.identity,asdict(cfg),cfg.observation.history_steps);policy=load_policy(ck,expected=identity)
    panel=json.loads(Path('learning/configs/timed_priority_standard_panel.json').read_text());panel.update(seed=a.seed,evaluation_seeds=[a.seed],priority_alphas=alphas,cases=[])
    spec=json.loads(Path('learning/configs/synthetic_turn_reproduction.json').read_text())
    configs={};warmups={}
    base=replace(cfg,random_events=None,disturbance_force=0.,disturbance_steer_rate=0.,disturbance_rear_torque=0.)
    for case in panel['reference_cases']:
        configs[case['name']]=replace(base,speed_reference=case['commands'][0][1],timed_reference=replace(cfg.timed_reference,training_mix=False,fixed_scenario=None,fixed=case['commands']))
    angle=np.deg2rad(spec['turn_angle_degrees']);v=spec['requested_speed_m_s'];peak=spec['peak_curvature_per_m'];t0=spec['precondition_seconds'];dt=cfg.controller.dt
    length=2*angle/peak;times=np.arange(int(np.ceil(length/v/dt))+1)*dt
    s0=np.minimum(v*times,length);s1=np.minimum(v*(times+dt),length)
    def heading(s):return peak*(s/2-length*np.sin(2*np.pi*s/length)/(4*np.pi))
    k=(heading(s1)-heading(s0))/(v*dt)
    commands=[[0.,spec['initial_speed_m_s'],0.]]+[[float(t0+t),v,float(v*c)] for t,c in zip(times,k)]
    configs['synthetic_turn']=replace(base,speed_reference=spec['initial_speed_m_s'],horizon_seconds=t0+spec['observation_seconds_after_command'],timed_reference=replace(cfg.timed_reference,training_mix=False,fixed_scenario=None,fixed=commands,speed_slew=abs(v-spec['initial_speed_m_s'])/dt*1.01,yaw_slew=float(np.max(np.abs(np.diff(v*k))))/dt*1.01))
    warmups['synthetic_turn']=t0
    if a.scenes:
        if len(set(a.scenes))!=len(a.scenes) or not set(a.scenes)<=set(configs):raise ValueError('invalid or duplicate scenario subset')
        configs={name:configs[name] for name in a.scenes}
        warmups={name:value for name,value in warmups.items() if name in configs}
    manifest=dict(checkpoint=str(ck),selection=best if best is not None else 'user-selected explicit checkpoint; not selected by this script',priority_alphas=alphas,seed=a.seed,residual_episodes=len(configs)*len(alphas),baseline_episodes=len(configs),role='historical development panel; not held-out evidence',scenario_notes='Four standard schedules use current slew and committed geometric projection; initial speed matches first request. Synthetic turn uses original spatial sin-squared path with explicit evaluation-only slew override; all methods zero residual for first 3 task seconds. All scenes also retain training closed-loop preparation before task time zero.',warmup_seconds=warmups)
    def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    if (root/'declaration.json').exists() and json.loads((root/'declaration.json').read_text())!=manifest:raise ValueError('existing output declaration differs')
    write(root/'declaration.json',manifest);write(root/'frozen/panel.json',panel);write(root/'frozen/task.json',asdict(cfg));write(root/'frozen/scenarios.json',{n:asdict(c) for n,c in configs.items()})
    start=time.time()
    def status(phase,**kw):write(root/'status.json',dict(phase=phase,complete=phase=='complete',elapsed_seconds=time.time()-start,**kw))
    def run(env,dest,alpha,actor=None):
        if (dest/'trace.npz').exists():
            d=json.loads((dest/'declaration.json').read_text())
            if d['config']!=json.loads(json.dumps(asdict(env.config))) or d['seed']!=a.seed or d.get('priority_override')!=alpha:raise ValueError('cached trace mismatch')
            audit_trace(dest);return
        evaluate(env,dest,seed=a.seed,priority_alpha=alpha,policy=actor,policy_identity=identity if actor else None)
    try:
        for case,c in configs.items():
            env=RecoveryEnv(c)
            for i,alpha in enumerate(alphas):
                dest=root/f'evaluation/alpha_{i}/seed_{a.seed}';dest.mkdir(parents=True,exist_ok=True)
                write(dest/'declaration.json',dict(panel=panel,checkpoint=str(ck),policy=identity,scenarios={n:asdict(c2) for n,c2 in configs.items()},priority_alpha_override=alpha))
                pair=dest/case;status('evaluation',scenario=case,alpha=alpha);print(case,alpha,flush=True)
                if i==0:run(env,pair/'baseline',alpha)
                elif not (pair/'baseline/trace.npz').exists():
                    src=root/f'evaluation/alpha_0/seed_{a.seed}/{case}/baseline';data=rescore_trace(audit_trace(src),alpha);target=pair/'baseline';target.mkdir(parents=True)
                    d=json.loads((src/'declaration.json').read_text());d.update(priority_override=alpha,baseline_reuse=str(src),reuse_scope='identical physical trace; shared reward replay; recorded observations retain source alpha')
                    write(target/'declaration.json',d);np.savez_compressed(target/'trace.npz',**data['trace']);write(target/'status.json',dict(status='complete',baseline_reused=True))
                count=[0];warmup=round(warmups.get(case,0)/dt)
                def actor(obs):
                    count[0]+=1
                    return np.zeros(2) if count[0]<=warmup else policy(obs)
                run(env,pair/'residual',alpha,actor)
        status('plotting');rows=generate_panel(root);status('complete',conditions=len(rows));print('COMPLETE',flush=True)
    except Exception as e:status('error',error=repr(e));raise


if __name__ == "__main__":
    main()
