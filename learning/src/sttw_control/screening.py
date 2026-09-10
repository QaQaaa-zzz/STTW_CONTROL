"""Baseline-only operating-condition screening; all outputs are development data."""
from dataclasses import asdict,replace
import itertools,json
from pathlib import Path
import numpy as np
from .env import RecoveryEnv,load_config
from .evaluation import evaluate
from .disturbance import recovery_metrics
from .media import state_series
from .path import eight_trace_features


def scenarios(task,protocol):
    """Keep origin/tangent initialization consistent for either turn direction."""
    if task.figure_eight is not None:
        for speed in protocol['speeds']:
            if speed<=0:raise ValueError('positive screening speed required')
            yield f'v{speed:g}_eight',replace(task,speed_reference=speed,random_events=None,action_mapping=None,horizon_seconds=protocol['horizon_seconds'],disturbance_start=protocol['event_start_seconds'],disturbance_force=0.,disturbance_steer_rate=0.)
        return
    if task.circle is None:raise ValueError('screening requires a reference path')
    for speed,radius,direction in itertools.product(protocol['speeds'],protocol['radii'],protocol['directions']):
        if speed<=0:raise ValueError('positive screening speed required')
        circle=replace(task.circle,radius=radius,center_x=0.,center_y=direction*radius,direction=direction)
        c=replace(task,circle=circle,speed_reference=speed,random_events=None,action_mapping=None,
                  horizon_seconds=protocol['horizon_seconds'],disturbance_start=protocol['event_start_seconds'],
                  disturbance_force=0.,disturbance_steer_rate=0.)
        yield f'v{speed:g}_r{radius:g}_d{direction}',c


def run(task_path,protocol_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p=json.loads(Path(protocol_path).read_text());task=load_config(task_path)
    conditions=list(scenarios(task,p))
    if len({name for name,_ in conditions})!=len(conditions):raise ValueError('duplicate operating conditions')
    episodes=len(conditions)*(1+len(p['events']))*len(p['seeds'])
    declaration={'protocol':p,'task':asdict(task),'role':'development_baseline_screening_not_independent_test',
                 'maximum_episodes':episodes,'maximum_control_steps':episodes*int(np.ceil(p['horizon_seconds']/task.controller.dt))}
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    completed=0
    def status(phase,**extra):
        (output/'status.json').write_text(json.dumps({'phase':phase,'completed_episodes':completed,**extra},indent=2)+'\n')
    try:
        for name,c in conditions:
            for seed in p['seeds']:
                nominal=None;nominal_ok=False
                cases=[{'name':'nominal'},*p['events']]
                for case in cases:
                    cfg=replace(c,disturbance_force=case.get('force',0.),disturbance_steer_rate=case.get('steer_rate',0.),
                                disturbance_duration=case.get('duration',.6),disturbance_waveform=case.get('waveform','half_sine'))
                    dest=output/name/f'seed_{seed}'/case['name'];status('running',condition=name,case=case['name'])
                    summary=evaluate(RecoveryEnv(cfg),dest,seed=seed)
                    trace=dict(np.load(dest/'trace.npz'));completed+=1
                    if case['name']=='nominal':
                        nominal=trace
                        tail=trace['time']>=p['event_start_seconds']
                        speed=state_series(trace,asdict(cfg))['speed']
                        radial=(eight_trace_features(trace['pose'],cfg.figure_eight)[:,0] if cfg.figure_eight is not None else np.linalg.norm(trace['qpos'][:,:2]-[cfg.circle.center_x,cfg.circle.center_y],axis=1)-cfg.circle.radius)
                        nominal_ok=bool(not summary['physical_failure'] and np.any(tail) and np.sqrt(np.mean(radial[tail]**2))<=p['nominal_radial_rmse_max'] and np.sqrt(np.mean((speed[tail]-cfg.speed_reference)**2))<=p['nominal_speed_rmse_max'])
                        recovery={}
                    else:recovery=recovery_metrics(trace,nominal,asdict(cfg))
                    row={'condition':name,'seed':seed,'case':case['name'],'nominal_eligible':nominal_ok,'summary':summary,'recovery':recovery}
                    with (output/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        status('complete')
    except Exception as exc:
        status('error',error=str(exc));raise
