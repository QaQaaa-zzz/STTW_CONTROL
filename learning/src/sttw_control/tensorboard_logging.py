"""TensorBoard projection of authoritative PPO JSON records; no model execution."""
from pathlib import Path
import hashlib
import json
import math
import numbers
import time


def validation_labels(declaration):
    """Match make_command_validator's case -> alpha -> seed flattening order."""
    task=declaration.get('task',{});cfg=declaration.get('training',{})
    if task.get('tracking') is not None:
        cases=cfg.get('validation_events') or [None]
        return [f'alpha_{a:g}/seed_{seed}/case_{i:02d}'
                for a in task['priority']['validation_alphas']
                for seed in cfg.get('validation_seeds',[]) for i in range(len(cases))]
    cases=cfg.get('command_validation_schedules')
    if task.get('motion_commands') is None or not cases:return []
    return [f'case_{i:02d}/alpha_{a:g}/seed_{s}' for i in range(len(cases))
            for a in task.get('priority',{}).get('validation_alphas',[])
            for s in cfg.get('validation_seeds',[])]


def scalar_values(record,labels=(),profile="full"):
    result={}
    def add(tag,value):
        if isinstance(value,numbers.Real) and math.isfinite(value):result[tag]=float(value)
    for key in ('mean_step_reward','learning_rate','control_transitions','stage_control_transitions','episode_ends'):
        add('train/'+key,record.get(key))
    for key,value in record.get('reward_components_mean_step',{}).items():add('reward_components/'+key,value)
    for key,value in zip(('policy','value','latent_entropy','approx_kl'),record.get('loss_metrics',[])):
        add(('kl/' if key=='approx_kl' else 'loss/')+key,value)
    audit=record.get('optimizer_audit',{})
    for key,value in audit.items():add('optimizer/'+key,value)
    if 'accepted_minibatches' in audit:
        add('optimizer/retained_minibatches',0 if audit.get('full_update_rolled_back') else audit['accepted_minibatches'])
    for key,value in record.items():
        if key.endswith('_seconds') or key.endswith('_per_second'):add('timing/'+key,value)
    for key,value in (record.get('device_memory_stats') or {}).items():add('memory/'+key,value)
    for group in ('validation','paired_episode_return'):
        for key,value in record.get(group,{}).items():
            if isinstance(value,list):
                for i,item in enumerate(value):
                    label=labels[i] if i<len(labels) else f'sample_{i:03d}'
                    add(f'{group}/{label}/{key}',item)
            else:add(f'{group}/{key}',value)
    for key,value in record.get('selection',{}).items():
        if key!='rank':add('selection/'+key,value)
    if profile == 'core':
        keep = {'train/learning_rate', 'train/mean_step_reward', 'loss/policy', 'loss/value', 'kl/approx_kl',
                'optimizer/retained_minibatches'}
        result = {k:v for k,v in result.items() if k in keep}
        for group in ('validation',):
            values = record.get(group, {}).get('episode_return', [])
            if isinstance(values, list):
                values = [v for v in values if isinstance(v, numbers.Real) and math.isfinite(v)]
                if values: result[group+'/mean_episode_return'] = sum(values)/len(values)
        for key in ('baseline','candidate','delta'):
            values = record.get('paired_episode_return', {}).get(key, [])
            values = [v for v in values if isinstance(v,numbers.Real) and math.isfinite(v)]
            if values:result['validation/mean_'+key+'_return'] = sum(values)/len(values)
        failures=record.get('validation',{}).get('failed',[])
        if failures:result['validation/failure_fraction']=sum(failures)/len(failures)
        for key in ('speed','yaw','alive','speed_tolerance','yaw_tolerance','yaw_tracking','failure','speed_tracking','underspeed_tracking','overspeed_tracking','path_tracking','speed_tail','path_tail','speed_budget','underspeed_budget','overspeed_budget','path_budget','return_time','recovery','attitude','roll_rate','action_delta'):

            add('reward_components/'+key, record.get('reward_components_mean_step',{}).get(key))
        for key in ('terminal_tracking_hold','return_deadline_missed','recovered_after_excursion'):
            values=record.get('validation',{}).get(key,[])
            if values:add('validation/'+key+'_fraction',sum(values)/len(values))
    elif profile != 'full':
        raise ValueError('Unknown TensorBoard profile')
    # These are kept even in the compact profile; missing values are not zeros.
    for key in ('deadline','return_overdue','action'):
        add('reward_components/'+key,record.get('reward_components_mean_step',{}).get(key))
    for key in ('candidate_exact_kl','candidate_kl_p95','final_exact_kl','kl_limit',
                'full_update_rolled_back','mean_action_abs','mean_action_saturation_fraction',
                'steer_latent_std','rear_latent_std','attempted_minibatches'):
        add('optimizer/'+key,record.get('optimizer_audit',{}).get(key))
    for key,value in record.get('sample_phase',{}).items():add('sample_phase/'+key,value)
    for key in ('control_transitions','episode_ends'):add('train/'+key,record.get(key))
    add('timing/rollout_control_steps_per_second',record.get('rollout_control_steps_per_second'))
    for key in ('sampling_policy_update',):add('train/'+key,record.get(key))
    add('selection/training_sample_best_update',record.get('best_reward_model',{}).get('update'))
    add('selection/development_gates_passed',record.get('selection',{}).get('development_gates_passed'))
    for i,group in enumerate(record.get('alpha_training_samples',[])):
        for key,value in group.items():
            if key not in ('alpha_lower','alpha_upper','upper_inclusive'):add(f'alpha_training/bin_{i}/{key}',value)
    return result


def register_run(logdir):
    """Index repo-owned events without recursively scanning models/worktrees."""
    logdir=Path(logdir).resolve()
    for ancestor in logdir.parents:
        if ancestor.name=='runs' and (ancestor.parent/'PROJECT.md').is_file():
            index=ancestor/'tensorboard'
            if index==logdir or index in logdir.parents:return
            index.mkdir(exist_ok=True)
            relative=logdir.parent.relative_to(ancestor).as_posix()
            name=relative.replace('/','__')+'_'+hashlib.sha256(relative.encode()).hexdigest()[:8]
            link=index/name
            if link.is_symlink() and link.resolve()==logdir:return
            link.symlink_to(logdir,target_is_directory=True)
            return


class TrainingEvents:
    """Lazy writer: leave output creation to the trainer; flush each iteration."""
    def __init__(self,path,profile="full",register=True):
        self.path=Path(path);self.writer=None;self.last_step=None
        self.profile=profile;self.register=register
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def close(self):
        if self.writer is not None:
            self.writer.close();self.writer=None
    def write(self,record,labels=()):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        step=int(record['update'])
        if self.last_step is not None and step<=self.last_step:raise ValueError('updates must be strictly increasing')
        values=scalar_values(record,labels,self.profile)
        if self.writer is None:
            self.writer=EventFileWriter(str(self.path))
            if self.register:register_run(self.path)
        self.writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[
            Summary.Value(tag=k,simple_value=v) for k,v in values.items()])))
        self.writer.flush();self.last_step=step


def import_metrics(paths,output):
    """Replay an explicit lineage once, refusing overlapping updates or overwrite."""
    paths=[Path(p) for p in paths];output=Path(output)
    if output.exists():raise FileExistsError(output)
    rows=[];sources=[]
    for path in paths:
        raw=path.read_bytes()
        declaration=path.parent/'declaration.json'
        labels=validation_labels(json.loads(declaration.read_text())) if declaration.exists() else []
        records=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        rows.extend((r,labels) for r in records)
        sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'records':len(records)})
    steps=[r['update'] for r,_ in rows]
    if not steps or any(b<=a for a,b in zip(steps,steps[1:])):raise ValueError('updates must be nonempty and strictly increasing')
    with TrainingEvents(output) as writer:
        for record,labels in rows:writer.write(record,labels)
    (output/'sources.json').write_text(json.dumps({'sources':sources,'step':'global PPO iteration','scope':'Historical replay; event wall time is import time, not training time'},indent=2)+'\n')
    return len(rows)
