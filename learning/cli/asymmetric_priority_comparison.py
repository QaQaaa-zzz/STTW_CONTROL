#!/usr/bin/env python3
"""Run declared RSL training arms sequentially; the legacy rho pair is supported."""
import argparse
import json
from pathlib import Path

from sttw_control.training import TrainingConfig, train


def continuation_config(original, target, offset, checkpoint, no_evaluation):
    """Target is a cumulative update index, never an additional budget."""
    if target <= offset:
        raise ValueError('target must exceed the saved update')
    config=dict(original, updates=target-offset, resume_checkpoint=checkpoint)
    if no_evaluation:
        config.update(training_reward_selection=True,evaluation_reward_best=False,
                      best_model_every_update=False,validation_updates=None)
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-task',type=Path)
    parser.add_argument('--control-task',type=Path)
    parser.add_argument('--arm',action='append',help='Repeat NAME=TASK.json for a named multi-arm campaign')
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-run',type=Path,help='Prior campaign; restore each arm from its latest complete RSL snapshot')
    parser.add_argument('--target-updates',type=int,help='Cumulative target, including restored updates')
    parser.add_argument('--no-evaluation',action='store_true',help='Disable all development rollouts, including initialization and final evaluation')
    args=parser.parse_args()
    if args.arm:
        if args.main_task or args.control_task:parser.error('use --arm or legacy main/control tasks')
        arms=[]
        for value in args.arm:
            name,sep,path=value.partition('=')
            if not sep or not name or not all(ch.isalnum() or ch in '_-' for ch in name) or not Path(path).is_file():
                parser.error('invalid arm; expected NAME=existing-task.json')
            arms.append((name,Path(path)))
        if len({name for name,_ in arms})!=len(arms):parser.error('duplicate arm names')
    else:
        if not args.main_task or not args.control_task:parser.error('provide --arm or both --main-task/--control-task')
        arms=(('rho34p2',args.main_task),('rho10',args.control_task))
    if args.resume_run and args.target_updates is None:
        parser.error('--resume-run requires --target-updates')
    if args.target_updates is not None and args.target_updates<=0:
        parser.error('--target-updates must be positive')
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    original=json.loads(args.training.read_text())
    completed=[]
    plans=[]
    def status(phase,**extra):
        target=root/'status.json';temporary=root/'status.tmp'
        temporary.write_text(json.dumps({'phase':phase,'complete':phase=='complete',
            'completed_arms':completed,**extra},indent=2,ensure_ascii=False)+'\n')
        temporary.replace(target)
    try:
        for name,task in arms:
            offset=0;checkpoint=None;arm_config=original
            parent=args.resume_run/name/'training' if args.resume_run else None
            if parent is not None:
                source=json.loads((parent/'declaration.json').read_text())
                arm_config=source['training']
                task=root/(name+'_task.json')
                task.write_text(json.dumps(source['task'],indent=2)+'\n')
                snapshots=list((parent/'checkpoints').glob('update_*/rsl_snapshot.pt'))
                if snapshots:
                    import torch
                    latest=max(snapshots,key=lambda p:int(p.parent.name.split('_')[-1]))
                    saved=torch.load(latest,map_location='cpu',weights_only=False)
                    offset=int(saved['update']);checkpoint=str(latest.parent.resolve())
                    if int(saved['control_transitions'])!=offset*arm_config['num_envs']*arm_config['rollout_steps']:
                        raise ValueError('snapshot transition budget differs from declared fixed batch')
                    del saved
            target=args.target_updates if args.target_updates is not None else original['updates']
            config=TrainingConfig(**continuation_config(arm_config,target,offset,checkpoint,args.no_evaluation))
            plan={'name':name,'restored_update':offset,'target_update':target,
                  'additional_updates':config.updates,'resume_checkpoint':checkpoint,
                  'additional_control_transitions':config.updates*config.num_envs*config.rollout_steps,
                  'cumulative_control_transitions':target*config.num_envs*config.rollout_steps,
                  'development_evaluation':not config.training_reward_selection,
                  'resume_semantics':'policy, critic, optimizer and RNG; fresh prepared physics',
                  'training':str(root/name/'training')}
            plans.append(plan)
            (root/(name+'_training.json')).write_text(json.dumps(config.__dict__,indent=2)+'\n')
            (root/'continuation.json').write_text(json.dumps({'arms':plans,'source_run':str(args.resume_run)},indent=2)+'\n')
            status('training',active_arm=name)
            result=train(task,root/name/'training',config)
            if not result.get('complete'):raise RuntimeError(name+' training did not complete')
            completed.append({'name':name,'training':str(root/name/'training'),
                'control_transitions':result['control_transitions'],
                'last_checkpoint':result['last_checkpoint'],
                'best_reward_checkpoint':result.get('best_reward_checkpoint')})
        status('complete')
    except Exception as exc:
        status('error',error=repr(exc));raise


if __name__=='__main__':main()
