"""Explicit finite endpoint-distillation -> engineering -> curriculum pipeline."""
from pathlib import Path
import argparse,copy,gc,json,os,pickle,subprocess,time,traceback
import numpy as np
import jax
import jax.numpy as j
from .preference_best import atomic_json
from .path_command_training import load_environment,_start_tensorboard,_verify_reward_http
from .path_command_distillation import load_teachers,collect,fit_student,inheritance_gate
from .path_command_curriculum import train_course,draw_panel
from .path_command_policy import export_torch_actor


def declaration(cfg):
    c=cfg['training_future_phase_C'];d=cfg['distillation'];updates=cfg['curriculum']['updates'];size=c['num_envs']*c['rollout_steps'];policy=size*updates
    ticks=round(c['episode_s']/cfg['timing']['lower_dt_s']);frequency=cfg['curriculum']['evaluation_every'];panels=1+(updates+frequency-1)//frequency
    eng_n,eng_steps=cfg['engineering_dimensions']
    budget=dict(teacher=d['teacher_lanes']*d['teacher_intervals']*4,correction=d['correction_lanes']*d['correction_intervals']*4,distill_gate=2*12*ticks,warmup=cfg['critic_warmup_batches']*size*4,engineering=eng_n*eng_steps*4,training=policy*4,dev=panels*30*ticks+len(cfg['curriculum']['stages'])*6*ticks)
    return dict(ppo_updates=updates,ppo_policy_steps=policy,lower_tick_caps=budget,total_lower_tick_cap=sum(budget.values()),bc_updates_max=d['initial_updates']+d['correction_updates'],teacher_actor_dim=351,lower_actor_dim=210,automatic_extension=False)


def validate(cfg):
    from .path_command_reward import reward_version
    if cfg['curriculum']['updates']<=0 or reward_version(cfg)!='path_persistent_preference_v3':raise ValueError('positive finite v3 update declaration required')
    if not 0<cfg['training_future_phase_C']['episode_s']<=10:raise ValueError('new episode horizon must be within 10 seconds')
    if cfg['network']['architecture']!='alpha_modulated_v1' or cfg['network']['actor_dim']!=351:raise ValueError('conditional351 required')
    if cfg['alpha_conditioning']['endpoint_probability_each']!=.4 or cfg['alpha_conditioning']['uniform_probability']!=.2:raise ValueError('fixed alpha mixture required')
    if cfg['curriculum']['previous_fraction']!=.25 or not cfg['curriculum']['stages']:raise ValueError('declared stages required')
    if cfg['path_reward']['failure_cost_rate_bound']!=8980.2 or cfg['path_reward']['roll_weight']!=80 or cfg['path_reward']['working_roll_weight']!=240:raise ValueError('shared roll and tail contract mismatch')


def sync_ledger(root,progress,event):
    import fcntl,re
    hub=Path('/home/qy/STTW_CONTROL/research-hub');p=hub/'PROJECT_STATE.md'
    with (hub/'.write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);(hub/'AGENTS.md').read_text();text=p.read_text()
        text=re.sub(r'state_revision: (\d+)',lambda m:'state_revision: '+str(int(m[1])+1),text,count=1)
        row='| STTW_CONTROL | Frozen local300; conditional speed/path | Distilled v3 curriculum400: '+str(progress['state'])+' / '+str(progress['stage'])+' | '+str(progress.get('completed_updates',0))+'/400 PPO slots; phase budgets/qualification separate | '+str(root/'status.json')+' |'
        text='\n'.join(row if line.startswith('| STTW_CONTROL |') else line for line in text.splitlines())+'\n'
        text+='\n### STTW distilled curriculum400 '+event+'\n\nsource_type=repo_artifact; '+json.dumps(dict(run=str(root),state=progress['state'],stage=progress['stage'],completed_updates=progress.get('completed_updates',0),lower_ticks=progress['lower_ticks']),ensure_ascii=False)+'\n'
        tmp=p.with_name('.PROJECT_STATE.curriculum.tmp');tmp.write_text(text);tmp.replace(p)


def execute(config_path,root):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from .path_command_evaluation import BatchedPanelEvaluator,evaluate_conditional
    root=Path(root).resolve();root.mkdir(exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('new campaign required; no implicit restart')
    cfg=json.loads(Path(config_path).read_text());validate(cfg);decl=declaration(cfg)
    from .runtime import configure_compilation_cache
    configure_compilation_cache()
    source=json.loads(Path(cfg['integration_manifest']).read_text());teacher_root=Path(cfg['teacher_run']);teacher_manifest=json.loads((teacher_root/'manifest.json').read_text())
    records,teachers=load_teachers(teacher_root)
    if any(x['lower_actor_sha256']!=source['candidate']['actor_sha256'] for x in records):raise ValueError('teacher/student frozen lower mismatch')
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=dict(config=cfg,teachers=records,source=source,budget=decl,code_revision=code,terminal_contract='terminal_contract_v2',user_authorized_ppo_updates=cfg['curriculum']['updates'])
    atomic_json(root/'manifest.json',manifest);(root/'pipeline').mkdir(exist_ok=True);atomic_json(root/'pipeline/manifest.json',manifest)
    progress=dict(state='initializing',stage='loading',pid=os.getpid(),declared_updates=cfg['curriculum']['updates'],completed_updates=0,lower_ticks={k:0 for k in decl['lower_tick_caps']},started_epoch=time.time())
    def status(**kw):
        progress.update(kw);progress['last_update_epoch']=time.time();atomic_json(root/'status.json',progress)
    def account(kind,count):
        value=progress['lower_ticks'][kind]+int(count)
        if value>decl['lower_tick_caps'][kind]:raise RuntimeError('declared '+kind+' physics cap exceeded')
        progress['lower_ticks'][kind]=value;status()
    def notice(title,body):
        from .direct_command_training import notify
        notify(title,body)
        with (root/'notices.jsonl').open('a') as f:f.write(json.dumps(dict(title=title,body=body,time=time.time()))+'\n')
    tb=_start_tensorboard(root);atomic_json(root/'declaration.json',decl);status(state='running',tensorboard_url=tb['url'])
    sync_ledger(root,progress,'launched')
    notice('STTW distillation started','Two frozen u200 endpoint teachers -> one alpha-conditioned Actor; PPO budget 400 updates after inheritance gate.')
    writer=SummaryWriter(str(root/'tensorboard/distillation'))
    try:
        tc=copy.deepcopy(teacher_manifest['config']);tc['evaluation_execution']='alpha_case_vmap_v1';tc['training_future_phase_C']['episode_s']=cfg['training_future_phase_C']['episode_s']
        sc=copy.deepcopy(tc);sc['network']['architecture']='alpha_modulated_v1'
        env,snapshot=load_environment(tc,source,'terminal_contract_v2')
        data=collect(env,snapshot,tc,teachers,None,root/'teacher_data',cfg['distillation']['teacher_lanes'],cfg['distillation']['teacher_intervals'],cfg['run_seed']+3,status,writer,lambda n:account('teacher',n))
        if not _verify_reward_http(root,tb,'distillation'):raise RuntimeError('distillation rollout reward unavailable over HTTP')
        actor=fit_student(data,records[0]['checkpoint'],sc,cfg['distillation']['initial_updates'],cfg['run_seed'],root/'distillation',writer,status)
        def gate(actor,index):
            params=export_torch_actor(actor,sc);out=root/f'inheritance_gate_{index}'
            evaluator=BatchedPanelEvaluator(env,snapshot,sc,status=lambda **kw:status(stage='distillation_validation',**{k:v for k,v in kw.items() if k not in ('stage','state')}),account=lambda count:account('distill_gate',count))
            panels=evaluate_conditional(evaluator,params,[0.,1.],out,diagnostic_only=True)
            result=inheritance_gate(panels,teacher_root,tc);atomic_json(out/'metrics.json',result);draw_panel(out,panels,sc)
            atomic_json(root/'distillation_gate.json',result);return result
        passed=gate(actor,1)['passed']
        if not passed:
            notice('STTW distillation correction','First inheritance gate did not pass. Running the single declared student-state correction pass; PPO has not started.')
            correction=collect(env,snapshot,tc,teachers,export_torch_actor(actor,sc),root/'correction_data',cfg['distillation']['correction_lanes'],cfg['distillation']['correction_intervals'],cfg['run_seed']+4,status,writer,lambda n:account('correction',n))
            data={k:np.concatenate([data[k],correction[k]]) for k in data}
            actor=fit_student(data,records[0]['checkpoint'],sc,cfg['distillation']['correction_updates'],cfg['run_seed']+1,root/'distillation_corrected',writer,status,actor)
            passed=gate(actor,2)['passed']
        writer.close()
        if not passed:
            status(state='needs_attention',stage='distillation_gate_failed',message='Declared two attempts exhausted; student inheritance failed. No PPO launched.')
            sync_ledger(root,progress,'inheritance_gate_failed')
            notice('STTW distillation needs attention','Inheritance gate did not pass after the declared correction. No curriculum PPO was launched.');return
        notice('STTW distillation completed','Endpoint inheritance passed. Starting discarded engineering smoke, fresh Critic warmup, then the declared curriculum PPO.')
        del data,teachers,env,snapshot;gc.collect();jax.clear_caches()
        train_course(cfg,source,actor,root/'engineering',status,account,notice,tb,smoke=True)
        gc.collect();jax.clear_caches()
        sync_ledger(root,progress,'engineering_passed_entering_PPO')
        train_course(cfg,source,actor,root/'training',status,account,notice,tb)
        status(state='completed',stage='reported',completed_updates=cfg['curriculum']['updates'])
        (root/'INDEX.md').write_text('# Distilled alpha curriculum\n\n[Training results](training/INDEX.md) · [Inheritance gate](distillation_gate.json) · [Progress](status.json)\n')
        sync_ledger(root,progress,'completed')
        notice('STTW pipeline completed','Distillation, inheritance evaluation, 400-update PPO and report indexes completed. Qualification is reported separately. '+str(root/'INDEX.md'))
    except BaseException as exc:
        writer.close();status(state='error',message=str(exc),error_type=type(exc).__name__)
        sync_ledger(root,progress,'error')
        notice('STTW experiment error',str(exc));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute(a.config,a.output)
    else:
        cfg=json.loads(Path(a.config).read_text());validate(cfg);print(json.dumps(declaration(cfg),indent=2))

if __name__=='__main__':main()
