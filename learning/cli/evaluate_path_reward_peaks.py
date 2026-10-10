#!/usr/bin/env python3
"""Bounded saved-checkpoint evaluation after the existing path run completes."""
import argparse,json,os,sys,time,hashlib,shutil,pickle,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def peak_neighbors(rows,saved):
    if not rows or not saved:raise ValueError('missing rewards or saved models')
    if not all(math.isfinite(r['mean_step_reward']) for r in rows):raise ValueError('nonfinite training rewards')
    peak=max(rows,key=lambda row:row['mean_step_reward'])
    model=peak['sampling_model_update']
    if model!=peak['update']-1:raise ValueError('unexpected sampling-model attribution')
    before=[u for u in saved if u<=model];after=[u for u in saved if u>=model]
    return peak,sorted(set(([max(before)] if before else [])+([min(after)] if after else [])))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def record_handoff(message):
    import fcntl,re
    path=Path('/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md')
    with open(path.parent/'.write.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);text=path.read_text()
        text=re.sub(r'state_revision: (\d+)',lambda m:'state_revision: '+str(int(m[1])+1),text,count=1)
        temporary=path.with_suffix('.tmp');temporary.write_text(text+'\n- STTW reward-peak supplement: '+message+'\n');temporary.replace(path)

def sources(run):
    result=[];current=Path(run).resolve()
    while current not in result:
        result.append(current);parent=json.loads((current/'manifest.json').read_text()).get('resume_parent')
        if not parent:break
        current=Path(parent).resolve()
    return result

def select(run,alpha):
    chain=sources(run);rows={};saved={}
    for root in reversed(chain):
        endpoint=root/f'alpha{alpha}';file=endpoint/'metrics.jsonl'
        if file.exists():
            for line in file.read_text().splitlines():
                row=json.loads(line)
                if row.get('terminal_contract')=='terminal_contract_v2':rows[row['update']]=dict(row,metrics_source=str(file))
        for checkpoint in (endpoint/'checkpoints').glob('update_*.pt'):
            update=int(checkpoint.stem.split('_')[-1]);actor=checkpoint.with_name(f'actor_{update:04d}.pkl')
            if actor.exists():saved[update]=(checkpoint,actor,root)
    peak,neighbors=peak_neighbors(list(rows.values()),list(saved))
    return peak,{u:saved[u] for u in neighbors}

def execute(run,out):
    # No GPU imports or allocation until the parent has finished.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import numpy as np
    import jax
    import torch
    from sttw_control.runtime import configure_compilation_cache
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_selection import cases,summarize_task,persist_task_best
    from sttw_control.path_command_reporting import report_final_panel
    configure_compilation_cache();torch.set_num_threads(2)
    manifest=json.loads((run/'manifest.json').read_text());cfg=manifest['config'];source=manifest['source']
    atomic(out/'manifest.json',dict(manifest,supplement_parent=str(run),selection='raw reward peak sampling-model brackets',training_updates=0))
    proposed={}
    for alpha in [0,1]:
        peak,chosen=select(run,alpha)
        proposed[f'alpha{alpha}']=dict(peak=peak,candidates=[dict(update=u,checkpoint=str(p),checkpoint_sha256=digest(p),actor=str(a),actor_sha256=digest(a)) for u,(p,a,_) in chosen.items()])
    atomic(out/'candidates_before_evaluation.json',proposed)
    env,snapshot=load_environment(cfg,source,'terminal_contract_v2');spent=0
    def status(**kw):atomic(out/'status.json',{'state':'running','stage':'evaluation','new_lower_ticks':spent,'maximum_new_lower_ticks':96000,**kw})
    def account(ticks):
        nonlocal spent
        spent+=ticks
        if spent>96000:raise RuntimeError('supplement evaluation budget exceeded')
    evaluator=BatchedPanelEvaluator(env,snapshot,cfg,status=status,account=account)
    # Copy existing baseline evidence only; no baseline simulation.
    from sttw_control.path_command_cache import reuse_zero_upper
    from sttw_control.geometric_path import build_path
    paths={route:build_path(route,np.asarray(env.physics.helpers.pose(snapshot.data)),cfg) for route in cfg['stage_C_evaluation']['route_ids']}
    if len(reuse_zero_upper(run,out,cfg,source,paths))!=6:raise ValueError('six matching baseline cases required')
    selection={}
    for alpha in [0,1]:
        endpoint=out/f'alpha{alpha}';endpoint.mkdir()
        shutil.copyfile(run/f'alpha{alpha}/last_completed.json',endpoint/'last_completed.json')
        incumbent=json.loads((run/f'alpha{alpha}/best_task_model.json').read_text())
        identity={k:incumbent[k] for k in ['schema','alpha','lower_actor_sha256','config_sha256','sampler_sha256','best_sha256','seed']}
        for key in ['actor','checkpoint']:
            if digest(incumbent[key])!=incumbent[key+'_sha256']:raise ValueError('incumbent SHA changed')
        old_source=json.loads((run/f'alpha{alpha}/final_best/source.json').read_text())
        if old_source['actor_sha256']!=incumbent['actor_sha256']:raise ValueError('incumbent panel identity mismatch')
        persist_task_best(endpoint,incumbent['checkpoint'],incumbent['actor'],incumbent['metrics'],dict(identity,source_update=incumbent['source_update']))
        panel_by_sha={incumbent['actor_sha256']:run/f'alpha{alpha}/final_best'}
        peak,candidates=select(run,alpha);records=[]
        for update,(checkpoint,actor,origin) in candidates.items():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['identity']!=identity or saved['update']!=update:raise ValueError('candidate checkpoint identity mismatch')
            with actor.open('rb') as file:actor_params=pickle.load(file)
            for i,layer in enumerate([0,2,4,6]):
                np.testing.assert_array_equal(actor_params['params'][f'Dense_{i}']['kernel'],saved['policy'][f'actor.{layer}.weight'].numpy().T)
                np.testing.assert_array_equal(actor_params['params'][f'Dense_{i}']['bias'],saved['policy'][f'actor.{layer}.bias'].numpy())
            candidate=dict(identity,source_update=update,actor=str(actor),actor_sha256=digest(actor),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
            panel=endpoint/'validation'/f'update_{update:04d}'
            existing=origin/f'alpha{alpha}/validation'/f'update_{update:04d}'
            complete=all((existing/f'{name}.npz').exists() and (existing/f'{name}_path.npz').exists() for name,_,_ in cases(cfg))
            if complete:
                shutil.copytree(existing,panel)
                traces={name:dict(np.load(panel/f'{name}.npz')) for name,_,_ in cases(cfg)}
                for name,route,_ in cases(cfg):
                    with np.load(panel/f'{name}_path.npz') as archive:
                        for key in ['s','xy','heading','curvature','goal','turn_end']:
                            np.testing.assert_allclose(archive[key],np.asarray(getattr(paths[route],key)),rtol=0,atol=2e-6)
            else:
                params=jax.tree.map(jax.numpy.asarray,actor_params)
                traces=evaluator.evaluate(params,alpha,panel)
            summary=summarize_task(traces,alpha,cfg)
            atomic(panel/'task_metrics.json',summary);atomic(panel/'source.json',candidate)
            persist_task_best(endpoint,checkpoint,actor,summary,dict(identity,source_update=update))
            panel_by_sha[candidate['actor_sha256']]=panel
            records.append(dict(candidate,physical_panel_reused=complete,metrics=summary))
        best=json.loads((endpoint/'best_task_model.json').read_text())
        for key in ['actor','checkpoint']:
            if digest(best[key])!=best[key+'_sha256']:raise ValueError('selected best SHA changed')
        shutil.copytree(panel_by_sha[best['actor_sha256']],endpoint/'final_best')
        atomic(endpoint/'final_best/source.json',best)
        selection[f'alpha{alpha}']=dict(peak_log_update=peak['update'],peak_sampling_model_update=peak['sampling_model_update'],peak_reward=peak['mean_step_reward'],metrics_source=peak['metrics_source'],candidates=records,previous_best=incumbent,new_best=best)
        atomic(out/'selection.json',selection)
        # Candidate coverage: fixed-world XY and same-alpha reward, including losers.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sttw_control.path_command_reporting import reward_series
        figures=endpoint/'candidate_plots';figures.mkdir()
        for name,_,_ in cases(cfg):
            fig,axes=plt.subplots(2,2,figsize=(12,9));xy,reward_ax,cumulative,error=axes.ravel()
            path=dict(np.load(out/'zero_upper'/f'{name}_path.npz'));xy.plot(*path['xy'].T,'k--',label='fixed path')
            panels={'B0':out/'zero_upper',f'previous best {incumbent["source_update"]}':run/f'alpha{alpha}/final_best'}
            panels.update({f'candidate {row["source_update"]}':endpoint/'validation'/f'update_{row["source_update"]:04d}' for row in records})
            for label,folder in panels.items():
                data=dict(np.load(folder/f'{name}.npz'));steps=np.arange(1,len(data['time'])+1);reward,_=reward_series(data,alpha,cfg)
                xy.plot(*data['actual_xy'].T,label=label);xy.scatter(*data['actual_xy'][-1],s=12)
                reward_ax.plot(steps,reward,label=label);cumulative.plot(steps,np.cumsum(reward),label=label)
                error.plot(data['time'],data['actual_forward_speed']-data['v_user'],label=label)
            xy.set(xlabel='world x (m)',ylabel='world y (m)');xy.set_aspect('equal',adjustable='datalim')
            reward_ax.set(xlabel='actual lower steps',ylabel='step reward');cumulative.set(xlabel='actual lower steps',ylabel='cumulative reward');error.set(xlabel='time (s)',ylabel='speed error (m/s)')
            for ax in axes.ravel():ax.legend(fontsize=7);ax.grid(alpha=.2)
            fig.suptitle(f'{name} | alpha{alpha} | training seed {identity["seed"]+alpha}; fixed prepared bank | terminal endpoints retained | no disturbance')
            fig.tight_layout();fig.savefig(figures/f'{name}.png',dpi=120);plt.close(fig)
    report_final_panel(out,cfg)
    atomic(out/'status.json',dict(state='completed',stage='reported',new_lower_ticks=spent,maximum_new_lower_ticks=96000,training_updates=0,original_best_pointers_unchanged=True))
    winners={key:dict(update=value['new_best']['source_update'],qualified_task=value['new_best']['qualified_task']) for key,value in selection.items()}
    record_handoff(f'completed; output={out}; new_lower_ticks={spent}; training_updates=0; selected={winners}; same fixed DEV with expanded reward-guided candidates, not independent holdout; original pointers preserved. See selection.json and report/INDEX.md.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args();run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(exist_ok=False)
    started=time.time();deadline=started+12*3600
    atomic(out/'plan.json',dict(parent=str(run),parent_pid=args.wait_pid,execution_source_sha256=digest(__file__),maximum_candidates_per_alpha=2,scenes_per_candidate=6,maximum_new_lower_ticks=96000,training_updates=0,selection='global raw reward maximum per alpha, attributed sampling_model_update; nearest saved checkpoint on either side',reuse_baseline=True,original_best_pointers_unchanged=True,deadline_epoch=deadline))
    try:
        while True:
            status=json.loads((run/'status.json').read_text())
            if status['state']=='completed':break
            if status['state']=='error' or not Path(f'/proc/{args.wait_pid}').exists():raise RuntimeError('parent did not complete; no competing evaluation launched')
            if time.time()>deadline:raise TimeoutError('parent wait exceeded bounded 12-hour window')
            atomic(out/'status.json',dict(state='waiting',stage='parent_training_and_report',parent_updates=status.get('endpoint_updates'),parent_stage=status.get('stage'),new_lower_ticks=0,training_updates=0,last_check_epoch=time.time()))
            time.sleep(30)
        execute(run,out)
    except BaseException as error:
        atomic(out/'status.json',dict(state='error',error_type=type(error).__name__,message=str(error)))
        record_handoff(f'error; output={out}; {type(error).__name__}: {error}; no automatic retry or training extension.')
        raise

if __name__=='__main__':main()
