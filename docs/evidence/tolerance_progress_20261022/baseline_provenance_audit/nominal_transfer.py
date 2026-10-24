"""Single declared24-trajectory nominal transfer, with automatic offline reports."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest
from prepare_predictive_confirmation import register


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    source=Path('runs/formula_prospective_confirmation_v1/declaration.json').resolve();decl=json.loads(source.read_text())
    model=json.loads(Path('runs/reserve_feature_expansion_v1/post_fixed_final_v1/models.json').read_text())['records']['H75']
    from sttw_control.feature_comparison_evaluation import authenticated_parent_record
    authenticated_parent_record(model)
    cfg=decl['config'];assert cfg['training_future_phase_C']['episode_s']==10.
    native=copy.deepcopy(decl);native.update(experiment_id='nominal_preference_transfer_v1',question='Frozen nominal preference utility transfer; no robustness-method claim',policies={'H75':model},speeds=[2.35,2.75],alphas=[0.,.25,.5,.75,1.],forces_N=[0.],onsets_s=[3.2],duration_s=.1,maximum_lower_ticks=40000,training_steps=0,diagnostic_physical_stop='contact_only_v1',data_role='Nominal transfer DEV; geometry previously used for disturbance study; no final TEST')
    native['source_sha256']=json.loads((source.parent/'plan.json').read_text())['source_sha256']
    declaration=root/'declaration.json';atomic_json(declaration,native)
    h75=root/'H75';h75.mkdir();zero=root/'zero';zero.mkdir()
    # Reuse the previously authenticated constant-zero legacy representation.
    oldzero=Path('runs/zero_middle_nominal_full_v1/plan.json').resolve();oldplan=json.loads(oldzero.read_text());actor=Path(oldplan['constant_actor'])
    if digest(actor)!=oldplan['constant_actor_sha256'] or oldplan['verified_equivalence_rows']<116:raise ValueError('Zero representation changed')
    cfgzero=copy.deepcopy(cfg);cfgzero['network']['architecture']='legacy'
    from sttw_control.path_command_selection import cases
    mapping=[dict(name=n,builder=b,actual_requested_speed_m_s=v) for n,b,v in cases(cfgzero)]
    assert len(mapping)==4
    scripts=Path(__file__).resolve().parent;design=scripts/'NOMINAL_TRANSFER_DESIGN.md'
    files=[Path(__file__).resolve(),design,declaration,source,oldzero,actor,scripts/'run_nominal_control.py',scripts/'analyze_nominal_control.py',scripts/'audit_preference_utility.py',Path('learning/cli/screen_contact_preference.py')]
    files+=[Path('learning/src/sttw_control')/(n+'.py') for n in ['preference_tolerance_screen','path_command_evaluation','path_command_policy','path_command_training','path_command_env','path_command_reward','controller','closed_loop_kernel','teleop_env','contact_only_environment']]
    sources={str(f.resolve()):digest(f) for f in files}
    zp=dict(project='STTW_CONTROL',purpose='Four full-task nominal follower transfer controls',source_declaration=str(declaration),config=cfgzero,source=native['source'],original_H75_evaluation=str(h75),constant_actor=str(actor),constant_actor_sha256=digest(actor),verified_equivalence_rows=oldplan['verified_equivalence_rows'],cases=mapping,declared_cases=4,maximum_lower_ticks=8000,training_steps=0,route_name_map={r['builder']:r['name'] for r in native['routes']},source_sha256=sources)
    atomic_json(zero/'plan.json',zp)
    atomic_json(root/'plan.json',dict(project='STTW_CONTROL',declared_cases=24,maximum_lower_ticks=48000,training_steps=0,source_sha256=sources,design=str(design),model=model,initialization='same complete prepared state; fixed actors; no optimizer',stop_rule='One nominal transfer only; no retraining/geometry extension/retry'))
    for status,count in [(root/'status.json',24),(h75/'status.json',20),(zero/'status.json',4)]:
        atomic_json(status,dict(phase='prepared',completed_cases=0,declared_cases=count,active_lower_ticks=0,training_steps=0));register(status.parent/'ACTIVE_RUN.json','STTW_CONTROL 名义偏好迁移 '+status.parent.name,status)
    print('prepared24 complete nominal trajectories;maximum48000 lower ticks;0 training')


def execute(root):
    plan=json.loads((root/'plan.json').read_text());s=dict(phase='running',pid=os.getpid(),stage='loading',completed_cases=0,declared_cases=24,active_lower_ticks=0,training_steps=0)
    def update(**kw):s.update(kw,checked_unix=time.time());atomic_json(root/'status.json',s)
    def run(stage,args):
        update(stage=stage)
        with (root/(stage+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT);update(child_pid=p.pid);code=p.wait()
        if code:raise RuntimeError(stage+' exited'+str(code)+'; preserve original error, no restart')
    update()
    try:
        for file,sha in plan['source_sha256'].items():
            if digest(file)!=sha:raise ValueError('Declared source changed '+file)
        scripts=Path(__file__).resolve().parent
        run('H75',['learning/cli/screen_contact_preference.py','--declaration',root/'declaration.json','--output',root/'H75'])
        h=json.loads((root/'H75/status.json').read_text())
        if h['phase']!='completed' or h['completed_cases']!=20:raise ValueError('H75 incomplete')
        update(completed_cases=20,active_lower_ticks=h['active_lower_ticks'])
        run('zero',[scripts/'run_nominal_control.py','--output',root/'zero'])
        z=json.loads((root/'zero/status.json').read_text())
        if z['phase']!='completed' or z['completed_cases']!=4:raise ValueError('Zero incomplete')
        total=h['active_lower_ticks']+z['active_lower_ticks']
        if total>plan['maximum_lower_ticks']:raise ValueError('Declared physics budget exceeded')
        update(completed_cases=24,active_lower_ticks=total)
        # All physical outcomes are complete before either offline comparison.
        run('paired_analysis',[scripts/'analyze_nominal_control.py','--run',root/'zero','--output',root/'analysis'])
        run('utility_analysis',[scripts/'audit_preference_utility.py','--source-analysis',root/'analysis/analysis.json','--zero-run',root/'zero','--output',root/'utility'])
        update(phase='completed',stage='all24_nominal_and120_factual_utility_scores_ready',report=str(root/'analysis/REPORT.md'))
    except BaseException as exc:update(phase='error',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare)(a.run.resolve())
