"""Prepare and execute a fixed24-branch zero-middle comparator, no training."""
import argparse,json,os,pickle,subprocess,sys,time
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from prepare_predictive_confirmation import register


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    old=Path('runs/formula_prospective_confirmation_v1').resolve()
    prior=json.loads((old/'plan.json').read_text())
    if json.loads((old/'campaign_status.json').read_text())['phase']!='completed':raise ValueError('Prior experiment still active')
    for file,sha in prior['source_sha256'].items():
        if digest(file)!=sha:raise ValueError('Original runtime changed '+file)
    model=json.loads((old/'left/prepare_plan.json').read_text())['model']
    from sttw_control.feature_comparison_evaluation import authenticated_parent_record
    authenticated_parent_record(model)
    with Path(model['actor']).open('rb') as f:params=pickle.load(f)
    for key in ['kernel','bias']:params['params']['Dense_3'][key]=np.zeros_like(params['params']['Dense_3'][key])
    actor=root/'zero_middle_actor.pkl'
    with actor.open('wb') as f:pickle.dump(params,f)
    from sttw_control.path_command_policy import make_path_actor
    import jax.numpy as jp
    cfg=json.loads((old/'declaration.json').read_text())['config']
    with np.load('runs/deployment_contract_audit_v1/real_prefix_inputs.npz') as z:real=z['upper_observation']
    random=np.random.default_rng(924403).uniform(-5,5,(100,351)).astype('float32');random[:,336]=np.linspace(0,1,100)
    obs=np.concatenate([real,random]);np.testing.assert_array_equal(np.asarray(make_path_actor(cfg).apply(params,jp.asarray(obs))),0.)
    policy=dict(actor=str(actor),actor_sha256=digest(actor),semantics='constant zero middle; no learned checkpoint',parent_actor=model['actor'],parent_actor_sha256=model['actor_sha256'])
    design=Path(__file__).with_name('NEW_GEOMETRY_ZERO_DESIGN.md').resolve()
    sources=dict(prior['source_sha256']);sources.update({str(x):digest(x) for x in [Path(__file__).resolve(),design,actor]})
    for side in ['left','right']:
        child=root/side;child.mkdir();plan=json.loads((old/side/'prepare_plan.json').read_text())
        plan.update(model=policy,parent_model=model,policy_name='zero_middle',alphas=[0.],declared_prefixes=4,declared_cases=12,
                    maximum_prepare_lower_ticks=4000,maximum_scan_lower_ticks=9600,maximum_total_lower_ticks=13600,
                    source_sha256=sources,design=str(design),stop_rule='24 fixed zero-middle branches; no extension/retry',verified_zero_inputs=len(obs))
        plan.pop('pre_scan_forecast_script',None)
        atomic_json(child/'prepare_plan.json',plan)
        for file,count in [('prepare_status.json',4),('status.json',12),('campaign_status.json',12)]:
            atomic_json(child/file,dict(phase='prepared',completed_cases=0,declared_cases=count,active_lower_ticks=0,training_steps=0))
            register(child/('ACTIVE_'+file.upper()),'STTW_CONTROL 新几何零中层 '+side+' '+file,child/file)
    atomic_json(root/'plan.json',dict(project='STTW_CONTROL',purpose='Independent geometry zero-middle comparator; not a formula retune',declared_cases=24,declared_prefixes=8,maximum_lower_ticks=27200,source_sha256=sources,paired_H75_source=str(old),verified_zero_inputs=len(obs),training_steps=0))
    atomic_json(root/'campaign_status.json',dict(phase='prepared',declared_cases=24,completed_cases=0,active_lower_ticks=0,training_steps=0))
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 新几何零中层24分支总控',root/'campaign_status.json')
    print('prepared24 branches;8 genuine prefixes;0 training;maximum27200 lower ticks')


def execute(root):
    plan=json.loads((root/'plan.json').read_text());s=dict(phase='running',pid=os.getpid(),stage='starting',declared_cases=24,completed_cases=0,active_lower_ticks=0,training_steps=0)
    def update(**kw):s.update(kw,checked_unix=time.time());atomic_json(root/'campaign_status.json',s)
    update()
    try:
        for side in ['left','right']:
            child=root/side
            with (child/'controller.log').open('x') as f:
                proc=subprocess.Popen([sys.executable,'-u','docs/evidence/tolerance_progress_20261022/alpha_physical_boundary/execute.py','--run',str(child)],stdout=f,stderr=subprocess.STDOUT)
                update(stage=side,child_pid=proc.pid);code=proc.wait()
            result=json.loads((child/'campaign_status.json').read_text())
            if code or result['phase']!='completed':raise RuntimeError(side+' failed; preserve original; no retry')
            s['completed_cases']+=result['completed_cases'];s['active_lower_ticks']+=result['active_lower_ticks']
            if s['active_lower_ticks']>plan['maximum_lower_ticks']:raise ValueError('Budget exceeded')
            update()
        update(phase='completed',stage='all24_zero_middle_branches_ready')
    except BaseException as exc:update(phase='error',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare)(a.run.resolve())
