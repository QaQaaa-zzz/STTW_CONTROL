"""Batch layout and padding only; physics/rewards remain in DirectCommandEnv."""
import numpy as np
import jax


def expand_cases(cases,methods,indices):
    if len(cases)*len(methods)>18:raise ValueError('fixed evaluation has at most18 trajectories')
    rows=np.stack([rows for _,rows,_ in cases for _ in methods]).astype(np.float32)
    alphas=np.tile(np.asarray(indices,np.float32),len(cases))
    bypass=np.tile(np.asarray([m=='B0' for m in methods]),len(cases))
    horizons=np.asarray([round(seconds/.005) for _,_,seconds in cases for _ in methods],np.int32)
    return rows,alphas,bypass,horizons


def masked_step(step,state,z,active,zero_log):
    return jax.lax.cond(active,lambda args:step(*args),lambda args:(args[0],zero_log),(state,z))


def timed_report(root,update,stage):
    """Separate reward reconstruction from plotting, including nested audits."""
    import time
    from . import preference_command_reporting as report
    from .direct_command_training import write
    timings=dict(reward_audit=0.,plotting=0.)
    original_audit,original_plots=report.reward_audit,report._plots
    def audit(*a,**kw):
        start=time.perf_counter()
        try:return original_audit(*a,**kw)
        finally:timings['reward_audit']+=time.perf_counter()-start
    def plots(*a,**kw):
        start=time.perf_counter();before=timings['reward_audit']
        try:return original_plots(*a,**kw)
        finally:timings['plotting']+=time.perf_counter()-start-(timings['reward_audit']-before)
    report.reward_audit,report._plots=audit,plots
    start=time.perf_counter()
    try:
        result=report.report_stage1(root,update) if stage==1 else report.report_stage2(root,update)
        if stage==51:
            for case,row in result.items():
                with np.load(root/f'evaluation{update}'/case/'B0.npz') as data:
                    row['B0']['reward_audit']=audit(dict(data),0,root,16)
            write(root/f'evaluation{update}/stage2_metrics.json',result)
    finally:report.reward_audit,report._plots=original_audit,original_plots
    timings['report_total']=time.perf_counter()-start
    write(root/f'evaluation{update}/report_timings.json',timings)
    if stage==51:
        checks=[row[f'alpha{a}']['reward_audit'] for row in result.values() for a in (0,1)] + [row['B0']['reward_audit'] for row in result.values()]
        if not checks or not all(row.get('passed') is True for row in checks):raise RuntimeError('independent numerical reward audit failed')
    return result


def policy_log_template(env,state):
    """Policy logs include scoring/motion fields absent from single-tick logs."""
    import jax.numpy as jp
    shape=jax.eval_shape(lambda s:env.policy_step(s,jp.zeros(2),False)[3],state)
    return jax.tree.map(lambda x:jp.zeros(x.shape,x.dtype),shape)
