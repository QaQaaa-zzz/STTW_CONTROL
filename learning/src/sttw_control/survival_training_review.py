"""Read-only stratified training diagnostics; no checkpoint selection or efficacy test."""
import numpy as np


def join_collection(cases,collection,*,update):
    if not collection.get('all_physical_tails_complete'):raise ValueError('complete physical tails required')
    seen=set();rows=[]
    for arm,values in collection['case_summaries'].items():
        for row in values:
            index=row['case_index']
            if index in seen or not 0<=index<len(cases):raise ValueError('duplicate/out-of-range case identity')
            case=cases[index]
            if (case['arm']!=arm or row['alpha']!=case['alpha'] or row['force_N']!=case['force_N']
                    or row['outcome'] not in ('contact','survived')):
                raise ValueError('case metadata or physical outcome mismatch')
            seen.add(index);rows.append(dict(case,**row,update=update))
    if len(seen)!=len(cases):raise ValueError('missing collection case')
    return rows


def summarize_strata(rows,*,dt_s):
    groups={}
    for r in rows:
        key=(r['update'],r['arm'],r['route'],r['alpha'],r['force_N'])
        groups.setdefault(key,[]).append(r)
    result=[]
    for key,group in groups.items():
        mean=lambda field:float(np.mean([r[field] for r in group]))
        result.append(dict(update=key[0],arm=key[1],route=key[2],alpha=key[3],force_N=key[4],episodes=len(group),
            survived=sum(r['outcome']=='survived' for r in group),contact=sum(r['outcome']=='contact' for r in group),
            hold_episodes=sum(r['hold_ticks']>0 for r in group),mean_hold_s=dt_s*mean('hold_ticks'),
            mean_task_scored_return=mean('task_scored_return'),mean_true_speed=mean('mean_true_speed'),
            mean_episode_max_task_path_error=mean('max_task_path_error'),
            mean_actor_decisions=mean('actor_decisions'),full_pulse_episodes=sum(r['full_pulse_exposure'] for r in group)))
    return result


def gradient_geometry(task_loss,physical_loss,parameters):
    """Instantaneous loss directions, before clipping/Adam. No optimizer step."""
    import torch
    def vector(loss):
        grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(parameters,grads)])
    task,physical=vector(task_loss),vector(physical_loss)
    a,b=torch.linalg.vector_norm(task),torch.linalg.vector_norm(physical)
    return dict(task_norm=float(a),physical_norm=float(b),combined_norm=float(torch.linalg.vector_norm(task+physical)),
        cosine=float(torch.dot(task,physical)/(a*b)) if float(a*b)>0 else None,
        physical_to_task_norm=float(b/a) if float(a)>0 else None)


def clipping_comparison(ratio,task_advantage,physical_advantage,*,beta,clip,denominator):
    """Compare declared separate surrogates with scalarize-then-clip, read-only.

    Uses the SAME frozen advantages/scales. There is no re-normalization,
    optimizer step, performance claim or suggestion to change a live run.
    """
    r,a,b=[np.asarray(x,np.float64) for x in (ratio,task_advantage,physical_advantage)]
    if (r.ndim!=1 or not r.size or r.shape!=a.shape or r.shape!=b.shape
            or not all(np.isfinite(x).all() for x in (r,a,b)) or np.any(r<0)
            or not np.isfinite(beta) or beta<0 or not 0<clip<1
            or not np.isfinite(denominator) or denominator<r.size):
        raise ValueError('aligned finite advantages, ratios, weight and denominator required')
    clipped=np.clip(r,1-clip,1+clip)
    surrogate=lambda x:np.minimum(r*x,clipped*x)
    separate=surrogate(a)+beta*surrogate(b);combined=surrogate(a+beta*b)
    gap=combined-separate
    return dict(rows=int(r.size),conflicting_rows=int(np.sum(a*(beta*b)<0)),
        clipped_ratio_rows=int(np.sum((r<1-clip)|(r>1+clip))),positive_gap_rows=int(np.sum(gap>1e-12)),
        separate_surrogate=float(separate.sum()/denominator),combined_surrogate=float(combined.sum()/denominator),
        combined_minus_separate=float(gap.sum()/denominator),max_row_gap=float(gap.max()))
