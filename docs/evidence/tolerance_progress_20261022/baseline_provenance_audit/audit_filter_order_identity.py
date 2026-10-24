"""Exact filter identity only: no vehicle integration or outcome prediction."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import sympy as sp

from sttw_control.piecewise_reference import mean_matched_library


def main():
    worktree=Path(__file__).resolve().parents[4]
    root=worktree/'runs/filter_order_identity_audit_v1'
    root.mkdir(exist_ok=False)
    run=worktree/'runs/temporal_reference_opportunity_v1_attempt2'
    design=json.loads((run/'design.json').read_text())
    plan=json.loads(Path(design['plan']).read_text())
    declaration=Path(plan['source_declaration'])
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert digest(declaration)==design['metadata_sha256'][str(declaration)]
    cfg=json.loads(declaration.read_text())['config']
    a,f0,q1,q2=sp.symbols('a f0 q1 q2',real=True)
    midpoint=a*f0+(1-a)*q1
    sequential=a*midpoint+(1-a)*q2
    mean=a*a*f0+(1-a*a)*(q1+q2)/2
    predicted=(1-a)**2*(q2-q1)/2
    assert sp.simplify(sequential-mean-predicted)==0
    reversed_order=a*(a*f0+(1-a)*q2)+(1-a)*q1
    assert sp.simplify(sequential-reversed_order-(1-a)**2*(q2-q1))==0
    # Discrete post-update sum identity from f_k=alpha*f_(k-1)+(1-alpha)*q_k.
    alpha,dt,total_f,total_q,fn=sp.symbols('alpha dt total_f total_q fn',real=True)
    recurrence_sum=sp.Eq(total_f,alpha*(total_f+f0-fn)+(1-alpha)*total_q)
    solved=sp.solve(recurrence_sum,total_f)[0]
    assert sp.simplify(dt*(solved-total_q)+alpha*dt/(1-alpha)*(fn-f0))==0
    constants,sequences=mean_matched_library(20261024)
    by_id={c['candidate']:c for c in constants}
    rows=[]
    for channel,tau in enumerate(cfg['residual_interface']['filter_tau_s']):
        lower_dt=cfg['timing']['lower_dt_s'];half=round(.2/lower_dt)
        alpha=math.exp(-lower_dt/tau);a=alpha**half;coefficient=(1-a)**2/2
        errors=[];dose_errors=[];gaps=[]
        for sequence in sequences:
            q=np.repeat(np.asarray(sequence['target_deltas'])[:,channel],half)
            c=np.repeat(np.asarray(by_id[sequence['mean_candidate']]['target_deltas'])[:,channel],half)
            def simulate(targets):
                f=.123;values=[]
                for target in targets:
                    f=alpha*f+(1-alpha)*target;values.append(f)
                return np.asarray(values)
            fs,fc=simulate(q),simulate(c)
            gap=fs[-1]-fc[-1]
            exact=coefficient*(q[-1]-q[0])
            errors.append(abs(gap-exact))
            dose_errors.append(abs(lower_dt*(fs-fc).sum()+alpha*lower_dt/(1-alpha)*gap))
            gaps.append(gap)
        assert max(errors)<2e-12 and max(dose_errors)<2e-12
        rows.append(dict(channel=['speed','steer'][channel],tau_s=tau,half_ticks=half,
                         endpoint_gap_coefficient=coefficient,
                         max_abs_endpoint_gap_in_fixed_library=float(np.max(np.abs(gaps))),
                         max_float64_endpoint_identity_error=max(errors),
                         max_float64_dose_identity_error=max(dose_errors)))
    receipt=dict(symbolic_endpoint_verified=True,symbolic_order_reversal_verified=True,
                 symbolic_discrete_dose_verified=True,numerical_rows=rows,
                 sources={str(p):digest(p) for p in [Path(__file__),declaration,
                     worktree/'learning/src/sttw_control/path_command_policy.py',
                     worktree/'learning/src/sttw_control/piecewise_reference.py']},
                 new_physics_ticks=0,training_steps=0,model_fits=0,
                 scope='Unclipped first-order correction filter only; excludes slew, amplitude back-calculation, actuator, vehicle and Actor response. No tolerance, recovery or novelty claim.')
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
