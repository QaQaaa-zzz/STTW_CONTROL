"""Static PNG/PDF and inspectable arrays; only completed physical prefixes shown."""
import csv,json
from pathlib import Path
import numpy as np

def save_trace(path,x):
    path=Path(path);np.savez_compressed(path.with_suffix('.npz'),**x)
    cols={}
    for k,v in x.items():
        v=np.asarray(v)
        if v.ndim==1:cols[k]=v
        else:
            flat=v.reshape(len(v),-1)
            for i in range(flat.shape[1]):cols[f'{k}_{i}']=flat[:,i]
    with path.with_suffix('.csv').open('w') as f:
        w=csv.writer(f);w.writerow(cols);w.writerows(zip(*cols.values()))

def plot_case(root,case,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(root)/'plots';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(15,16),constrained_layout=True)
    ax=axes.ravel();colors={'B0':'black','G0':'tab:blue','G1':'tab:orange'}
    for method,x in traces.items():
        t=x['time'];color=colors.get(method,'tab:green')
        ax[0].plot(t,x['actual_forward_speed'],label=method,c=color)
        ax[1].plot(t,x['actual_delta'],label=method,c=color)
        ax[2].plot(t,x['phi'],label=method,c=color)
        ax[2].plot(t,x['phi_dot'],label=method+' rate (rad/s)',c=color,ls=':')
        ax[3].plot(t,x['yaw_unwrapped'],label=method,c=color)
        ax[4].plot(t,x['e_psi_unwrapped'],label=method,c=color)
        ax[5].plot(t,x['applied_residual'][:,0],label=method+' front',c=color)
        ax[5].plot(t,x['applied_residual'][:,1],label=method+' rear',c=color,ls=':')
        ax[6].step(t,x['mode'],where='post',label=method,c=color)
        ax[7].plot(x['actual_xy'][:,0],x['actual_xy'][:,1],label=method,c=color)
        if x['physical_failure'][-1]:
            ax[7].scatter(*x['actual_xy'][-1],marker='x',c=color,s=80)
            for a in ax[:7]:a.axvline(t[-1],color=color,ls=':',alpha=.6)
    x=next(iter(traces.values()));t=x['time']
    ax[0].plot(t,x['limited_command'][:,0],'--',c='gray',label='original limited command')
    ax[1].plot(t,x['limited_command'][:,1],'--',c='gray',label='original limited command')
    ax[3].plot(t,x['reference_yaw_unwrapped'],'--',c='gray',label='original intent')
    ax[7].plot(x['reference_xy'][:,0],x['reference_xy'][:,1],'--',c='gray',label='intent only')
    for a,title in zip(ax,['Forward speed (m/s)','Actual steer angle (rad)','Roll (rad) / roll rate (rad/s)','Unwrapped heading (rad)','Unwrapped heading debt (rad)','Bounded residual (rad/s)','Mode: 0 TRACK, 1 CONFLICT, 2 RECOVER, 3 EMERGENCY','XY (m); no position objective']):
        a.set_title(title);a.grid(alpha=.25);a.legend(fontsize=8)
    ax[2].axhline(.3,color='red',ls='--',lw=.7);ax[2].axhline(-.3,color='red',ls='--',lw=.7)
    for a in ax[:7]:
        a.set_xlabel('Task time (s)')
        if case.startswith('C'):a.axvspan(3,5,color='gray',alpha=.12)
    ax[7].set_aspect('equal',adjustable='datalim');ax[7].set_xlabel('X (m)');ax[7].set_ylabel('Y (m)')
    fig.suptitle(f'{case} | frozen V1 | B0 / alpha 0 / alpha 1 as available | no policy checkpoint\nNo external disturbance; traces stop at actual physical failure')
    for suffix in ['png','pdf']:fig.savefig(out/f'{case}.{suffix}',dpi=150)
    plt.close(fig)

def report(root,status,metrics,case_manifest):
    root=Path(root)
    (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    (root/'case_manifest.json').write_text(json.dumps(case_manifest,indent=2)+'\n')
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    flat=[]
    def walk(obj,prefix=''):
        if isinstance(obj,dict):
            for k,v in obj.items():walk(v,f'{prefix}.{k}' if prefix else k)
        elif isinstance(obj,list):
            for i,v in enumerate(obj):walk(v,f'{prefix}[{i}]')
        else:flat.append((prefix,obj))
    walk(metrics)
    with (root/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['metric','value']);w.writerows(flat)
    budget=json.loads((root/'budget.json').read_text())
    pointer=root/'contracts/current_contracts.json'
    contract_link=json.loads(pointer.read_text())['result'] if pointer.exists() else 'contracts/physical_contracts.json'
    lines=['# Teleop preference governor V1 audit','',f"Status: **{status['status']}**.",'',
        'No training, no neural policy, no parameter tuning. Full-state oracle simulation only.',
        'Reference is original slew-limited v/delta intent. Heading recovery is distinct from XY recovery.','',
        '| Gate | Result |','|---|---|']
    for k,v in status.get('gates',{}).items():lines.append(f'| {k} | {v} |')
    lines+=['',f"Failure classification: {status.get('classification','none recorded')}.",
        '',f"Predictor reserved ticks: {budget['predictor_ticks']:,} / {budget['predictor_limit']:,}; plant reserved ticks: {budget['plant_ticks']:,}.",
        f"Main episodes: {budget['main_episodes']}; ablations: {budget['ablation_episodes']}; contract rollouts: {budget['contract_rollouts']}.",
        'Reserved work is conservative after interruptions; detailed events and physics substeps are in budget.json.',
        '', '## Evidence', '', '[Exact metrics](metrics.json), [flattened metrics CSV](metrics.csv), [budget](budget.json), [case manifest](case_manifest.json).',
        f'[Preparation](contracts/preparation.json), [physical contracts]({contract_link}), [numerical diagnosis](numerical_diagnosis/REPORT.md).','',
        '## Limits','', 'Unrun gates are NOT passed. Emergency, incomplete horizon and budget exhaustion disqualify task success.',
        'Compilation is reported separately; synchronized solve latency includes candidate generation, prediction and selection.','']
    for case in case_manifest:
        if (root/'plots'/f'{case}.png').exists():lines+= [f'## {case}','',f'![{case}](plots/{case}.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'INDEX.md').write_text('# Evidence index\n\n[Report](REPORT.md) · [Metrics](metrics.csv) · [Budget](budget.json)\n\n'+'\n'.join(f'- [{p.name}]({p.relative_to(root)})' for p in sorted(root.rglob('*.npz')))+'\n')
