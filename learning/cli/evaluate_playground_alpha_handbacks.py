"""Check the fixed-alpha handback prerequisite on existing successful prefixes."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_playground_task_reversal import write_rows
from run_playground_wide_gap_oracle import run, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--report', type=Path, help='Read completed batches and export their trajectories')
    args=parser.parse_args()
    target=args.run.resolve()
    plan=json.loads((target/'plan.json').read_text())
    source=Path(plan['input'])
    config=json.loads((source/'screen.json').read_text())
    destination=args.report or target
    destination.mkdir(parents=True,exist_ok=True)
    lanes=[*plan['lanes'],plan['control']]
    fig,axes=plt.subplots(len(lanes),1,figsize=(10,2.2*len(lanes)),layout='constrained',squeeze=False)
    rows=[]
    for alpha in plan['alphas']:
        folder=source/'screen'
        if alpha:
            alphas=list(config['oracle_initial_alpha'])
            for lane in plan['lanes']:
                alphas[lane]=alpha
            folder=target/f'alpha{alpha:g}'
            if args.report is None:
                run(dict(config,oracle_initial_alpha=alphas),folder,'source',10,target)
        results=json.loads((folder/'status.json').read_text())['results']
        audits=json.loads((folder/'jump_validation.json').read_text())['cases']
        for panel,lane in zip(axes[:,0],lanes):
            result,audit=results[lane],audits[lane]
            path=Path(result['case_dir'])/'trajectory.npz'
            with np.load(folder/path) as z,np.load(source/'screen'/path) as initial:
                for field in ('qpos','qvel'):
                    np.testing.assert_array_equal(z[field][0],initial[field][0])
                control_error=float(np.max(np.abs(z['qpos']-initial['qpos']))) if lane==plan['control'] and z['qpos'].shape==initial['qpos'].shape else None
                trace=z['trace']
                if not alpha:
                    panel.plot(*z['published_reference_xy'].T,':',color='0.4',label='Reference',lw=.8)
                color={0:'#0072b2',.5:'#d55e00',1:'#009e73'}[alpha]
                label=f'Batch {alpha:g} (fixed alpha 0)' if lane==plan['control'] else f'Approach alpha {alpha:g}'
                panel.plot(*z['qpos'][:,:2].T,color=color,label=label,lw=1.2)
                panel.plot(*z['qpos'][-1,:2],'x',color=color)
                panel.set(aspect='equal',xlim=(1,18),ylim=(-1.4,1.4),xlabel='World x (m)',ylabel='World y (m)',
                          title=f"Root {result['variant']['root_variant']} | lane {lane}")
                entry=result['rb_entry_s']
                speed=None
                if entry is not None:
                    k=np.argmin(np.abs(trace[:,0]-entry))
                    speed=float(trace[k,7]*np.cos(trace[k,6])+trace[k,8]*np.sin(trace[k,6]))
            load=min((t for t in (audit['solid_bank_landing_s'],audit['both_far_after_flight_s']) if t is not None),default=None)
            hand=result['handoff_s']
            rows.append(dict(batch_alpha=alpha,approach_alpha=0 if lane==plan['control'] else alpha,
                unchanged_control=lane==plan['control'],root=result['variant']['root_variant'],lane=lane,
                entry_s=entry,entry_forward_m_s=speed,flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in audit['events']),
                loaded_s=load,handoff_s=hand,qualified=bool(load is not None and hand is not None and load<=hand+1e-5),
                goal=result['goal'],termination=result['termination'],control_qpos_max_diff=control_error))
        write_rows(destination/'alpha_handbacks.csv',rows)
    for ax in axes[:,0]:
        ax.legend(ncol=4,fontsize=8)
    fig.savefig(destination/'alpha_handbacks_xy.png',dpi=145)
    save(destination/'terminal.json',dict(scored=len(rows),experimental_cases=sum(not r['unchanged_control'] for r in rows)))


if __name__=='__main__':
    main()
