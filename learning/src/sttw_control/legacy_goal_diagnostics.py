"""Post-completion figures and logged-reward reconciliation; no environment."""
import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
from .legacy_goal_analysis import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);a=parser.parse_args()
    root=a.root.resolve();source=root/'evaluation';out=root/'diagnostics'
    assert json.loads((source/'status.json').read_text())['phase']=='completed'
    assert json.loads((root/'analysis/status.json').read_text())['phase']=='completed'
    index=json.loads((source/'case_index.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    plan=json.loads((root/'plan.json').read_text());runtime=Path(plan['runtime_root'])/'sttw_control/path_command_reward.py'
    assert sha(runtime)==plan['runtime_expected_sources'][str(runtime)]
    tree=ast.parse(runtime.read_text())
    keys=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_COST_KEYS' for t in n.targets))
    traces={};cost_rows=[];max_reward_residual=0.;scale=manifest['config']['path_reward']['scale']
    for item in index:
        assert sha(item['raw_file'])==item['raw_sha256']
        with np.load(item['raw_file']) as z:d={k:z[k] for k in z.files}
        c=item['case'];cid=item['case_id'];active=d['active_tick'];traces[cid]=d
        assert d['effective_costs'].shape[1]==len(keys)  # This panel has no tolerance shaping.
        components=-scale*.005*d['effective_costs'][active]
        residual=d['tick_reward'][active]-components.sum(axis=1)
        max_reward_residual=max(max_reward_residual,float(abs(residual).max()))
        np.testing.assert_allclose(d['tick_reward'][active],components.sum(axis=1),atol=2e-4,rtol=1e-5)
        cost_rows.append(dict(case_id=cid,alpha=c['alpha'],role=c['role'],
            running_reward=float(d['tick_reward'][active].sum(dtype=np.float64)),
            scored_reward=float(d['scored_tick_reward'][active].sum(dtype=np.float64)),
            terminal_adjustment=float((d['scored_tick_reward'][active]-d['tick_reward'][active]).sum(dtype=np.float64)),
            **{f'signed_{k}':float(components[:,i].sum(dtype=np.float64)) for i,k in enumerate(keys)}))
    out.mkdir(exist_ok=False)
    with (out/'cost_integrals.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cost_rows[0]));w.writeheader();w.writerows(cost_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors={'parent':'#0072B2','task_only':'#D55E00','physical_credit':'#009E73','floor':'#882255','zero_middle':'#444444'}
    styles={'parent':'--','task_only':'-.','physical_credit':'-','floor':':','zero_middle':(0,(4,1,1,1))}
    figures=[]
    for route in ('left','right'):
        with np.load(source/(route+'_original_path.npz')) as z:p={k:z[k] for k in z.files}
        for event in sorted({x['case']['event_id'] for x in index}):
            selected=[x for x in index if x['case']['route_group']==route and x['case']['event_id']==event]
            figxy,xyaxes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
            extent=[]
            for ax,alpha in zip(xyaxes,(.5,.75)):
                subset=[x for x in selected if x['case']['alpha'] in (alpha,None)]
                goal=np.asarray(subset[0]['original_goal_xy']);ax.add_patch(Circle(goal,1,fill=False,color='black'));ax.plot(*goal,'k*')
                original=p['xy'][p['s']<=float(p['goal'])];ax.plot(*original.T,':',color='.65',label='Original path');extent.append(original)
                fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
                for item in subset:
                    c=item['case'];d=traces[item['case_id']];active=d['active_tick'];xy=d['actual_xy'][active];role=c['role'];t=(d['physical_tick'][active]+1)*.005
                    contact=d['contact_failure'][active];distance=np.linalg.norm(xy-goal,axis=1)
                    bad=np.flatnonzero(contact);stop=int(bad[0]) if len(bad) else len(t)
                    hits=np.flatnonzero((np.arange(len(t))<stop)&(distance<=1));end=int(hits[0])+1 if len(hits) else len(t)
                    extent.append(xy[:end]);label=role+(' (shared)' if c['alpha'] is None else '')
                    ax.plot(*xy[:end].T,color=colors[role],linestyle=styles[role],label=label)
                    ax.scatter(*xy[end-1],color=colors[role],marker='D' if len(hits) else 'x' if contact.any() else 's',s=28)
                    series=[distance,d['actual_forward_speed'][active]-c['requested_speed_m_s'],d['phi'][active],d['phi_dot'][active],d['reference'][active,0],d['reference'][active,1],d['tick_reward'][active],np.cumsum(d['scored_tick_reward'][active],dtype=np.float64)]
                    labels=['Original goal distance (m)','Original speed request error (m/s)','Roll (rad)','Roll rate (rad/s)','Speed reference (m/s)','Steer reference (rad)','Running reward /5ms (current route)','Cumulative scored reward (current route)']
                    for axis,values,ylabel in zip(axes.flat,series,labels):
                        axis.plot(t,values,color=colors[role],linestyle=styles[role],label=label);axis.set(xlabel='Task time, post-step (s)',ylabel=ylabel)
                        axis.grid(alpha=.2)
                        if c['force_N']:
                            onset=c['pulse_onset_task_tick']*.005;axis.axvspan(onset,onset+c['pulse_duration_s'],color='.85')
                        if len(hits):axis.scatter(t[end-1],values[end-1],color=colors[role],marker='D',s=12)
                    axes.flat[0].axhline(1,color=colors[role],linewidth=.4)
                axes.flat[0].legend(fontsize=7)
                fig.suptitle(f'{route}/{event}, original alpha={alpha:g}; mean Actor; each real endpoint retained\nV3 reward follows executed route, not canonical original-task utility; no cross-alpha reward ranking')
                name=f'trace_{route}_{event}_a{alpha:g}.png';fig.savefig(out/name,dpi=110,facecolor='white');plt.close(fig);figures.append(name)
                ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f'{route}/{event}, alpha={alpha:g}');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend(fontsize=7)
            cloud=np.vstack(extent);lo=cloud.min(axis=0)-1.1;hi=cloud.max(axis=0)+1.1
            for ax in xyaxes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
            figxy.suptitle('Task prefixes to first original-goal1m arrival (diamond), contact (x), or20s miss (square)\nPost-arrival tail omitted only in this view; complete20s paths retained in ../analysis/xy_*.png')
            name=f'task_prefix_{route}_{event}.png';figxy.savefig(out/name,dpi=130,facecolor='white');plt.close(figxy);figures.append(name)
    write(out/'receipt.json',dict(phase='completed',cases=len(index),figures=figures,max_logged_running_reward_component_residual=max_reward_residual,
        cost_keys=keys,lower_dt_s=.005,new_physics_steps=0,source_analysis_sha256=sha(root/'analysis/review.json'),
        module_sha256=sha(__file__),scope='Posthoc diagnostic only. V3 on executed route after replanning is not original-user Jalpha. Full observed windows and actual endpoints; task-prefix view explicitly stops on first arrival, complete20s separate. No smoothing or independent-sample uncertainty claims.'))
    print('diagnostics complete:',len(figures),'figures; reward residual',max_reward_residual)


if __name__=='__main__':main()
