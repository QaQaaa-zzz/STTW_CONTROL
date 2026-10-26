"""Read-only audit and actual/reference XY for the six real entry prefixes."""
import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from sttw_control.forward_entry_contexts import prefix_config
from sttw_control.geometric_path import build_path
from sttw_control.preference_best import digest


def analyze(root):
    root=Path(root).resolve();p=json.loads((root/'protocol.json').read_text())
    out=root/'analysis';out.mkdir(exist_ok=False)
    status=json.loads((root/'collection/status.json').read_text())
    if status['phase']!='completed':raise ValueError('completed collection required')
    receipt=json.loads((root/'collection/receipt.json').read_text())
    if digest(root/'protocol.json')!=receipt['protocol_sha256']:
        raise ValueError('protocol identity changed')
    if digest(root/'collection/full_states.pkl')!=receipt['full_states_sha256']:
        raise ValueError('full dynamic states changed')
    source=Path(p['source_run']);cfg=json.loads((source/'common_environment_config.json').read_text())
    with (root/'collection/initial_states.pkl').open('rb') as f:initial=pickle.load(f)
    with (root/'collection/full_states.pkl').open('rb') as f:final=pickle.load(f)
    with np.load(root/'collection/prefix_logs.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    dynamics,daxes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    records=[]
    for lane,spec in enumerate(p['prefixes']):
        raw={k:v[:,lane].reshape((-1,*v.shape[3:])) for k,v in arrays.items()}
        active=raw['active_tick'].astype(bool);ticks=int(active.sum())
        start=jax.tree.map(lambda a:a[lane],initial)
        end=jax.tree.map(lambda a:a[lane],final)
        if int(end.physical.physical_tick)-int(start.physical.physical_tick)!=ticks:
            raise ValueError('actual physical clock/active count disagreement')
        for key in ('contact_failure','nonfinite','lower_fault','policy_fault'):
            if key not in raw:raise ValueError('missing physical evidence '+key)
        local=prefix_config(cfg,spec)
        kind='straight' if spec['turn_angle_deg']==0 else ('left90_R2' if spec['turn_angle_deg']>0 else 'right90_R2')
        path=build_path(kind,np.asarray(start.physical.reference_pose),local)
        xy=raw['actual_xy'][active];time=np.arange(ticks)*.005
        ax=axes.flat[lane];extent=max(float(np.max(raw['path_progress'][active]))+1.,4.) if ticks else 4.
        keep=np.asarray(path.s)<=extent
        ax.plot(np.asarray(path.xy)[keep,0],np.asarray(path.xy)[keep,1],'--',color='0.45',label='Original prefix reference')
        ax.plot(xy[:,0],xy[:,1],color='#0072B2',label='Actual frozen lower / zero upper')
        ax.scatter(*np.asarray(start.physical.reference_pose)[:2],marker='o',color='k',s=20)
        if ticks:ax.scatter(*xy[-1],marker='x',color='#D55E00',s=40,label='Actual prefix end')
        ax.set(title=spec['name']+f' | {ticks}/320 ticks',xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        da=daxes.flat[lane];da.plot(time,raw['actual_forward_speed'][active],color='#0072B2',label='Actual speed (m/s)')
        da.axhline(spec['speed_m_s'],ls='--',color='0.4',label='Requested speed (m/s)')
        da.plot(time,raw['phi'][active],color='#D55E00',label='Roll (rad)')
        da.set(title=spec['name'],xlabel='Prefix time (s)',ylabel='Speed (m/s) / roll (rad)');da.grid(alpha=.2)
        records.append(dict(spec,ticks=ticks,valid=receipt['contexts'][lane]['valid'],
            max_abs_roll_rad=float(np.max(np.abs(raw['phi'][active]))) if ticks else None,
            initial_history_valid_frames=int(np.asarray(start.history.mask).sum()),
            final_history_valid_frames=int(np.asarray(end.history.mask).sum()),
            final_physical_tick=int(end.physical.physical_tick),final_task_tick=int(end.tick),
            dynamic_state_fields=list(end.__dataclass_fields__)))
    axes.flat[0].legend(fontsize=7);daxes.flat[0].legend(fontsize=7)
    fig.suptitle('Six real forward entry histories — shared bank3 ancestor; no task comparison yet')
    dynamics.suptitle('Prefix requests and actual motion — no force disturbance or policy training')
    fig.savefig(out/'prefix_actual_reference_xy.png',dpi=160)
    dynamics.savefig(out/'prefix_speed_roll.png',dpi=160)
    plt.close(fig);plt.close(dynamics)
    result=dict(contexts=records,valid_contexts=sum(r['valid'] for r in records),
                actual_lower_ticks=sum(r['ticks'] for r in records),new_physics_steps=0,
                optimizer_updates=0,visual_review_complete=False,
                scope='Source and target prefixes share an ancestor. Saved complete PathState includes lower/history/filter, before rebind. Valid prefix collection does not prove downstream neural increment or independent initial-root generalization.')
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='contexts'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    analyze(parser.parse_args().run)
