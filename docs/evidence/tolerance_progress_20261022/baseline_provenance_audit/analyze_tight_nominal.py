"""All declared lower-only tight-turn prefixes, without a task-success label."""
from pathlib import Path
import argparse,json,pickle
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve()
    plan=json.loads((run/'prepare_plan.json').read_text());status=json.loads((run/'prepare_status.json').read_text())
    assert status['phase']=='completed'
    results=json.loads((run/'prefix_results.json').read_text())['cases'];assert len(results)==len(plan['conditions'])==8
    decl=json.loads(Path(plan['source_declaration']).read_text());source=decl['source'];cc=source['controller']
    bank=pickle.loads(Path(source['prepared_bank']).read_bytes());idx=source['bank_index']
    origin=np.asarray(bank.data.qpos)[idx,:2];yaw0=float(np.asarray(bank.yaw_unwrapped)[idx]);dt=cc['dt']
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,9),layout='constrained')
    curves,caxes=plt.subplots(8,2,figsize=(12,23),layout='constrained')
    rows=[]
    for i,(r,ax) in enumerate(zip(results,axes.flat)):
        with np.load(r['source_npz']) as d:
            active=d['active_tick'].astype(bool);trace={k:d[k][active] for k in d.files}
        n=len(trace['time']);guide=[origin.copy()];heading=yaw0
        for speed,delta in trace['reference']:
            rate=speed*np.tan(delta)*np.cos(cc['caster'])/cc['wheelbase']
            midpoint=heading+.5*dt*rate
            guide.append(guide[-1]+dt*speed*np.array([np.cos(midpoint),np.sin(midpoint)]));heading+=dt*rate
        guide=np.asarray(guide);xy=trace['actual_xy']
        ax.plot(*guide.T,':',color='gray',label='prescribed command guide');ax.plot(*xy.T,label='actual local300')
        ax.scatter(*origin,c='black',marker='s');ax.scatter(*xy[-1],marker='o' if r['balance_label']=='NO_FLOOR_CONTACT' else 'x')
        ax.set(title=r['state_id'],xlabel='world x (m)',ylabel='world y (m)');ax.set_aspect('equal')
        time=trace['time']+dt
        caxes[i,0].plot(time,trace['reference'][:,0],':',label='prescribed speed reference')
        caxes[i,0].plot(time,trace['actual_forward_speed'],label='actual body speed')
        caxes[i,0].set(title=r['state_id'],xlabel='time (s)',ylabel='m/s')
        caxes[i,1].plot(time,trace['phi'],label='actual roll')
        caxes[i,1].plot(time,trace['reference'][:,1],':',label='prescribed steering reference')
        caxes[i,1].set(xlabel='time (s)',ylabel='rad')
        caxes[i,1].axhline(.3,color='gray',ls='--');caxes[i,1].axhline(-.3,color='gray',ls='--')
        rows.append(dict(**r,final_speed_error_m_s=r['actual_speed_m_s']-r['speed_ref_m_s'],
            max_abs_roll_rad=float(np.max(np.abs(trace['phi']))),reference_horizon_s=n*dt,
            scope='Lower-only no-force feasibility, not geometric path task success or middle policy execution.'))
    axes.flat[0].legend(fontsize=8);fig.suptitle('Frozen local300, prescribed ramped commands, no external force, no upper Actor')
    fig.savefig(run/'all_paths.png',dpi=150);plt.close(fig)
    caxes[0,0].legend(fontsize=8);caxes[0,1].legend(fontsize=8);curves.savefig(run/'speed_roll_reference.png',dpi=130);plt.close(curves)
    middle_config=Path(__file__).resolve().parents[4]/'runs/preference_full_tail_length75_v1/training_config.json'
    follower_config=json.loads(middle_config.read_text())['follower']
    summary=dict(cases=rows,active_lower_ticks=status['active_lower_ticks'],new_analysis_physics_steps=0,
        policy_updates=0,passed_no_contact=sum(r['balance_label']=='NO_FLOOR_CONTACT' for r in rows),
        original_follower_steer_reference_cap_rad=follower_config['steer_reference_max_rad'],
        original_middle_config=str(middle_config),
        integration_boundary='Targets .45/.55 exceed original middle follower nominal cap .3. Full-middle tight path needs an explicitly versioned common reference protocol; frozen models are unchanged.')
    (run/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(passed=summary['passed_no_contact'],active_lower_ticks=status['active_lower_ticks']),indent=2))


if __name__=='__main__':main()
