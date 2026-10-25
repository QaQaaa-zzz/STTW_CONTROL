"""Offline paired recovery-task analysis, including both held and live tails."""
import argparse
import csv
import json
from pathlib import Path
import os
import time
import traceback
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.rejoin_comparison import geometric_task_trace
from sttw_control.survival_endpoint_task_metrics import task_metrics


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(root):
    source_status=json.loads((root/'status.json').read_text())
    if source_status['phase']!='completed' or source_status['completed_cases']!=36:
        raise ValueError('complete36 physical collection required, do not restart collection')
    out=root/'analysis';out.mkdir(exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),stage='read_actual_pose_and_geometry',physics_steps=0,optimizer_updates=0,checked_unix=time.time())
    atomic_json(out/'status.json',state)
    from prepare_short_recovery_campaign import register
    register(out/'ACTIVE_RUN.json','STTW_CONTROL 实时原路径接回36回合分析',out/'status.json')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cfg=json.loads((root/'common_environment_config.json').read_text())
        rows=json.loads((root/'results.json').read_text());index=json.loads((root/'case_index.json').read_text())
        assert len(rows)==len(index)==36
        byindex={r['case_id']:r for r in index};all_data={};augmented=[];paths={}
        for row in rows:
            item=byindex[row['case_id']];file=root/item['file']
            if digest(file)!=item['sha256']:raise ValueError('changed raw '+row['case_id'])
            with np.load(file) as z:data={k:z[k] for k in z.files}
            if item['reference'] not in paths:
                with np.load(root/item['reference']) as z:paths[item['reference']]={k:z[k] for k in z.files}
            path=paths[item['reference']]
            geometry=geometric_task_trace(data,path,cfg,initial_xy=row['initial_xy'])
            active=data['active_tick'];current_fault=data['nonfinite']|data['lower_fault']
            # Keep raw domain_invalid untouched. The NEW task explicitly allows
            # a tracking-domain crossing; this mask is metric input, not a state reset.
            diagnostic=dict(data,**geometry,task_reward_valid=active,
                domain_exit=np.zeros_like(active),policy_fault=np.zeros_like(active))
            m=task_metrics(diagnostic,speed=row['speed'],goal_s=float(path['goal']),config=cfg)
            if current_fault[active].any():raise ValueError('unresolved engineering fault')
            n=int(active.sum());observed_xy=data['actual_xy'][active]
            distance=float(np.linalg.norm(np.diff(np.vstack([row['initial_xy'],observed_xy]),axis=0),axis=1).sum())
            ref=data['reference'][active]
            variation=np.abs(np.diff(ref,axis=0)).sum(axis=0) if n>1 else np.zeros(2)
            domain=data['domain_invalid']&active;events=np.flatnonzero(domain)
            label='contact' if row['physical_outcome']=='contact' else 'safe_task' if m['task_recovered'] else 'safe_task_failed'
            if row['physical_outcome'] not in ('contact','survived') or m['task_recovered'] is None:label='unknown'
            result=dict(row,**m,category=label,domain_observed=bool(events.size),
                first_domain_interval_start_s=float(data['time'][events[0]]) if events.size else None,
                traveled_distance_m=distance,max_abs_original_path_error_m=float(np.max(np.abs(geometry['path_cross_track'][active]))),
                speed_ref_total_variation_m_s=float(variation[0]),steer_ref_total_variation_rad=float(variation[1]))
            augmented.append(result);all_data[row['case_id']]=dict(data,**geometry)
            np.savez_compressed(out/f"shadow_{len(augmented)-1:03d}.npz",**geometry)
        groups={}
        for row in augmented:groups.setdefault((row['role'],row['tail_rule']),[]).append(row)
        summary=[]
        for (role,tail),group in groups.items():
            summary.append(dict(role=role,tail_rule=tail,n=len(group),
                survived=sum(r['physical_outcome']=='survived' for r in group),
                joint_task_success=sum(r['category']=='safe_task' for r in group),
                contact=sum(r['category']=='contact' for r in group),unknown=sum(r['category']=='unknown' for r in group),
                mean_path_rmse_m=float(np.mean([r['task_path_rmse_m'] for r in group])),
                mean_observed_s=float(np.mean([r['task_observed_s'] for r in group])),
                mean_distance_m=float(np.mean([r['traveled_distance_m'] for r in group]))))
        paired=[];lookup={(r['source_case_id'],r['tail_rule']):r for r in augmented}
        for row in augmented:
            if row['tail_rule']!='held':continue
            live=lookup[row['source_case_id'],'live_zero']
            da,db=all_data[row['case_id']],all_data[live['case_id']]
            ia=da['prefix_active']&da['active_tick'];ib=db['prefix_active']&db['active_tick']
            common=min(int(ia.sum()),int(ib.sum()))
            xy_gap=float(np.max(np.abs(da['actual_xy'][ia][:common]-db['actual_xy'][ib][:common]))) if common else None
            roll_gap=float(np.max(np.abs(da['phi'][ia][:common]-db['phi'][ib][:common]))) if common else None
            paired.append(dict(source_case_id=row['source_case_id'],role=row['role'],route=row['route'],alpha=row['alpha'],
                held_id=row['case_id'],live_id=live['case_id'],held_outcome=row['category'],live_outcome=live['category'],
                survival_change=int(live['physical_outcome']=='survived')-int(row['physical_outcome']=='survived'),
                joint_task_change=int(live['category']=='safe_task')-int(row['category']=='safe_task'),
                path_rmse_change_m=live['task_path_rmse_m']-row['task_path_rmse_m'],
                observed_time_change_s=live['task_observed_s']-row['task_observed_s'],
                distance_change_m=live['traveled_distance_m']-row['traveled_distance_m'],
                prefix_ticks_held=int(ia.sum()),prefix_ticks_live=int(ib.sum()),
                prefix_max_xy_gap_m=xy_gap,prefix_max_roll_gap_rad=roll_gap))
        atomic_json(out/'results.json',augmented);atomic_json(out/'summary.json',summary)
        save_csv(out/'summary.csv',summary);save_csv(out/'paired_changes.csv',paired)
        # Each source pair gets actual XY first, then complete shadow task/error
        # and physical curves. NaNs remain gaps; no extrapolated post-contact tail.
        figures=[]
        for num,pair in enumerate(paired):
            a,b=lookup[pair['source_case_id'],'held'],lookup[pair['source_case_id'],'live_zero']
            path=paths[byindex[a['case_id']]['reference']]
            fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
            ax=axes.flat[0];ax.plot(path['xy'][:,0],path['xy'][:,1],'--',color='.6',label='Original path')
            finite_xy=[]
            for r,color,style in [(a,'#D55E00','--'),(b,'#0072B2','-')]:
                d=all_data[r['case_id']];valid=d['active_tick'];t=d['time'][valid];xy=d['actual_xy'][valid]
                finite_xy.append(xy);label=r['tail_rule']+' '+r['category']
                ax.plot(xy[:,0],xy[:,1],color=color,linestyle=style,label=label)
                ax.scatter(*xy[-1],color=color,marker='x' if r['physical_outcome']=='contact' else 'o',s=35)
                events=np.flatnonzero(d['domain_invalid']&valid)
                if events.size:ax.scatter(*d['actual_xy'][events[0]],color=color,marker='s',facecolors='none',s=45)
                fields=[('actual_forward_speed','Actual speed (m/s)'),('actual_delta','Actual steer (rad)'),
                    ('phi','Roll (rad)'),('phi_dot','Roll rate (rad/s)'),
                    ('path_cross_track','Original path error (m)'),('path_heading_error','Original heading error (rad)')]
                for panel,(key,title) in zip(axes.flat[1:7],fields):
                    panel.plot(t,d[key][valid],color=color,linestyle=style,label=r['tail_rule']);panel.set_ylabel(title)
                axes.flat[7].plot(t,d['reference'][valid,0],color=color,linestyle=style,label=r['tail_rule']);axes.flat[7].set_ylabel('Given speed reference (m/s)')
                axes.flat[8].plot(t,d['reference'][valid,1],color=color,linestyle=style,label=r['tail_rule']);axes.flat[8].set_ylabel('Given steer reference (rad)')
            cloud=np.concatenate(finite_xy);lo=cloud.min(axis=0)-1;hi=cloud.max(axis=0)+1
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=7)
            for panel in axes.flat[1:]:
                panel.axvspan(a['pulse_onset_task_tick']*.005,a['pulse_onset_task_tick']*.005+a['pulse_duration_s'],alpha=.15,color='#E69F00')
                panel.set_xlabel('Lower tick interval start (s)');panel.grid(alpha=.2)
            axes.flat[1].axhline(a['speed'],color='.5',linestyle=':',linewidth=1)
            fig.suptitle(f"{a['role']} | {a['route']} | alpha={a['alpha']} | {a['force_N']:g}N | user speed={a['speed']:g}m/s\n"
                         'Square=domain event; x=contact; circle=observed end. Both tails use actual-pose task diagnostics.')
            name=f'pair_{num:02d}.png';fig.savefig(out/name,dpi=125);plt.close(fig);figures.append(dict(source_case_id=a['source_case_id'],file=name))
        lines=['# Held versus live original-path zero-correction pilot','',
            '36 factual DEV episodes / 18 paired source conditions. Two related route/event contexts; no independent seeds or nominal retention claim.',
            'All task errors are recomputed from actual XY and exact yaw under one original-path projection. Domain exit is allowed by this new recovery-task contract; historical488 labels unchanged.',
            'Early-contact errors cover shorter observations and are not ranked as improved tracking. No speculative post-contact continuation.','',
            '|Role|Tail|n|No contact|Joint task|Contact|Unknown|Mean observed s|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:lines.append(f"|{r['role']}|{r['tail_rule']}|{r['n']}|{r['survived']}|{r['joint_task_success']}|{r['contact']}|{r['unknown']}|{r['mean_observed_s']:.3f}|")
        lines+=['','Paired survival gain/loss: '+str(sum(r['survival_change']>0 for r in paired))+'/'+str(sum(r['survival_change']<0 for r in paired)),
                'Paired joint-task gain/loss: '+str(sum(r['joint_task_change']>0 for r in paired))+'/'+str(sum(r['joint_task_change']<0 for r in paired)),
                '', '## Actual XY and complete diagnostics','',
                'Generated figures still require explicit visual review. Original prefix rewards/legacy domain penalty remain in raw arrays; they are not a new recovery-task total return.']
        lines += [f"- [{r['source_case_id']}]({r['file']})" for r in figures]
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        atomic_json(out/'receipt.json',dict(actual_episodes=36,paired_conditions=18,generated_figures=18,
            physics_steps=0,optimizer_updates=0,figures_visually_reviewed=False,
            raw_sha256={r['file']:r['sha256'] for r in index},analysis_code_sha256=digest(__file__)))
        state.update(phase='completed',stage='numeric_analysis_and_figures_generated_pending_visual_review',checked_unix=time.time());atomic_json(out/'status.json',state)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());state.update(phase='error',error=repr(exc),checked_unix=time.time());atomic_json(out/'status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run.resolve())
