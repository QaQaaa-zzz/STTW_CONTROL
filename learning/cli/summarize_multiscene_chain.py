"""Report engineering stage coverage for every declared continuous-task scene."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.continuous_task import ContinuousTask

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text());rows=[];n=len(m['cases']);fig,axes=plt.subplots((n+1)//2,2,figsize=(13,4*((n+1)//2)),layout='constrained',squeeze=False)
for ax,c in zip(axes.flat,m['cases']):
    spec=json.loads(Path(c['config']).read_text());out=Path(spec['output']);s=json.loads((out/'chain_summary.json').read_text());physical=json.loads((out/'results.json').read_text())
    z=np.load(out/'trajectory.npz');trace=z['trace'][z['valid'][:,0],0];route=np.load(out/'ground_route.npz')['xy']
    # Show the declared task and observed travel; omit the unused continuation tail.
    view_end=max(float(trace[:,1].max()),spec['completion']['final_goal_xy'][0])+2.
    visible_route=route[route[:,0]<=view_end]
    ax.plot(*visible_route.T,'k:',label='Original reference (task view)');ax.plot(trace[:,1],trace[:,2],label='Actual XY')
    ax.plot(trace[-1,1],trace[-1,2],'o' if s['complete'] else 'x',color='red',label='Observed endpoint')
    ax.axvspan(spec['platform_front_x'],spec['platform_back_x'],alpha=.12,color='gray',label='Platform X range')
    for filename,marker,label in [('connection_decision.json','s','Approach decision'),('handoff.json','^','Jump handoff'),('landing_handoff.json','v','Ground return')]:
        f=out/filename
        if f.exists():
            event=json.loads(f.read_text());point=np.asarray(event['root'])[0,:2];ax.plot(*point,marker,markersize=6,label=label)
    ax.set_aspect('equal',adjustable='box');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f"{c['case_id']}; full chain={s['complete']}");ax.legend(fontsize=6)
    program=dict(goals=[dict(goal_id='destination',allowed_routes=['original'],required_skills=['platform_jump','ground_return','continued_drive'])])
    routes=[dict(route_id='original',goal_id='destination',task_id=c['case_id'],skill_id='update2',candidates=['fixed'],reference_path=spec['ground_route'])]
    task=ContinuousTask(program,routes,[dict(candidate_id='fixed',speed_m_s=2.,alpha=.5)],None)
    task.record_execution('original','fixed',time_s=0.,reason='Recorded frozen full-route execution')
    skills=[]
    if s['checks']['stable_landing']:skills.append('platform_jump')
    if s['checks']['ground_return']:skills.append('ground_return')
    if s['checks']['continued_time'] and s['checks']['continued_forward']:skills.append('continued_drive')
    accepted=task.observe_goal('destination',arrived=s['checks']['ordered_approach'] and s['checks'].get('final_goal',False),upright=s['checks']['full_window'],within_allowed_region=s['checks']['original_path_corridor'],completed_skills=skills)
    assert accepted==s['complete']
    progress=dict(status='complete' if accepted else 'incomplete',role='Post-run physical task-progress replay; not online LLM execution',events=task.events,remaining=[] if accepted else ['destination'])
    (out/'task_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    rows.append(dict(scene=c['case_id'],unmet_checks=[k for k,v in s['checks'].items() if not v],stage_reached=next((k for k in ['continued','ground_return','strict_support','airborne_after_handoff','reached_release_line'] if s['stages'][k]),'ground_approach'),checks=s['checks'],stages=s['stages'],stage_times=s['stage_times'],complete=s['complete'],duration_s=float(trace[-1,0]),terminal_flags=physical['cases'][0]['terminal_flags'],task_progress=str(out/'task_progress.json'),xy=str(out/'xy_jump.png')))
for ax in list(axes.flat)[n:]:ax.set_visible(False)
fig.savefig(a.manifest.parent/'xy_all_scenes.png',dpi=150);plt.close(fig)
result=dict(role='Engineering integration, no success-rate adoption gate',scenes=rows,optimizer_updates=0)
(a.manifest.parent/'stage_summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Multi-scene continuous-task integration','','![All declared scenes](xy_all_scenes.png)','',
       'Frozen update2 + H75/ECBC, zero lower residual. New geometry; one run per scene, no robustness claim. Existing strict labels retained.',
       'Task progress is replayed from physical results after each run; not online LLM completion.', '',
       '| Scene | Reached jump | Airborne | Strict support | Ground return | Continued | Full chain |', '|---|---|---|---|---|---|---|']
for r in rows:lines.append('| '+r['scene']+' | '+' | '.join(str(r['stages'][k]) for k in ['reached_release_line','airborne_after_handoff','strict_support','ground_return','continued'])+' | '+str(r['complete'])+' |')
lines.extend(['', '## Unfinished stages and physical termination', '', '| Scene | Furthest observed stage | Unmet checks | Termination evidence |', '|---|---|---|---|'])
for r in rows:
    flags=r['terminal_flags'];stop=[k for k in ['ground_domain_exit','ground_policy_fault','roll_limit','low_root','nonfinite_state'] if flags.get(k)]
    if not flags.get('ground_alive',True):stop.append('ground_alive=false')
    lines.append('| '+r['scene']+' | '+r['stage_reached']+' | '+', '.join(r['unmet_checks'])+' | '+(', '.join(stop) or 'No recorded termination flag')+' |')
(a.manifest.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([{k:r[k] for k in ('scene','stages','complete')} for r in rows]))
