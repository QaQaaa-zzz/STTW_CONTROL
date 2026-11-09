"""Report engineering stage coverage for every declared continuous-task scene."""
import argparse,json
from copy import deepcopy
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
    parent_comparison=None
    if c.get('parent'):
        parent=Path(c['parent'])/'evaluation'
        previous=np.load(parent/'trajectory.npz');before=previous['trace'][previous['valid'][:,0],0]
        previous_report=json.loads((parent/'chain_summary.json').read_text())
        ax.plot(before[:,1],before[:,2],'--',alpha=.7,label='Parent actual XY')
        ax.plot(before[-1,1],before[-1,2],'x',color='gray',label='Parent endpoint')
        cutoff=min(s['stage_times']['release'] or trace[-1,0],previous_report['stage_times']['release'] or before[-1,0])
        new_prefix=trace[trace[:,0]<cutoff];old_prefix=before[before[:,0]<cutoff]
        parent_comparison=dict(run=str(parent),checks=previous_report['checks'],stages=previous_report['stages'],prefix_end_exclusive_s=float(cutoff),prefix_identical=bool(np.array_equal(new_prefix,old_prefix)),new_prefix_samples=len(new_prefix),old_prefix_samples=len(old_prefix))
    ax.plot(trace[-1,1],trace[-1,2],'o' if s['complete'] else 'x',color='red',label='Observed endpoint')
    ax.axvspan(spec['platform_front_x'],spec['platform_back_x'],alpha=.12,color='gray',label='Platform X range')
    for filename,marker,label in [('connection_decision.json','s','Approach decision'),('handoff.json','^','Jump handoff'),('landing_handoff.json','v','Ground return')]:
        f=out/filename
        if f.exists():
            event=json.loads(f.read_text());point=np.asarray(event['root'])[0,:2];ax.plot(*point,marker,markersize=6,label=label)
    ax.set_aspect('equal',adjustable='box');ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f"{c['case_id']} | {spec['skills'][0]}; strict full chain={s['complete']}");ax.legend(fontsize=6)
    program=dict(goals=[dict(goal_id='destination',allowed_routes=['original'],required_skills=['platform_jump','ground_return','continued_drive'])])
    candidates=[dict(candidate_id='initial',speed_m_s=spec['approach_speed'],alpha=spec['ground_alpha'])]
    decision_file=out/'connection_decision.json'
    decision=json.loads(decision_file.read_text()) if decision_file.exists() else None
    if decision:candidates.append(dict(candidate_id='exit',speed_m_s=decision['candidate']['speed_m_s'],alpha=decision['candidate']['alpha']))
    routes=[dict(route_id='original',goal_id='destination',task_id=c['case_id'],skill_id=spec['skills'][0],candidates=[v['candidate_id'] for v in candidates],reference_path=spec['ground_route'])]
    task=ContinuousTask(program,routes,candidates,None)
    task.record_execution('original','initial',time_s=0.,reason='Initial reference reconstructed from executed config')
    if decision:task.record_execution('original','exit',time_s=decision['time'],reason='Recorded actual approach decision')
    skills=[]
    if s['checks']['stable_landing']:skills.append('platform_jump')
    if s['checks']['ground_return']:skills.append('ground_return')
    if s['checks']['continued_time'] and s['checks']['continued_forward']:skills.append('continued_drive')
    filtered_task=deepcopy(task) if s.get('contact_filtered_protocol') else None
    accepted=task.observe_goal('destination',arrived=s['checks']['ordered_approach'] and s['checks'].get('final_goal',False),upright=s['checks']['full_window'],within_allowed_region=s['checks']['original_path_corridor'],completed_skills=skills)
    assert accepted==s['complete']
    progress=dict(status='complete' if accepted else 'incomplete',role='Post-run physical task-progress replay; not online LLM execution',events=task.events,remaining=[] if accepted else ['destination'])
    if filtered_task is not None:
        checks=s['contact_filtered_protocol']['checks']
        filtered_skills=[name for name,passed in [('platform_jump',checks['stable_landing']),('ground_return',checks['ground_return']),('continued_drive',checks['continued_time'] and checks['continued_forward'])] if passed]
        accepted_filtered=filtered_task.observe_goal('destination',arrived=checks['ordered_approach'] and checks.get('final_goal',False),upright=checks['full_window'],within_allowed_region=checks['original_path_corridor'],completed_skills=filtered_skills)
        assert accepted_filtered==s['contact_filtered_protocol']['complete']
        progress['contact_filtered_20ms']=dict(status='complete' if accepted_filtered else 'incomplete',events=filtered_task.events,remaining=[] if accepted_filtered else ['destination'],scope='Separate declared deployment contact protocol; strict status above unchanged')
    (out/'task_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    rows.append(dict(scene=c['case_id'],parent_comparison=parent_comparison,skill=spec['skills'][0],filtered_protocol=s.get('contact_filtered_protocol'),unmet_checks=[k for k,v in s['checks'].items() if not v],stage_reached=next((k for k in ['continued','ground_return','strict_support','airborne_after_handoff','reached_release_line'] if s['stages'][k]),'ground_approach'),checks=s['checks'],stages=s['stages'],stage_times=s['stage_times'],complete=s['complete'],duration_s=float(trace[-1,0]),terminal_flags=physical['cases'][0]['terminal_flags'],task_progress=str(out/'task_progress.json'),xy=str(out/'xy_jump.png')))
for ax in list(axes.flat)[n:]:ax.set_visible(False)
fig.savefig(a.manifest.parent/'xy_all_scenes.png',dpi=150);plt.close(fig)
result=dict(role='Engineering integration, no success-rate adoption gate',scenes=rows,optimizer_updates=0)
(a.manifest.parent/'stage_summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Multi-scene continuous-task integration','','![All declared scenes](xy_all_scenes.png)','',
       'Frozen skills listed per scene + H75/ECBC, zero lower residual. One run per scene, no robustness claim. Original strict labels retained; contact-filter labels are separate.',
       'Task progress is replayed from physical results after each run; not online LLM completion.', '',
       '| Scene | Skill | Reached jump | Airborne | Strict support | Ground return | Continued | Strict full chain |', '|---|---|---|---|---|---|---|---|']
for r in rows:lines.append('| '+r['scene']+' | '+r['skill']+' | '+' | '.join(str(r['stages'][k]) for k in ['reached_release_line','airborne_after_handoff','strict_support','ground_return','continued'])+' | '+str(r['complete'])+' |')
paired=[r for r in rows if r['parent_comparison'] is not None]
if paired:
    lines.extend(['', '## Recorded parent comparison', '', '| Scene | Parent reached jump | Current reached jump | Ground prefix identical |', '|---|---|---|---|'])
    for r in paired:
        old=r['parent_comparison'];lines.append('| '+r['scene']+' | '+str(old['stages']['reached_release_line'])+' | '+str(r['stages']['reached_release_line'])+' | '+str(old['prefix_identical'])+' |')
filtered=[r for r in rows if r['filtered_protocol'] is not None]
if filtered:
    lines.extend(['', '## Separately declared 20ms contact protocol', '', '| Scene | Filtered support | Filtered full chain |', '|---|---|---|'])
    for r in filtered:
        f=r['filtered_protocol'];lines.append('| '+r['scene']+' | '+str(f['checks']['stable_landing'])+' | '+str(f['complete'])+' |')
lines.extend(['', '## Unfinished stages and physical termination', '', '| Scene | Furthest observed stage | Unmet checks | Termination evidence |', '|---|---|---|---|'])
for r in rows:
    flags=r['terminal_flags'];stop=[k for k in ['ground_domain_exit','ground_policy_fault','roll_limit','low_root','nonfinite_state'] if flags.get(k)]
    if not flags.get('ground_alive',True):stop.append('ground_alive=false')
    lines.append('| '+r['scene']+' | '+r['stage_reached']+' | '+', '.join(r['unmet_checks'])+' | '+(', '.join(stop) or 'No recorded termination flag')+' |')
(a.manifest.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([{k:r[k] for k in ('scene','stages','complete')} for r in rows]))
