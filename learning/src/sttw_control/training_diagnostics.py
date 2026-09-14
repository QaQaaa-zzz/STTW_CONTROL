"""Source-backed PPO curves; reward plateaus are not convergence certificates."""
from pathlib import Path
import hashlib
import json
import numpy as np

PANELS=[('policy','Policy surrogate loss'),('value','Value loss (includes 0.5 factor)'),
        ('kl','Approx. KL diagnostic (symlog)'),('reward','Mean step reward (symlog; different objectives)'),
        ('entropy','Gaussian entropy before tanh'),('coverage','Disturbed samples / collected steps'),
        ('recovery','Development joint recovery fraction'),('failed','Development physical failure fraction'),
        ('path','Development radial RMSE [m; observed windows]'),('speed','Development speed RMSE [m/s; observed windows]'),('final_kl','Accepted policy exact Gaussian KL'),('yaw','Development yaw-rate RMSE [rad/s]')]


def series(rows):
    out={k:[] for k,_ in PANELS};out['steps']=[];out['update']=[]
    previous=0
    for row in rows:
        step=row['control_transitions'];out['steps'].append(step);out['update'].append(row['update'])
        losses=row.get('loss_metrics',[])
        for key,i in [('policy',0),('value',1),('entropy',2),('kl',3)]:out[key].append(float(losses[i]) if len(losses)>i else float('nan'))
        out['reward'].append(row.get('mean_step_reward',float('nan')))
        out['final_kl'].append(row.get('optimizer_audit',{}).get('final_exact_kl',float('nan')))
        disturbed=row.get('steer_disturbed_transitions',float('nan'))+row.get('force_disturbed_transitions',float('nan'))+row.get('rear_disturbed_transitions',0)
        out['coverage'].append(disturbed/(step-previous) if step>previous else float('nan'));previous=step
        v=row.get('validation',{});out['yaw'].append(float(np.mean(v['yaw_rmse'])) if 'yaw_rmse' in v else float('nan'));mask=np.asarray(v.get('event_present',[]),bool)
        for key,field in [('recovery','post_event_hold_complete'),('failed','failed'),('path','radial_rmse'),('speed','speed_rmse')]:
            a=np.asarray(v.get(field,[]))
            if 'yaw_rmse' in v and key in ('failed','speed'):
                out[key].append(float(a.mean()) if len(a) else float('nan'));continue
            out[key].append(float(a[mask].mean()) if len(a)==len(mask) and mask.any() else float('nan'))
    return out


def read_metrics(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('empty metrics')
    if any(b['control_transitions']<=a['control_transitions'] for a,b in zip(rows,rows[1:])):raise ValueError('non-increasing training steps')
    return rows


def draw(entries,destination,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(6,2,figsize=(15,19),layout='constrained')
    colors=plt.get_cmap('tab20')
    for i,(label,s) in enumerate(entries):
        for ax,(key,name) in zip(axes.flat,PANELS):
            x=np.asarray(s['steps'])/1e6;y=np.asarray(s[key],float);valid=np.isfinite(y)
            ax.plot(x[valid],y[valid],label=label,color=colors(i%20),marker='o',ms=3,lw=1.2)
    for ax,(key,name) in zip(axes.flat,PANELS):
        ax.set_title(name,loc='left');ax.set_xlabel('Collected control steps [million]');ax.grid(alpha=.2)
        if key=='reward':ax.set_yscale('symlog',linthresh=.005)
        if key in ['kl','value']:ax.set_yscale('symlog',linthresh=.01)
        if key in ['recovery','failed','coverage']:ax.set_ylim(-.03,1.03)
        if not any(np.isfinite(s[key]).any() for _,s in entries):ax.text(.5,.5,'Not logged',ha='center',transform=ax.transAxes)
    for ax in list(axes.flat)[len(PANELS):]:ax.set_visible(False)
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle(title+'\nRaw logged values; gaps mean unavailable. Development samples are not holdout.\nReward/loss across changed tasks are not a performance ranking; no automatic convergence claim.',fontsize=11)
    for ext in ['png','pdf']:
        tmp=destination/f'curves.tmp.{ext}';fig.savefig(tmp,dpi=130);tmp.replace(destination/f'curves.{ext}')
    plt.close(fig)


def plot_training(directory):
    directory=Path(directory);rows=read_metrics(directory/'metrics.jsonl');s=series(rows)
    draw([(directory.parent.name if directory.name=='training' else directory.name,s)],directory/'diagnostics','PPO training progress')
    return s


def compare_runs(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    entries=[];manifest=[];errors=[]
    for path in sorted(root.rglob('metrics.jsonl')):
        try:
            rows=read_metrics(path);s=series(rows)
            d=json.loads((path.parent/'declaration.json').read_text())
            if 'training' not in d:raise ValueError('missing PPO training declaration')
            label=str(path.parent.relative_to(root))
            dest=output/'runs'/path.parent.relative_to(root)
            draw([(label,s)],dest,label)
            first_failure=next((r['update'] for r in rows if 'validation' in r and np.any(r['validation'].get('failed',[]))),None)
            rec={'run':label,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'steps':rows[-1]['control_transitions'],'first_development_failure_update':first_failure,'last_approx_kl':s['kl'][-1],'task':d.get('task'),'training':d['training'],'figure':str((dest/'curves.png').relative_to(output))}
            manifest.append(rec);entries.append((label,s,rec))
        except (ValueError,KeyError,OSError) as exc:errors.append({'path':str(path),'error':str(exc)})
    # Include every run individually; paginate overlays so legends remain legible.
    groups=[('priority',[(name,s) for name,s,r in entries if (r['task'] or {}).get('priority') is not None and r['steps']>=1000000 and len(s['update'])>=8]),
            ('other',[(name,s) for name,s,r in entries if not ((r['task'] or {}).get('priority') is not None and r['steps']>=1000000 and len(s['update'])>=8)])]
    links=[]
    for name,group in groups:
        for start in range(0,len(group),6):
            dest=output/f'{name}_{start//6+1}';draw(group[start:start+6],dest,'PPO comparison: '+name)
            links.append(f'- [{name} {start//6+1}]({dest.name}/curves.png)')
    (output/'manifest.json').write_text(json.dumps({'runs':manifest,'unavailable':errors},indent=2,allow_nan=False)+'\n')
    (output/'INDEX.md').write_text('# PPO training curves\n\nAll discovered PPO metrics, including engineering smoke runs. Different objectives/budgets are not matched experiments.\n\n'+ '\n'.join(links)+'\n\n'+ '\n'.join(f"- [{r['run']}]({r['figure']}) — {r['steps']} steps" for r in manifest)+'\n\nUnavailable: '+json.dumps(errors,ensure_ascii=False)+'\n')
    return {'runs':len(manifest),'unavailable':errors}
