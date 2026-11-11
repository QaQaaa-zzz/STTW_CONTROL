"""Run a frozen finite evidence queue, release GPU between stages, notify completion."""
import argparse,json,subprocess,sys,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();q=a.queue;plan=json.loads((q/'plan.json').read_text());python=sys.executable

def command(args,log):
    with log.open('a') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)

def status(stage,**fields):
    (q/'status.json').write_text(json.dumps(dict(stage=stage,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),indent=2))

def notify(message,receipt):
    command([python,'learning/scripts/notify_sttw_job.py','--message',message,'--receipt',str(receipt)],q/'notifications.log')

current='preparing';completed=[]
try:
    for item in plan['stages']:
        r=Path(item['path']);current=item['name']
        if a.resume and (r/'analysis_summary.json').exists() and json.loads((r/'status.json').read_text())['stage']=='complete':
            completed.append(current);continue
        status('preparing',current=current,completed=completed)
        if current=='h75_height_scan':
            for scene in sorted(r.glob('*_scene.json')):
                if a.resume and (r/'terrains'/scene.stem.removesuffix('_scene')/'terrain.npz').exists():continue
                command([python,'learning/cli/build_route_pebbles.py','--scene-config',str(scene),'--output',str(r/'terrains'/scene.stem.removesuffix('_scene'))],r/'terrain_build.log')
            command([python,str(r/'check_geometry.py')],r/'terrain_build.log')
        if current=='h75_selector_layout_test':
            command([python,'learning/cli/freeze_height_selector.py','--development',plan['stages'][0]['path'],'--test-run',str(r)],r/'freeze.log')
            command([python,str(q/'prepare_test_layouts.py')],r/'terrain_build.log')
            for scene in sorted(r.glob('*_scene.json')):
                if a.resume and (r/'terrains'/scene.stem.removesuffix('_scene')/'terrain.npz').exists():continue
                command([python,'learning/cli/build_route_pebbles.py','--scene-config',str(scene),'--output',str(r/'terrains'/scene.stem.removesuffix('_scene'))],r/'terrain_build.log')
        status('queued_or_running',current=current,completed=completed)
        command(['/bin/bash','learning/scripts/gpu_queue.sh',python,'-u','learning/cli/run_alpha_mechanism.py','--run',str(r)]+(['--resume'] if a.resume else []),r/'runtime.log')
        status('analyzing',current=current,completed=completed)
        command([python,'learning/cli/summarize_alpha_schedules.py','--run',str(r)],r/'analysis.log')
        if item.get('state_selection'):
            command([python,'learning/cli/analyze_state_alpha_scan.py','--run',str(r)],r/'state_selection.log')
        completed.append(current)
        notify(f'STTW阶段完成：{current}，{item["contexts"]}工况/{item["episodes"]}条；结果{r}/ANALYSIS.md。后续自动排队。',r/'completion_notice.json')
    status('complete',completed=completed)
    notify(f'STTW全部证据队列完成：{plan["total_episodes"]}条，{len(plan["stages"])}阶段；索引{q}/README.md',q/'completion_notice.json')
except Exception as error:
    status('error',current=current,completed=completed,error=str(error))
    notify(f'STTW证据队列异常停止：{current}，见{q}/status.json；未自动扩大或重跑。',q/'error_notice.json')
    raise
