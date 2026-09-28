#!/usr/bin/env python3
"""Read-only completion detector. Never launches/resumes experiments or training.

Exit codes: 0 all declared checks finished without failure; 1 a check failed;
2 still running; 3 missing/interrupted/malformed evidence. These are CHECK states,
not experiment gate acceptance. --watch exits once finished or interrupted.
"""
import argparse
import json
from pathlib import Path
import time


def alive(check):
    pid=check.get('pid')
    if not pid:return False
    try:
        cmd=(Path('/proc')/str(int(pid))/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        token=check.get('process_token')
        return bool(token and token in cmd)
    except (OSError,ValueError):return False


def inspect(run):
    run=Path(run);manifest=json.loads((run/'checks_watch.json').read_text());checks=[]
    for declared in manifest['checks']:
        result=run/declared['result'];state='running' if alive(declared) else 'missing_or_interrupted'
        row=dict(name=declared['name'],result=str(result),state=state)
        if result.exists():
            try:
                data=json.loads(result.read_text())
                field=declared.get('pass_field')
                if field is None:row.update(state='finished',passed=None)
                elif type(data.get(field)) is bool:row.update(state='finished',passed=data[field])
                else:row.update(state='missing_or_interrupted',error='missing boolean acceptance field')
                row['result_summary']=data
            except (OSError,ValueError) as exc:row['error']=str(exc)
        checks.append(row)
    finished=bool(checks) and all(c['state']=='finished' for c in checks)
    failed=any(c.get('passed') is False for c in checks)
    if finished:status='finished_with_failed_check' if failed else 'checks_finished'
    elif any(c['state']=='missing_or_interrupted' for c in checks):status='missing_or_interrupted'
    else:status='running'
    out=dict(status=status,all_finished=finished,
        all_checks_passed=finished and not failed and all(c.get('passed') is True for c in checks),
        checks=checks,next_action='Wait for user; do not auto-run later gates.',
        experiment_success=False)
    budget=run/'budget.json'
    if budget.exists():
        d=json.loads(budget.read_text());out['budget']={k:d[k] for k in ['predictor_ticks','predictor_limit','plant_ticks','main_episodes','contract_rollouts'] if k in d}
    return out


def notification_text(result):
    title='偏好控制器仿真结束' if result['all_finished'] else '偏好控制器检查异常'
    if result['status']=='finished_with_failed_check':
        body='本次已启动的短检查均已结束，其中一项检查未通过。\n144 个重复候选结果存在差异，需要诊断。\n温和指令、冲突与航向恢复、随机面板尚未运行。\n已按你的要求暂停，不会自动继续。'
    elif result['all_finished']:
        body='本次已登记的检查均已结束。\n检查结束不代表所有实验 gate 通过；请查看检测报告。\n后续面板尚未运行，等待你通知继续。'
    else:
        body='检查结果缺失、损坏或进程已退出。\n未记为通过，不会自动启动后续实验。'
    return title,body


def notify(run,result):
    import fcntl,os,shutil,subprocess
    run=Path(run);record=run/'completion_notification.json'
    with (run/'.completion_notification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if record.exists():return json.loads(record.read_text())
        title,body=notification_text(result)
        env=os.environ.copy()
        env.setdefault('DISPLAY',':0')
        env.setdefault('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}')
        env.setdefault('DBUS_SESSION_BUS_ADDRESS','unix:path='+env['XDG_RUNTIME_DIR']+'/bus')
        if shutil.which('zenity'):
            with (run/'completion_popup.log').open('a') as log:
                child=subprocess.Popen(['zenity','--info','--no-wrap','--width=540','--title='+title,'--text='+body],
                    env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            time.sleep(.25)
            if child.poll() not in (None,0):raise RuntimeError('desktop popup failed; see completion_popup.log')
            receipt=dict(backend='zenity',pid=child.pid,title=title,status=result['status'],sent_at_unix=time.time())
        else:
            done=subprocess.run(['notify-send','--print-id','--app-name=STTW_CONTROL','--expire-time=0',title,body],env=env,capture_output=True,text=True,check=True)
            receipt=dict(backend='notify-send',notification_id=done.stdout.strip(),title=title,status=result['status'],sent_at_unix=time.time())
        tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,record)
        return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--notify',action='store_true',help='Show one desktop popup when checks finish; persistent deduplication');parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=float,default=5.)
    parser.add_argument('--json',action='store_true');args=parser.parse_args()
    if args.interval<.1:parser.error('--interval must be >= .1 seconds')
    root=Path(__file__).resolve().parents[2]
    if args.run is None:
        args.run=root/json.loads((root/'runs/teleop_pref_governor_v1/ACTIVE_CHECKS.json').read_text())['run']
    while True:
        result=inspect(args.run)
        if args.json:print(json.dumps(result,indent=2),flush=True)
        else:
            print('Check status:',result['status'],flush=True)
            for c in result['checks']:print(f"  {c['name']}: {c['state']}; passed={c.get('passed','not available')}",flush=True)
            print('Experiment gates are not certified by check completion. No automatic continuation.',flush=True)
            if 'budget' in result:print('Budget:',json.dumps(result['budget']),flush=True)
        if args.notify and result['status']!='running':
            print('Popup:',json.dumps(notify(args.run,result),ensure_ascii=False),flush=True)
        if not args.watch or result['status']!='running':
            return {'checks_finished':0,'finished_with_failed_check':1,'running':2,'missing_or_interrupted':3}[result['status']]
        time.sleep(args.interval)

if __name__=='__main__':raise SystemExit(main())
