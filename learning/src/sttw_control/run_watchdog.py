"""Independent Linux desktop completion and fault monitor. Never modifies or restarts a run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
        fields=text.rsplit(')',1)[1].split()
        if fields[0] in ('Z','X'):return None
        return fields[19]  # /proc stat field 22: starttime, robust to spaces in comm.
    except (OSError,IndexError):return None


def inspect_run(status,alive):
    try:
        value=json.loads(Path(status).read_text())
        if not isinstance(value,dict):raise ValueError('status is not an object')
    except (OSError,ValueError) as e:
        if not alive:return 'error',f'Process exited without readable final status: {e}'
        return 'retry',str(e)
    phase=value.get('phase',value.get('state',''))
    if phase in ('error','failed','failure','crashed'):
        return 'error',str(value.get('error',value.get('message',phase)))
    if phase in ('complete','completed','success') or value.get('complete') is True:
        return 'complete',None
    if not alive:return 'error',f'Process exited unexpectedly; last phase: {phase}'
    return 'running',None


def popup(title,message,*,error=True):
    # argv only: error text is never interpreted as shell commands or markup.
    if shutil.which('zenity'):
        p=subprocess.Popen(['zenity','--error' if error else '--info','--no-markup','--title',title,'--text',message,'--width=600'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        time.sleep(.2)
        if p.poll() is None or p.returncode==0:return 'zenity'
    if shutil.which('notify-send'):
        subprocess.run(['notify-send','--urgency=critical' if error else '--urgency=normal',title,message],check=True,timeout=10)
        return 'notify-send'
    raise RuntimeError('No working desktop popup backend (zenity/notify-send)')


def notify_completed_stages(paths,notified):
    """Persistable stage events are separate from final pipeline completion."""
    for path in paths:
        path=Path(path);key=str(path.resolve())
        if key in notified or inspect_run(path,True)[0]!='complete':continue
        label='训练阶段' if path.parent.name=='training' else path.parent.name+'阶段'
        message=f'{label}已结束：{path.parent}\n\n阶段完成不代表模型达标。评估、出图是否结束请查看主流水线状态。'
        backend=popup('STTW '+label+'完成',message,error=False)
        notified[key]={'notified_unix':time.time(),'backend':backend,'message':message}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status',type=Path,required=True)
    p.add_argument('--launch',type=Path,required=True,help='JSON containing the pipeline launcher pid')
    p.add_argument('--output',type=Path,required=True,help='Independent watchdog state and alert records')
    p.add_argument('--stage-status',type=Path,action='append',help='Stage status; defaults to sibling training/status.json')
    p.add_argument('--interval',type=float,default=10.)
    p.add_argument('--confirmations',type=int,default=3)
    p.add_argument('--test-popup',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse a stopped monitor directory; do not notify completed runs twice')
    a=p.parse_args()
    if a.interval<=0 or a.confirmations<1:p.error('positive interval and confirmations required')
    previous={}
    if a.resume:
        previous=json.loads((a.output/'status.json').read_text())
        if previous['status_path']!=str(a.status.resolve()):p.error('resume status path mismatch')
        if previous.get('completion_notified'):return
        if process_identity(previous['watchdog_pid']) is not None:p.error('previous monitor is still running')
    stage_notified=previous.get('stage_notified',{})
    stages=a.stage_status if a.stage_status is not None else [a.status.parent/'training/status.json']
    a.output.mkdir(parents=True,exist_ok=a.resume)
    pid=int(json.loads(a.launch.read_text())['pid']);identity=process_identity(pid)
    def save(phase,**extra):
        value=dict(phase=phase,watchdog_pid=os.getpid(),watched_pid=pid,process_starttime=identity,status_path=str(a.status.resolve()),updated_unix=time.time(),stage_notified=stage_notified,**extra)
        temp=a.output/'status.tmp';temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(a.output/'status.json')
    try:
        if a.test_popup:popup('STTW 报警功能测试','这是监视器测试弹窗，不代表训练报错。关闭此窗口不影响监视。')
        count=0
        while True:
            notify_completed_stages(stages,stage_notified)
            alive=identity is not None and process_identity(pid)==identity
            phase,detail=inspect_run(a.status,alive)
            if phase=='complete':
                message=f'运行已正常结束：{a.status.parent}\n\n流水线状态已完成。请查看该目录的结果与分析索引。'
                (a.output/'completion.txt').write_text(message+'\n')
                backend=popup('STTW 仿真完成',message,error=False)
                save('complete',reason='watched pipeline completed',completion_notified=True,backend=backend)
                return
            count=count+1 if phase in ('error','retry') else 0
            save('monitoring',consecutive_errors=count,last_observation=phase,detail=detail)
            if count>=a.confirmations:
                message=f'运行：{a.status.parent}\n进程：{pid}\n错误：{detail}\n\n请查看该目录的状态文件及 training.log / evaluation.log。监视器没有停止或重启程序。'
                (a.output/'alert.txt').write_text(message+'\n')
                backend=popup('STTW 程序故障',message)
                save('alerted',detail=detail,backend=backend);return
            time.sleep(a.interval)
    except Exception as exc:
        save('error',error=repr(exc))
        try:popup('STTW 监视器自身异常',f'{exc}\n监视器输出：{a.output}')
        except Exception:pass
        raise
