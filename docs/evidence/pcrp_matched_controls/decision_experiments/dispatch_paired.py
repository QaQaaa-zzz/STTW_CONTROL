"""Run the approved paired stages, preserve their gate outcomes, and notify by Bark."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from dispatch_step0 import record

ROOT = Path(__file__).resolve().parent
STATE = ROOT/'paired_services'
REPO = Path('/home/qy/STTW_CONTROL/runs/worktrees/sttw-gravel')
PYTHON = '/home/qy/miniconda3/envs/mjp/bin/python'
ALLOCATION = {'E2':1240,'E1a':978,'E1b':6846,'E1c':96}
DESCRIPTIONS = {'E2':'两轴共60批，同槽相位及初速扰动评测',
                'E1a':'三案例同槽 CEM 试验和阳性对照',
                'E1b':'另21个案例的同槽 CEM 主实验','E1c':'64条单脊偏移与32条原地形对照'}
def read(p):
    return json.loads(p.read_text())
def save(p,data):
    t=p.with_suffix('.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def status(stage, state, **kwargs):
    save(STATE/(stage+'.json'),dict(stage=state,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs))
def bark(stage, event, msg):
    receipt=STATE/f'{stage}_{event}_receipt.json'
    if receipt.exists() and read(receipt)['status']=='sent':
        return
    subprocess.run(['/usr/bin/python3',str(REPO/'learning/scripts/notify_sttw_job.py'),'--message',msg,'--receipt',str(receipt)],check=True)
    if read(receipt)['status']!='sent':
        raise RuntimeError(f'Bark 未返回成功，见 {receipt}')
def folder(stage):
    return ROOT/'e2' if stage=='E2' else ROOT/'e1'/stage
def progress(stage):
    p=folder(stage)/'status.json'
    if not p.exists():
        return '等待调度'
    d=read(p)
    keep=('stage','active','active_batch','generation','completed_new','completed_total','completed','candidate')
    return ', '.join(f'{k}={d[k]}' for k in keep if k in d)
def start(stage):
    pin=Path(read(STATE/(stage+'_pin.json'))['worktree'])
    frozen=read(ROOT/'paired_authorization.json')
    if frozen['decision']!='matched_packing' or sum(ALLOCATION.values())+72>frozen['main_experiment_rollout_cap']:
        raise ValueError('运行决议或 9,500 主预算不符')
    deadline=(datetime.now(timezone.utc)+timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S UTC')
    prefix=['systemd-run','--user','--quiet',f'--unit=pcrp-matched-{stage.lower()}-wait30',
            '--on-calendar='+deadline,'--timer-property=AccuracySec=1s','--timer-property=Persistent=true']
    subprocess.run([*prefix,'/usr/bin/python3',str(Path(__file__).resolve()),stage,'overdue'],check=True)
    env=dict(os.environ,PYTHONPATH=str(pin/'learning/src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def cli(script,*args):
        subprocess.run([PYTHON,'-u',str(pin/'learning/cli'/script),*map(str,args)],cwd=pin,env=env,check=True)
    dest=folder(stage)
    status(stage,'running',reserved=ALLOCATION[stage],repo=str(pin))
    msg=f'PCRP {stage} {DESCRIPTIONS[stage]}开始，预登记 {ALLOCATION[stage]} 条；各物理批使用共享 GPU 队列。'
    record(stage+'同槽阶段启动',msg+' 目录 '+str(dest))
    bark(stage,'started',msg)
    if stage=='E2':
        cli('pcrp_phase_grid.py','--run',ROOT,'--matched','--execute')
        if read(dest/'status.json')['completed_new']!=ALLOCATION[stage]:
            raise RuntimeError('E2 实际物理数与预登记不符')
        cli('analyze_pcrp_e2.py','--run',ROOT)
        result=read(dest/'analysis/summary.json')
        msg=f"E2 完成。1,160 个主工况，1,240 条物理回放。主判定 {result['decisions']['nominal_centers']}。"
        report=dest/'analysis/README.md'
        if result.get('flat_control_needs_review'):
            warning='E2 平地阴性对照出现结局翻转；需核查配对和稳定性，保留全部原始轨迹。'
            record('E2平地阴性对照警报',warning+' '+str(report))
            bark(stage,'flat_review',warning+' 报告 '+str(report))
    elif stage in ('E1a','E1b'):
        cli('pcrp_oracle_search.py','--run',ROOT,'--stage',stage,'--execute')
        result=read(dest/'status.json')
        if result['stage']!='complete' or result['completed_new']!=ALLOCATION[stage]:
            raise RuntimeError('E1 实际搜索或预算与预登记不符')
        if not result['flat_control_passed']:
            bark(stage,'flat_failure',f'PCRP {stage} 平地阳性对照未通过。后续依赖阶段停止，保留全部回合并生成失败报告。')
        cli('analyze_pcrp_e1.py',dest)
        report=dest/'analysis/README.md'
        if not result['flat_control_passed']:
            raise RuntimeError(f'平地阳性对照失败，报告 {report}')
        msg=f"{stage} 完成 {result['completed_new']} 条，平地阳性对照通过。"
    else:
        cli('pcrp_oracle_robustness.py','--run',ROOT,'--confirmed-protocol',ROOT/'e1c_scope.json')
        result=read(dest/'status.json')
        if result['stage']!='complete' or result.get('completed_physical',result.get('completed'))!=96:
            raise RuntimeError('E1c 实际物理数与96条预登记不符')
        report=dest/'README.md'
        if result.get('decision')=='nominal_replay_mismatch':
            d=read(dest/'decision.json')
            text=f"PCRP E1c 同槽名义对照仅 {d['matched_nominal_controls']}/32 通过。地形稳健性与路线判断须结合差异复核。"
            record('E1c名义对照差异',text+' '+str(report))
            bark(stage,'nominal_mismatch',text+' 报告 '+str(report))
        msg='E1c 已完成 64 条真实脊位扰动及 32 条原地形对照。'
    if not report.is_file():
        raise FileNotFoundError(report)
    status(stage,'complete',physical_rollouts=ALLOCATION[stage],report=str(report))
    record(stage+'同槽阶段完成',msg+' '+str(report))
    bark(stage,'complete',f'PCRP {msg} 报告 {report}')
def stopped(stage):
    p=STATE/(stage+'.json')
    state=read(p) if p.exists() else {}
    result=os.environ.get('SERVICE_RESULT','unknown')
    if result=='success' and state.get('stage')=='complete':
        return
    info=state.get('error') or f"systemd {result} {os.environ.get('EXIT_STATUS','')}"
    msg=f'PCRP {stage} 出现问题：{info}。当前 {progress(stage)}。已保留日志和回放，依赖阶段停止。'
    status(stage,'error',error=info,service_result=result)
    bark(stage,'error',msg)
    record(stage+'同槽阶段异常',msg)
if __name__=='__main__':
    stage,action=sys.argv[1:]
    if stage not in ALLOCATION:
        raise ValueError(stage)
    if action=='run':
        try:
            start(stage)
        except (Exception,SystemExit) as exc:
            status(stage,'error',error=f'{type(exc).__name__}: {exc}');raise
    elif action=='overdue':
        path=STATE/(stage+'.json')
        if path.is_file() and read(path)['stage'] not in ('complete','error'):
            bark(stage,'wait30',f'PCRP {stage} 自进入队列已经半小时，当前 {progress(stage)}；按约通知。')
    elif action=='stopped':
        stopped(stage)
    else:
        raise ValueError(action)
