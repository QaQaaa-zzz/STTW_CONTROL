"""Hand E1 back to its original dependency chain after the fixed-slot physics replay."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
from dispatch_paired import ROOT, STATE, bark, read, record, save


def main(launch_path):
    launch = read(launch_path)
    folder = Path(launch['output'])
    verified = folder / 'summary.json'
    score = read(verified) if verified.exists() else {}
    stage = folder.name
    if not score.get('passed'):
        message = f'E1 常驻同槽复验 {stage} 未通过，原轨迹已保留，正式搜索等待复核。{folder}'
        record('E1常驻复验未通过 '+stage, message)
        bark(stage, 'error', message)
        return 1

    worktree = Path(launch['worktree'])
    head = subprocess.check_output(['git', '-C', worktree, 'rev-parse', 'HEAD'], text=True).strip()
    if head != launch['commit'] or not (score['one_process'] and len(score['batches']) == 3):
        raise ValueError('Frozen source or probe batch count differs from the replay registration')
    key = ROOT / 'resident_upgrade/planned_stop.json'
    if not key.is_file():
        raise RuntimeError('The original E1 search was not held at the registered candidate boundary')
    for stage_name in ('E1a', 'E1b'):
        pin = STATE / f'{stage_name}_pin.json'
        saved = ROOT / 'resident_upgrade' / f'{stage_name}_pre_resident_pin.json'
        if not saved.exists():
            save(saved, read(pin))
        save(pin, dict(commit=head, worktree=str(worktree), execution='resident_generation',
                       resume_existing=stage_name == 'E1a', verified_replay=str(verified)))
    units = [f'pcrp-matched-{name}.service' for name in ('e1a', 'e1b', 'e1c', 'final')]
    for unit in units:
        if subprocess.run(['systemctl','--user','is-failed','--quiet',unit]).returncode==0:
            subprocess.run(['systemctl','--user','reset-failed',unit],check=True)
    os.replace(key, key.with_name('handoff_complete.json'))
    subprocess.run(['systemctl','--user','start','--no-block',units[-1]], check=True)
    deadline=(datetime.now(timezone.utc)+timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S UTC')
    subprocess.run(['systemd-run','--user','--quiet','--unit=pcrp-resident-resumed-'+stage,
                    '--on-calendar='+deadline,'--timer-property=AccuracySec=1s',
                    '/usr/bin/python3',str(Path(__file__).resolve()),str(launch_path),'overdue'],check=True)
    msg = (f'E1 常驻实机复验通过：单进程3批共9条的轨迹与原档逐位一致。'
           f'已恢复E1a并排入E1b/E1c/最终分析。验证 {verified}')
    record('E1常驻通过并接续 '+stage, msg)
    bark(stage, 'complete', msg)
    return 0


if __name__ == '__main__':
    launch_path=Path(sys.argv[1])
    if sys.argv[2:]==['overdue']:
        state=read(ROOT/'e1/E1a/status.json')
        if state['stage']=='running':
            bark(Path(read(launch_path)['output']).name, 'resume_wait30',
                 f'E1a 恢复后已运行半小时；已完成 {state["completed_new"]}/978 条。后续按冻结预算继续并报告异常。')
    else:
        raise SystemExit(main(launch_path))
