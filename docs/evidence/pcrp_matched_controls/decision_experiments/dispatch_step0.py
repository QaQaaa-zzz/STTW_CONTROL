"""Dispatch the frozen PCRP gate under the existing shared GPU lock."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
HUB = Path('/home/qy/STTW_CONTROL/research-hub')
UNIT = 'pcrp-decision-replay'
PREFIX = 'PCRP_DECISION_20261122'
REPORT = ROOT / 'step0' / 'README.md'

def record(event, detail):
    now = datetime.now(timezone.utc).isoformat()
    with (HUB / '.write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (HUB / 'AGENTS.md').read_text()
        path = HUB / 'STTW_PROJECT_STATE.md'
        state = path.read_text()
        key = f'**{PREFIX} / {event}**'
        if key in state:
            return
        revision = int(re.search(r'^state_revision: (\d+)$', state, re.M)[1]) + 1
        state = re.sub(r'^state_revision: \d+$', f'state_revision: {revision}', state, count=1, flags=re.M)
        state = re.sub(r'^updated_at_utc: .*$', f'updated_at_utc: {now}', state, count=1, flags=re.M)
        heading = '## 当前方案\n'
        if heading not in state:
            raise ValueError('Missing active-plan heading')
        state = state.replace(heading, heading + f'\n{key} {detail}\n', 1)
        with tempfile.NamedTemporaryFile(mode='w',dir=HUB,delete=False) as tmp:
            os.fchmod(tmp.fileno(), path.stat().st_mode & 0o777)
            tmp.write(state)
        os.replace(tmp.name, path)

def notice(event, message):
    subprocess.run(['/usr/bin/python3','learning/scripts/notify_sttw_job.py','--message',message,
                    '--receipt',str(ROOT / f'{event}_notice.json')],check=False)

def mark(filename, detail):
    (ROOT / filename).write_text(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
            detail=detail),ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    if sys.argv[1:] == ['register']:
        record('预登记', f'必做协议原件和 SHA、预算、数据角色见 {ROOT}/registration.json 及 step0/protocol.json；0 新训练、上限 9500 条。只调度第0步。sttw-gravel 38f5dc1c 基础，复用 H75/Local300、三地形两组，2 次独立进程各 36 条，复现原档绝对坐标。每一有效 200Hz XY 按 atol=1e-8 m、rtol=1e-5 比较，5kHz 终点与类型精确比较；108 对全部相符才通过。任一不符暂停 E1/E2。DEV 401–408；E1d/e 未调度。运行入口 {ROOT}/dispatch_step0.py，主机 qy-MS-7E06，用户服务 {UNIT} 共用 GPU 锁。7202 为独立在跑任务。')
    else:
        os.chdir('/home/qy/STTW_CONTROL/runs/worktrees/pcrp-replay-3d90f818')
        os.environ['PYTHONPATH'] = '/home/qy/STTW_CONTROL/runs/worktrees/pcrp-replay-3d90f818/learning/src'
        mark('step0_started.json','GPU lock acquired')
        try:
            record('第0步启动', f'GPU 锁已取得，开始 2×36 条冻结回放，运行目录 {ROOT}/step0。')
            rc = subprocess.run(['/home/qy/miniconda3/envs/mjp/bin/python','-u','learning/cli/pcrp_replay_gate.py',
                                 '--run',str(ROOT)]).returncode
            if rc:
                raise RuntimeError(f'replay exit code {rc}')
            result = json.loads((ROOT/'step0/status.json').read_text())
            if result['stage']!='complete' or not isinstance(result['passed'],bool) or not REPORT.is_file():
                raise RuntimeError('missing replay verdict or report')
            subprocess.run(['/home/qy/miniconda3/envs/mjp/bin/python','learning/cli/pcrp_replay_audit.py',
                            '--run',str(ROOT)],check=True)
            with REPORT.open('a') as report:
                report.write('\n[原档的批次差异、浮点分辨率与逐轨迹结果](packing_sensitivity/complete.md)\n')
            conclusion = '第0步通过，可进入 E2、E1a' if result['passed'] else '第0步未通过，按方案暂停 E1/E2，待决定重复设计'
            message = f'PCRP 判决实验 72 条回放完成。{conclusion}。逐条件 XY 和比较表见 {REPORT}'
            mark('terminal.json',message)
            record('第0步结果',message)
            notice('terminal',message)
        except Exception as exc:
            message = f'PCRP 判决实验第0步运行或交付错误：{exc}。请查看 {ROOT}/step0 和 journalctl --user -u {UNIT}'
            mark('terminal.json',message)
            notice('error',message)
            record('第0步错误',message)
            raise

        if result['passed']:
            from dispatch_e2 import submit_after_gate
            try:
                submit_after_gate()
            except Exception as exc:
                msg = f'PCRP 第0步通过，E2 提交错误：{exc}。请核查 pcrp-decision-phase.service。'
                notice('e2_schedule_error',msg)
                record('E2提交错误',msg)
                raise
