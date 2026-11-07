#!/usr/bin/env python3
"""Wait for one chapter-three queue process and send one Bark terminal notice."""
import argparse
import datetime
import json
import os
from pathlib import Path
import select
import subprocess
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument('--pid', type=int, required=True)
parser.add_argument('--unit', default='ch3-speed-resume.service')
parser.add_argument('--status', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
fd = os.pidfd_open(args.pid)
select.select([fd], [], [])
os.close(fd)
state = json.loads(args.status.read_text())
unit = subprocess.run(['systemctl', '--user', 'show', args.unit, '-p', 'ExecMainStatus', '-p', 'Result'], capture_output=True, text=True).stdout
fields = dict(line.split('=', 1) for line in unit.splitlines() if '=' in line)
finished = state.get('status') in ('completed_analysis_pending', 'complete', 'completed_slice') and fields.get('Result') == 'success'
when = datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
stage = state.get('stage', 'unknown')
title = '第三章仿真完成：可以继续推进任务' if finished else '第三章仿真异常：可以继续推进任务'
body = f'时间：{when}\n状态：{state.get("status")}\n阶段：{stage}\n队列状态：{fields.get("Result")}，退出码 {fields.get("ExecMainStatus")}\n进度文件：{args.status}'
body += '\n可以继续推进任务。请在对话里回复“继续”，让我收取结果并推进下一步。' if finished else '\n可以继续推进任务。请在对话里回复“继续”，让我先处理运行异常。'
config_path = Path('/home/qy/.config/jit-email/bark.json')
if config_path.stat().st_mode & 0o077:
    raise PermissionError('Bark config permissions must remain private')
config = json.loads(config_path.read_text())
payload = json.dumps(dict(device_key=config['device_key'], title=title, body=body, group='STTW 第三章', level='active'), ensure_ascii=False).encode()
request = Request(config['server'].rstrip('/') + '/push', data=payload, headers={'Content-Type': 'application/json; charset=utf-8'}, method='POST')
with urlopen(request, timeout=15) as response:
    result = json.loads(response.read())
if response.status != 200 or result.get('code') != 200:
    raise RuntimeError('Bark rejected the terminal notice')
args.receipt.write_text(json.dumps(dict(status='sent', title=title, time=when, response_code=response.status), ensure_ascii=False, indent=2) + '\n')
