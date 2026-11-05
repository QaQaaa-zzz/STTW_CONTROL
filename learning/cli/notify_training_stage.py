"""One-shot desktop notification for an explicitly selected training process."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--pid',type=int,required=True)
a=p.parse_args()
receipt=a.run/'stage_notification.json'
if not receipt.exists():
    status=json.loads((a.run/'status.json').read_text())
    while status['stage'] not in ('training_complete','failed') and Path(f'/proc/{a.pid}').exists():
        time.sleep(5)
        status=json.loads((a.run/'status.json').read_text())
    complete=status['stage']=='training_complete'
    message=(f"训练完成 {status['completed_updates']} 轮，候选评测尚待完成" if complete
             else f"训练进程结束但未完成，已记录 {status['completed_updates']} 轮，请检查日志")
    result=subprocess.run(['notify-send','STTW 底层训练阶段通知',message],check=False)
    receipt.write_text(json.dumps(dict(stage=status['stage'],pid=a.pid,
        desktop_notification_exit_code=result.returncode,time=time.time()),indent=2))
