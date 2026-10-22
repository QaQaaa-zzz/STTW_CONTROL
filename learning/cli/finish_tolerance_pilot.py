#!/usr/bin/env python3
"""One-shot postprocessing continuation for a single declared pilot run."""
import argparse,json,time,subprocess,sys,fcntl,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run).resolve()
def status(phase,**kw):
 temp=root/'analysis_status.tmp';temp.write_text(json.dumps(dict(phase=phase,checked_unix=time.time(),**kw),indent=2));temp.replace(root/'analysis_status.json')
status('waiting_for_execution')
while True:
 execution=json.loads((root/'status.json').read_text())
 if execution['phase'] in ('completed','error','failed'):break
 time.sleep(20)
if execution['phase']!='completed':
 status('blocked_by_execution',execution_status=execution['phase']);sys.exit(1)
try:
 status('running')
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_tolerance_pilot.py')),str(root)],check=True)
 summary=json.loads((root/'summary.json').read_text());assert summary['cases']==42 and (root/'xy_all_cases.png').stat().st_size>0
 status('completed',report=str(root/'REPORT.md'),xy=str(root/'xy_all_cases.png'))
 ledger=Path('/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md')
 entry=f'''### 2026-10-22 容扰度首轮脉冲面板自动回收\n\nsource_type=repo_artifact；verification=reproduced（真实MJX面板及自动分析，非独立容扰度预测验证）。执行目录 `{root}`，42/42轨迹完成，成功{summary['success_count']}/42（包含6条零外力重复，不是42个独立样本）；physical_failures={summary['physical_failures']}，working_failures={summary['working_failures']}，domain_failures={summary['domain_failures']}，hold_failures={summary['hold_failures']}。实际面板lower ticks={execution['active_lower_ticks']}，另本次preflight8ticks；先前错误成本1400+8ticks保留。结果见REPORT.md、summary.json、逐轨迹NPZ和xy_all_cases.png；图像已生成，尚待人工/agent视觉审阅。无训练、无新模型采用。单一路径/速度与有限幅值网格不认证连续容扰边界；E_b/gamma/完整闭环灵敏度尚待标定，未证明动态指标优势。下一步先检查名义失败与方向结果，再登记模型/代理标定及留出验证协议。\n\n'''
 with open(ledger.parent/'.write.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);s=ledger.read_text().replace('## 2. STTW_CONTROL\n','## 2. STTW_CONTROL\n\n'+entry,1);s=re.sub(r'state_revision: (\d+)',lambda m:'state_revision: '+str(int(m[1])+1),s,count=1);ledger.write_text(s)
except BaseException as e:
 status('error',error=f'{type(e).__name__}: {e}');raise
