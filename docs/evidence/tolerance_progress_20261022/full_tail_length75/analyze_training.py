"""Read-only audit of the completed fixed-length continuation; no model selection."""
import argparse,hashlib,json,statistics
from pathlib import Path

def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=a.root/'runs/preference_full_tail_v1/uniform/metrics.jsonl'
    new=a.root/'runs/preference_full_tail_length75_v1/uniform/metrics.jsonl'
    status=json.loads(new.with_name('status.json').read_text())
    if status['phase']!='completed' or status['completed_updates']!=77:raise ValueError('Training must be complete')
    before,after=read(old),read(new)
    assert [r['update'] for r in before]==list(range(1,28))
    assert [r['update'] for r in after]==list(range(28,78))
    assert status['executed_policy_transitions']==50*512*128
    allrows=before+after
    for r in allrows:
        assert sum(r['pulse']['force_histogram'])==r['pulse']['fired']
    windows={}
    for name,lo,hi in [('ppo1_5',3,7),('ppo21_25',23,27),('ppo26_30',28,32),('ppo46_50',48,52),('ppo71_75',73,77)]:
        rr=[r for r in allrows if lo<=r['update']<=hi];assert len(rr)==5
        windows[name]={k:statistics.mean(r[k] for r in rr) for k in ['mean_step_reward','domain_failures','physical_failures']}
        windows[name].update(parent_kl=statistics.mean(r['optimizer']['parent_kl'] for r in rr),value_loss=statistics.mean(r['optimizer']['value'] for r in rr),pulse_fired=sum(r['pulse']['fired'] for r in rr))
    out=dict(source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (old,new)},new_updates=50,new_policy_transitions=status['executed_policy_transitions'],windows=windows,new_pulses=sum(r['pulse']['fired'] for r in after),hard_kl_stops=sum(bool(r['optimizer']['hard_kl_stop']) for r in after),nonfinite_stops=sum(bool(r['optimizer']['nonfinite_stop']) for r in after),sample_seconds=sum(r['sample_seconds'] for r in after),optimize_seconds=sum(r['optimize_seconds'] for r in after))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'training_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# 25→75 PPO：完整训练过程审计','','固定末端对照；训练曲线仅作描述，不能替代固定评测。续训重置完整仿真状态，继承优化器/RNG等边界状态，不是逐帧连续。','','| PPO窗口 | 平均每步奖励 | 平均域退出 | 施扰事件 | 父策略KL |','|---|---:|---:|---:|---:|']
    for k,v in windows.items():lines.append(f"| {k} | {v['mean_step_reward']:.4f} | {v['domain_failures']:.2f} | {v['pulse_fired']} | {v['parent_kl']:.5f} |")
    lines+=['',f"新增50更新、{out['new_policy_transitions']:,}策略转移、{out['new_pulses']}次施扰事件；不是独立初态数。hard-KL停止{out['hard_kl_stops']}，非有限停止{out['nonfinite_stops']}。",'', '各窗口来自变化策略的训练采样，无独立重复，不做收敛显著性结论。固定75 PPO对25 PPO的原任务成功、无扰保留和失败配对才决定加长训练的实际价值。采样耗时不包括编译和报告，总成本以运行声明/状态为准。']
    (a.output/'TRAINING_AUDIT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
