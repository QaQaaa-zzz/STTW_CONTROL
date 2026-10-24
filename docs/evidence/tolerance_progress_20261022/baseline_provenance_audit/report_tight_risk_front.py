"""Posthoc risk-score discrimination audit, without choosing an operating threshold."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    rows = [r for r in data['rows'] if r['force_N'] != 0]
    if len(rows) != 64 or any(r['balance_label'] not in ('FLOOR_CONTACT', 'NO_FLOOR_CONTACT') for r in rows):
        raise ValueError('expected 64 known nonzero outcomes')
    labels = np.array([r['event_contact'] for r in rows], dtype=bool)
    if labels.sum() != 31:
        raise ValueError('contact denominator mismatch')
    names = [*rows[0]['scores'], 'analytic']
    report = {}
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for name in names:
        scores = np.array([r['scores'][name] if name != 'analytic' else abs(r['force_N']) - r['analytic_capacity_N'] for r in rows])
        if not np.all(np.isfinite(scores)):
            raise ValueError('nonfinite risk score')
        thresholds = np.r_[-np.inf, np.unique(scores), np.inf]
        predictions = scores[None, :] >= thresholds[:, None]
        misses = np.sum(~predictions & labels, axis=1)
        rejects = np.sum(predictions & ~labels, axis=1)
        default = scores >= .5 if name != 'analytic' else scores > 0
        dm = int(np.sum(~default & labels))
        dr = int(np.sum(default & ~labels))
        minimum = int(rejects[misses == 0].min())
        report[name] = dict(default_missed_contact=dm, default_conservative=dr,
                            posthoc_minimum_conservative_at_zero_misses=minimum,
                            curve=[dict(missed_contact=int(m), conservative=int(c)) for m, c in zip(misses, rejects)])
        line, = ax.plot(rejects, misses, drawstyle='steps-post', label=f'{name}; zero-miss min={minimum}')
        ax.scatter(dr, dm, color=line.get_color(), marker='x', s=65)
    ax.set(xlabel='Conservative rejects among 33 no-contact branches', ylabel='Missed contacts among 31 contact branches',
           title='Common 64 nonzero branches; x marks frozen default threshold')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    out = args.output
    out.mkdir(exist_ok=False)
    fig.savefig(out / 'risk_front.png')
    plt.close(fig)
    receipt = dict(source=str(args.analysis.resolve()), source_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
                   total=64, contacts=31, no_contact=33, models=report, new_physics=0, fits=0,
                   role='Posthoc diagnostic. No threshold selected or adopted. Correlated branches of eight states; not independent test trials.')
    (out / 'analysis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    lines = ['# 解析代理是否只是更保守：统一工作点诊断', '',
             '目的：避免把不同默认阈值造成的零漏报，误当成解析模型独有的预测能力。',
             '方法：复用同一8状态、64非零分支（31触地/33未触地），所有分数统一为越大风险越高。解析分数为|F|-预测方向容量；其他为固定模型概率。遍历已有分数工作点，不拟合、不采用新阈值。', '',
             '[漏报—保守误拒曲线](risk_front.png)', '',
             '|模型|默认漏报|默认保守误拒|事后零漏报所需最少误拒|', '|---|---:|---:|---:|']
    for name, r in report.items():
        lines.append(f"|{name}|{r['default_missed_contact']}|{r['default_conservative']}|{r['posthoc_minimum_conservative_at_zero_misses']}|")
    lines += ['', '判断：解析默认零漏报存在保守性代价；在该面板上尚未超过简单速度分数。事后零漏报工作点不能直接用于部署，也不证明未来零漏报。数据是相关的确定性DEV分支，不作独立样本置信推断。',
              '决定：不为这一结果继续扫解析参数或模型超参数；保持原阈值身份。保留解析坐标作解释和基线，是否进入控制须由同状态配对决策收益证明。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({name: r['posthoc_minimum_conservative_at_zero_misses'] for name, r in report.items()}))


if __name__ == '__main__':
    main()
