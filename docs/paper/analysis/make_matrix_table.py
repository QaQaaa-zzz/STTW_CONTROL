"""Build the compact full-matrix manuscript table from saved analysis summaries."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h75', type=Path, required=True)
    parser.add_argument('--u170', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    h75 = json.loads((args.h75 / 'summary.json').read_text())
    u170 = json.loads((args.u170 / 'summary.json').read_text())
    if len(h75) != 24 or len(u170) != 24:
        parser.error('both complete three-speed summaries are required')
    columns = [('PP', h75), ('a0', h75), ('a0.5', h75), ('a1', h75),
               ('a0', u170), ('a0.5', u170), ('a1', u170)]
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}',
             r'\caption{Complete matched route-matrix summary. Each cell gives original-goal entries (out of 12); mean measured bend speed (m/s) / mean original-path bend MAE (m). Each speed--event cell comprises six routes and two complete initial states; no model is trained in this evaluation. All 504 physical trajectories and per-case metrics are retained.}',
             r'\label{tab:matrix}',
             r'\begin{tabular}{@{}llccccccc@{}}', r'\toprule',
             r'Speed & Force & PP & H75 $\alpha=0$ & H75 $\alpha=.5$ & H75 $\alpha=1$ & U170 $\alpha=0$ & U170 $\alpha=.5$ & U170 $\alpha=1$\\',
             r'\midrule']
    for speed, label in [('low', '1.90'), ('mid', '2.50'), ('high', '3.05')]:
        for event, event_label in [('zero', '0'), ('pulse', '$+36$ N')]:
            cells = []
            for method, source in columns:
                row = next(r for r in source if r['speed_group'] == speed and r['event'] == event and r['method'] == method)
                cells.append(f"{row['arrived']};{row['bend_mean_speed_m_s']:.3f}/{row['bend_original_cross_mae_m']:.3f}")
            lines.append(f'{label} & {event_label} & ' + ' & '.join(cells) + r'\\')
        if speed != 'high':
            lines.append(r'\addlinespace[2pt]')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    args.out.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
