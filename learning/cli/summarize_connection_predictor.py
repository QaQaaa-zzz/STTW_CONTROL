"""Summarize the saved supervised DEV pilot without refitting or simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--records', type=Path, help='Optional archived records path')
a = p.parse_args()
report = json.loads((a.run / 'results.json').read_text())
records = [json.loads(s) for s in (a.records or Path(report['records'])).read_text().splitlines() if s.strip()]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
diagnostics = []
for i, fold in enumerate(report['folds']):
    history = json.loads((a.run / f'fold{i}_training.json').read_text())
    epochs = [h['epoch'] for h in history]
    for ax, field in zip(axes[:2], ('classification_loss', 'time_loss')):
        ax.plot(epochs, [h[field] for h in history], label=f'Fold {i+1}')
        ax.axvline(fold['selected_epoch'], color='gray', alpha=.15)
    observed = {f"v{r['candidate']['speed_m_s']}_a{r['candidate']['alpha']}": r for r in records if r['origin_id'] == fold['held_out_DEV_origin']}
    metrics = {}
    for method in ('mlp', 'kernel'):
        estimates = fold[method]['estimates']
        known = [e for e in estimates if e['known'] and e['complete_probability'] is not None]
        errors = [(e['complete_probability'] - observed[e['candidate_id']]['Y'])**2 for e in known]
        times = [abs(e['success_cost'] - observed[e['candidate_id']]['T_after_decision_s']) for e in known if observed[e['candidate_id']]['Y'] and e['success_cost'] is not None]
        metrics[method] = dict(known_candidates=len(known), brier=float(np.mean(errors)) if errors else None,
                              success_time_mae_s=float(np.mean(times)) if times else None)
    probabilities = [e['complete_probability'] for e in fold['mlp']['estimates']]
    axes[2].scatter([i+1]*len(probabilities), probabilities, marker='x', label=f'Fold {i+1}')
    diagnostics.append(dict(origin=fold['held_out_DEV_origin'], selected_epoch=fold['selected_epoch'],
                            epochs=fold['epochs'], metrics=metrics,
                            tuning_selected_candidates=sorted({h['validation']['candidate_id'] for h in history}),
                            max_selected_model_probability=max(probabilities)))
axes[0].set(title='Training classification loss', xlabel='Epoch', ylabel='BCE')
axes[1].set(title='Training conditional-time loss', xlabel='Epoch', ylabel='Standardized Huber')
axes[2].axhline(report['config']['minimum_completion'], color='black', linestyle='--', label='Release threshold')
axes[2].set(title='Selected model: held-out DEV probabilities', xlabel='Fold (one held-out origin)', ylabel='Uncalibrated probability', ylim=(0, 1), xticks=range(1,len(diagnostics)+1))
axes[0].legend(fontsize=8)
axes[2].legend(fontsize=7)
fig.suptitle('Five-root supervised pilot; no independent TEST or controller training')
fig.savefig(a.run / 'training_diagnostics.png', dpi=160)
plt.close(fig)
(a.run / 'diagnostics.json').write_text(json.dumps(dict(folds=diagnostics, note='Brier is descriptive on held-out DEV; 6 candidates per root are dependent. Losses are pre-update; selected epoch is post-update. No calibration claim.'), indent=2)+'\n')
print(json.dumps(diagnostics, indent=2))

# Fixed bins describe held-out predictions; they do not fit a calibration map.
quality = {}
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.plot([0, 1], [0, 1], ':', color='gray', label='Identity')
for method, marker in (('mlp', 'o'), ('kernel', 's')):
    predictions = []
    selected_outcomes = []
    for fold in report['folds']:
        actual = {f"v{r['candidate']['speed_m_s']}_a{r['candidate']['alpha']}": r for r in records if r['origin_id'] == fold['held_out_DEV_origin']}
        for e in fold[method]['estimates']:
            if e['known'] and e['complete_probability'] is not None:
                predictions.append((e['complete_probability'], actual[e['candidate_id']]['Y'], fold['held_out_DEV_origin']))
        choice = fold[method]
        selected_outcomes.append(dict(origin=fold['held_out_DEV_origin'], fallback=choice['fallback'],
                                      candidate_id=choice['candidate_id'], complete=choice['complete'],
                                      checks=actual[choice['candidate_id']]['checks']))
    bins = []
    for left, right in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        rows = [r for r in predictions if left <= r[0] and (r[0] < right or right == 1.)]
        bins.append(dict(left=float(left), right=float(right), branches=len(rows),
                         roots=len({r[2] for r in rows}),
                         mean_probability=float(np.mean([r[0] for r in rows])) if rows else None,
                         observed_fraction=float(np.mean([r[1] for r in rows])) if rows else None))
    populated = [b for b in bins if b['branches']]
    ax.scatter([b['mean_probability'] for b in populated], [b['observed_fraction'] for b in populated], marker=marker, label=method)
    for b in populated:
        ax.annotate(str(b['branches']), (b['mean_probability'], b['observed_fraction']), xytext=(3, 5), textcoords='offset points', fontsize=8)
    common = [f for f in report['folds'] if f[method]['complete'] and f['validation_selected_fixed']['complete']]
    quality[method] = dict(known_branches=len(predictions), total_branches=6*len(report['folds']),
                          brier=float(np.mean([(p-y)**2 for p,y,_ in predictions])), reliability_bins=bins,
                          selected_outcomes=selected_outcomes,
                          common_success_origins=[f['held_out_DEV_origin'] for f in common],
                          time_difference_vs_fixed_s=[f[method]['time_s']-f['validation_selected_fixed']['time_s'] for f in common])
ax.set(xlabel='Mean uncalibrated predicted probability', ylabel='Observed completion fraction',
       xlim=(-.02,1.02), ylim=(-.02,1.08), title='DEV reliability; labels = branch counts')
ax.legend()
fig.savefig(a.run/'prediction_reliability.png', dpi=160)
plt.close(fig)
(a.run/'prediction_quality.json').write_text(json.dumps(dict(methods=quality,
    note='Five held-out DEV roots with six dependent branches each; folds share training data. Fixed bins [0,.2,.4,.6,.8,1], final bin includes 1. No fitted calibration, confidence intervals, or independent TEST claim. Positive paired time difference means slower than validation-selected fixed choice.'), indent=2)+'\n')
