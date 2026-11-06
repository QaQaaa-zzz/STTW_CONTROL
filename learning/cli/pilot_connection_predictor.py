"""Small whole-origin DEV comparison; never opens the independent TEST split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sttw_control.connection_predictor import fit_predictor
from sttw_control.empirical_capability import ConnectionCapability, select_connection_estimate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--records', type=Path, required=True)
p.add_argument('--config', type=Path, default=Path('learning/configs/jump_connection_capability_dev.json'))
p.add_argument('--output', type=Path, required=True)
p.add_argument('--brier-tiebreak', action='store_true', help='Use tuning Brier only when actual chosen completion/time tie')
a = p.parse_args()
records = [json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
config = json.loads(a.config.read_text())
candidates = config['candidates']
ids = [f"v{c['speed_m_s']}_a{c['alpha']}" for c in candidates]
groups = {}
excluded = []
for origin in sorted({r['origin_id'] for r in records}):
    rows = [r for r in records if r['origin_id'] == origin]
    by_id = {f"v{r['candidate']['speed_m_s']}_a{r['candidate']['alpha']}":r for r in rows}
    if len(rows) != len(ids) or set(by_id) != set(ids):
        raise ValueError('Exactly six observed candidate outcomes required per origin')
    if not all(r['supervised_input_complete'] and (r['decision_prefix'] or {}).get('corridor_intact_at_decision') is True for r in rows):
        excluded.append(dict(origin_id=origin,reason='missing observable input or invalid decision prefix',observed_rows=len(rows)))
    else:
        groups[origin] = [by_id[c] for c in ids]
if len(groups) < 5:
    raise ValueError('Pilot needs at least five complete origins for whole-root train/tune/DEV separation')
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
origins = sorted(groups)
results = []

def inputs(rows):
    return np.array([r['state'] + [r['candidate']['speed_m_s'], r['candidate']['alpha']] for r in rows], dtype=np.float32)

def score(estimates, rows):
    choice = select_connection_estimate(estimates, config['minimum_completion'], config['probability_tolerance'],
                                        previous_candidate='v2.0_a0.5', time_tolerance=config['time_tolerance_s'])
    selected = choice['candidate_id'] if choice else 'v2.0_a0.5'
    actual = rows[ids.index(selected)]
    return dict(candidate_id=selected,complete=actual['Y'],time_s=actual['T_total_s'],
                fallback=choice is None,estimates=estimates,source_case_id=actual['case_id'],
                brier=float(np.mean([(e['complete_probability']-r['Y'])**2 for e,r in zip(estimates,rows) if e['complete_probability'] is not None])))

for fold, held in enumerate(origins):
    tuning = origins[(fold + 1) % len(origins)]
    train_origins = [o for o in origins if o not in (held, tuning)]
    train = [r for o in train_origins for r in groups[o]]
    kernel_rows = [dict(state=r['state'],task_id='hold',skill_id='frozen',candidate_id=ids[groups[r['origin_id']].index(r)],
                        complete=r['Y'],success_cost=r['T_after_decision_s']) for r in train]
    kernel = ConnectionCapability(kernel_rows,config['feature_scales'],config['bandwidth'],config['minimum_support'])

    def predict(model, mean, scale, time_mean, time_scale, rows):
        with torch.no_grad():
            output = model((torch.tensor(inputs(rows)) - mean) / scale)
            probability = output[:,0].sigmoid().tolist()
            times = (output[:,1] * time_scale + time_mean).clamp_min(0.).tolist()
        support = kernel.query(rows[0]['state'],'hold','frozen',ids)['estimates']
        estimates = [dict(e,complete_probability=probability[i],success_cost=times[i]) for i,e in enumerate(support)]
        return score(estimates, rows)

    model,best,history = fit_predictor(inputs(train),[r['Y'] for r in train],
        [r['T_after_decision_s'] if r['Y'] else 0. for r in train],
        lambda model,mean,scale,tm,ts:predict(model,mean,scale,tm,ts,groups[tuning]),seed=20261106+fold,brier_tiebreak=a.brier_tiebreak)
    mlp = predict(model,best['input_mean'],best['input_scale'],best['time_mean'],best['time_scale'],groups[held])
    baseline = score(kernel.query(groups[held][0]['state'],'hold','frozen',ids)['estimates'],groups[held])
    fixed_id = min(range(len(ids)),key=lambda i:(-groups[tuning][i]['Y'],groups[tuning][i]['T_after_decision_s'] if groups[tuning][i]['Y'] else float('inf'),i))
    fixed = groups[held][fixed_id]
    result = dict(held_out_DEV_origin=held,tuning_origin=tuning,train_origins=train_origins,
        epochs=len(history),selected_epoch=best['epoch'],effective_batch=len(train),
        mlp=mlp,kernel=baseline,validation_selected_fixed=dict(candidate_id=ids[fixed_id],complete=fixed['Y'],time_s=fixed['T_total_s']))
    results.append(result)
    torch.save(dict(best,input_dim=inputs(train).shape[1],input_schema='rb_jump_decision_features_v2 plus speed_m_s,alpha',
                    train_origins=train_origins,tuning_origin=tuning,held_out_DEV_origin=held),a.output/f'fold{fold}.pt')
    (a.output/f'fold{fold}_training.json').write_text(json.dumps(history,indent=2)+'\n')

summary = {}
for method in ('mlp','kernel','validation_selected_fixed'):
    rows = [r[method] for r in results]
    selected = [r for r in rows if not r.get('fallback',False)]
    summary[method] = dict(complete=sum(r['complete'] for r in rows),roots=len(rows),
        explicit_choices=len(selected),false_clearances=sum(not r['complete'] for r in selected))
report = dict(role='Five-root nested pilot DEV; small sample, uncalibrated probabilities, no deployment adoption',
    records=str(a.records.resolve()),config=config,excluded_origins=excluded,summary=summary,folds=results,
    brier_tiebreak=a.brier_tiebreak, optimizer_steps=sum(r['epochs'] for r in results),new_physics_steps=0,device='cpu',
    limits='Three train roots, one tuning root, one held-out DEV root per fold. Root-balanced full batch; success-only standardized Huber plus all-row BCE. Early stopping uses tuning selected-action outcome. Kernel uses identical observable fields and candidate set. Both retain actual 2.0/.5 when support/risk thresholds reject choices. Missing-input origin retained in exclusion inventory. No probability calibration or independent TEST; estimates are not guarantees.')
(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
