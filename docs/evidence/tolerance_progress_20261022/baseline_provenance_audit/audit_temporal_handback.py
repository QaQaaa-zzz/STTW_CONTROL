"""Read the already executed prefix handback, without advancing any state."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np


def main():
    worktree=Path(__file__).resolve().parents[4]
    run=worktree/'runs/temporal_reference_opportunity_v1_attempt2'
    execution=run/'execution'
    out=worktree/'runs/temporal_handback_audit_v1';out.mkdir(exist_ok=False)
    manifest=json.loads((execution/'manifest.json').read_text())
    receipt=json.loads((execution/'handback_receipt.json').read_text())
    plan=manifest['plan'];rows=plan['cases']
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert len(rows)==64 and receipt['intended_ticks']==88
    for source,sha in plan['source_sha256'].items():assert digest(source)==sha
    with (execution/'handback_states.pkl').open('rb') as f:state=pickle.load(f)
    filtered=np.asarray(state.filtered);offsets=np.asarray(state.offsets)
    assert filtered.shape==offsets.shape==(64,2)
    targets=np.asarray(manifest['realized_targets'])
    a=np.exp(-.2/np.asarray(manifest['config']['residual_interface']['filter_tau_s']))
    lookup={(row['route'],row['candidate']):i for i,row in enumerate(rows)}
    pairs=[]
    for i,row in enumerate(rows):
        if row['family']!='temporal':continue
        j=lookup[row['route'],row['mean_candidate']]
        prediction=(1-a)*(a*(targets[i,0]-targets[j,0])+(targets[i,1]-targets[j,0]))
        observed=filtered[i]-filtered[j]
        pairs.append(dict(route=row['route'],temporal=row['candidate'],constant=row['mean_candidate'],
                          normal_prefix_both=bool(receipt['reached_without_stop'][i] and receipt['reached_without_stop'][j]),
                          filtered_gap=observed.tolist(),offset_gap=(offsets[i]-offsets[j]).tolist(),
                          no_backcalculation_prediction=prediction.tolist(),
                          conditional_prediction_residual=(observed-prediction).tolist()))
    assert len(pairs)==32
    summary=[]
    for route in sorted({p['route'] for p in pairs}):
        selected=[p for p in pairs if p['route']==route]
        summary.append(dict(route=route,logical_pairs=len(selected),
                            normal_prefix_pairs=sum(p['normal_prefix_both'] for p in selected),
                            max_abs_filtered_gap=np.max(np.abs([p['filtered_gap'] for p in selected]),axis=0).tolist(),
                            max_abs_offset_gap=np.max(np.abs([p['offset_gap'] for p in selected]),axis=0).tolist(),
                            max_abs_conditional_residual=np.max(np.abs([p['conditional_prediction_residual'] for p in selected]),axis=0).tolist()))
    sources=[Path(__file__),execution/'manifest.json',execution/'handback_receipt.json',execution/'handback_states.pkl']
    result=dict(prefixes_reached_without_stop=sum(receipt['reached_without_stop']),prefixes=64,
                pairs=pairs,summary=summary,sources={str(p):digest(p) for p in sources},
                extra_physics_ticks=0,training_steps=0,model_fits=0,
                scope='Executed .44s prefix only, not final safety/task outcomes. Conditional filter identity may fail under absolute-reference back-calculation or unequal start-filter states; full clip traces pending. Other vehicle/controller/history states also differ. No reset or causal mediation claim.')
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
