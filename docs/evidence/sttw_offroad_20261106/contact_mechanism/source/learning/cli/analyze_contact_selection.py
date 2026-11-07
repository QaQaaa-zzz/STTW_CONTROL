"""Fixed-regularization, leave-parameter-seed-out recovery-alpha comparison."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
data=json.loads((a.run/'contact_branch_outcomes.json').read_text());rows=data['groups'];alphas=[0,.25,1]
base_keys=['speed_m_s','roll_rad','roll_rate_rad_s','entry_alpha','center_error_limit_m','front_vertical_load_N']
contact_keys=['normal_signed','normal_absolute','lateral_signed','lateral_absolute','negative_absolute','migration_signed','migration_absolute']
x0=np.array([[row['base_features'][k] for k in base_keys] for row in rows])
x0=np.c_[x0,[row['contact_features_oracle']['front_near_zero_vertical_load_s'] for row in rows]]
base_keys.append('front_near_zero_vertical_load_s')
x1=np.c_[x0,[[row['contact_features_oracle']['impulse_Nms'][k] for k in contact_keys] for row in rows]]
seeds=np.array([row['seed'] for row in rows]);targets=[]
for row in rows:
    by_alpha={o['alpha']:o for o in row['outcomes'] if not o['unreset_source']}
    targets.append([[by_alpha[alpha]['path_peak_abs_m'],by_alpha[alpha]['speed_rmse_m_s'],float(by_alpha[alpha]['full_remaining_window_alive'])] for alpha in alphas])
y=np.array(targets,dtype=float)
if not np.isfinite(y).all():raise ValueError('Missing branch observations must be resolved or reported separately, not imputed as success')
reports={};ridge=1.
for name,x in [('base',x0),('base_plus_contact_oracle',x1)]:
    predictions=np.empty_like(y);folds=[]
    for seed in np.unique(seeds):
        train=seeds!=seed;test=~train;mean=x[train].mean(0);scale=x[train].std(0);scale=np.where(scale>1e-12,scale,1.)
        xt=np.c_[np.ones(train.sum()),(x[train]-mean)/scale];xv=np.c_[np.ones(test.sum()),(x[test]-mean)/scale]
        penalty=np.eye(xt.shape[1])*ridge;penalty[0,0]=0
        coefficient=np.linalg.solve(xt.T@xt+penalty,xt.T@y[train].reshape(train.sum(),-1))
        predictions[test]=(xv@coefficient).reshape(test.sum(),3,3)
        folds.append(dict(heldout_seed=int(seed),training_groups=int(train.sum()),test_groups=int(test.sum()),training_mean=mean.tolist(),training_scale=scale.tolist(),coefficient=coefficient.tolist()))
    predictions[:,:,:2]=np.maximum(predictions[:,:,:2],0);predictions[:,:,2]=predictions[:,:,2].clip(0,1)
    selections=[]
    for i,row in enumerate(rows):
        pred=predictions[i];width=row['base_features']['center_error_limit_m'];feasible=(pred[:,0]<=width)&(pred[:,2]>=.5)
        candidates=np.flatnonzero(feasible)
        chosen=int(candidates[np.argmin(pred[candidates,1])]) if len(candidates) else min(range(3),key=lambda j:(pred[j,2]<.5,max(0,pred[j,0]-width),pred[j,1]))
        actual=y[i,chosen];oracle=[o for o in row['outcomes'] if not o['unreset_source'] and o['feasible_center_path']]
        best_speed=min(o['speed_rmse_m_s'] for o in oracle) if oracle else None
        actual_feasible=bool(actual[2]>=.5 and actual[0]<=width)
        selections.append(dict(group=row['group'],seed=row['seed'],speed_m_s=row['speed_m_s'],entry_alpha=row['entry_alpha'],chosen_alpha=alphas[chosen],predicted_empty_set=not bool(len(candidates)),actual_empty_set=row['empty_feasible_set'],actual_feasible=actual_feasible,actual_survived=bool(actual[2]>=.5),actual_peak_path_m=float(actual[0]),actual_path_excess_m=float(max(0,actual[0]-width)),actual_speed_rmse_m_s=float(actual[1]),speed_regret_m_s=float(actual[1]-best_speed) if actual_feasible and best_speed is not None else None,predictions=pred.tolist()))
    reports[name]=dict(folds=folds,selections=selections,summary=dict(groups=len(rows),actual_feasible=sum(r['actual_feasible'] for r in selections),actual_survived=sum(r['actual_survived'] for r in selections),actual_empty_sets=sum(r['actual_empty_set'] for r in selections),mean_path_excess_m=float(np.mean([r['actual_path_excess_m'] for r in selections])),mean_speed_rmse_m_s=float(np.mean([r['actual_speed_rmse_m_s'] for r in selections]))))
result=dict(role='DEV offline surrogate, entire parameter seeds held out; not unseen-terrain evaluation or deployed closed-loop selection; contact features are simulator-truth oracle',protocol=dict(alphas=alphas,base_features=base_keys,added_contact_features=contact_keys,ridge_regularization=ridge,normalization='training fold only; intercept unpenalized; same folds and regularization for both models',targets=['peak center-path error','requested-speed RMSE','remaining-window survival'],physical_prediction_bounds='errors clipped at0 and survival score clipped0to1; no tuning',selection='predicted feasible set then speed; preserve empty set flag and evaluate fallback violations',limitations='30 possible contexts and5 fixed parameter seeds; no hyperparameter search or new policy training'),models=reports)
(a.run/'contact_selection_heldout.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['summary'] for k,v in reports.items()}))
