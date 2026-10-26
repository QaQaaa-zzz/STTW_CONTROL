"""Authenticate helper/origin provenance and describe near-terminal value errors."""
import json,pickle,time
from pathlib import Path
import numpy as np
from sttw_control.preference_best import digest,atomic_json
from sttw_control.live_training_signal_audit import discounted_returns


def main():
    out=Path('runs/live_training_signal_audit_v1').resolve();source=Path('runs/matched_live_ppo_v1').resolve()
    p=json.loads((source/'plan.json').read_text());decl=json.loads((out/'declaration.json').read_text())
    driver=Path('docs/evidence/tolerance_progress_20261022/live_training_signal/run_audit.py').resolve();assert digest(driver)==decl['analysis_code_sha256']
    helpers={}
    for name in ['live_training_signal_audit','live_task_credit','survival_endpoint_task_metrics','rejoin_comparison','anticipatory_analysis','preference_best']:
        file=Path('learning/src/sttw_control')/(name+'.py');sha=digest(file)
        if str(file.resolve()) in p['source_sha256']:assert sha==p['source_sha256'][str(file.resolve())]
        helpers[str(file.resolve())]=sha
    origins={}
    for context,c in p['contexts'].items():
        assert digest(c['initial'])==c['initial_sha256']
        with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
        xy=np.asarray(bank.physical.data.qpos[0,:2]).tolist()
        route=Path('runs/matched_live_ppo_v1_fixed64_evaluation')/context;index=route/'reentry/case_index.json'
        borrowed=json.loads(index.read_text())[0]['initial_xy'];np.testing.assert_allclose(xy,borrowed,rtol=0,atol=1e-6)
        origins[context]=dict(initial_state_file=c['initial'],initial_state_sha256=c['initial_sha256'],actual_xy=xy,
            borrowed_index=str(index.resolve()),borrowed_index_sha256=digest(index),borrowed_xy=borrowed)
    bins=[];schedule=json.loads((source/'training_cases.json').read_text())
    for receipt in json.loads((out/'raw_receipts.json').read_text()):
        n=receipt['collection'];d=source/'execution'/f'update_{n:04d}'
        for name,key in [('physical','physical_sha256'),('decisions','decisions_sha256')]:assert digest(d/(name+'.npz'))==receipt[key]
        with np.load(d/'physical.npz') as z:
            active=z['active_tick'];score=z['scored_tick_reward'];assert np.all(score[active]<=1e-9)
            upper=active.reshape(500,4,120).any(1);reward=np.where(active,score,0).reshape(500,4,120).sum(1,dtype=np.float64)
        mc=discounted_returns(reward,upper,gamma=.997)
        with np.load(d/'decisions.npz') as z:values=z['old_task_value'];legal=z['valid']
        for arm in ['plain','support']:
            columns=np.array([c['arm']==arm for c in schedule[n-1]])
            for tick in [0,250,450,490,499]:
                mask=columns&legal[tick];pred=values[tick,mask];target=mc[tick,mask]
                bins.append(dict(collection=n,arm=arm,upper_tick=tick,time_s=tick*.02,samples=int(mask.sum()),
                    prediction_mean=float(pred.mean()),observed_MC_mean=float(target.mean()),rmse=float(np.sqrt(np.mean((pred-target)**2))),
                    positive_prediction_fraction=float(np.mean(pred>0)),observed_positive_reward_count=0))
    atomic_json(out/'near_terminal_values.json',bins)
    atomic_json(out/'provenance_and_interpretation.json',dict(driver_sha256=digest(driver),helper_sha256=helpers,origins=origins,extension_code_sha256=digest(__file__),checked_unix=time.time(),
        gae_calibration_label='Self-bootstrapped GAE target consistency, not independent calibration: residual equals minus reconstructed advantage.',
        MC_interpretation='Finite empirical discounted-return residual contains stochastic uncertainty. Does not establish population bias or causal Actor degradation.',
        sign_constraint='All configured task costs and terminal penalties are nonnegative costs, so true future task reward is nonpositive. Positive critic predictions violate that known sign even without MC calibration.',
        suggested_next='Frozen-policy offline critic fitting with whole-episode development holdout before modifying reward/Actor. No policy-training benefit proven.',physics_steps=0,optimizer_updates=0))
    print('helper_and_actual_origin_provenance_verified',len(bins),'value_time_bins')


if __name__=='__main__':main()
