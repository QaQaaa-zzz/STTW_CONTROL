"""Fixed offline MC value refit; never runs physics or updates Actors."""
import json,os,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from sttw_control.preference_best import atomic_json,digest,config_hash
from sttw_control.live_training_signal_audit import discounted_returns,value_calibration
from sttw_control.finite_value_fit import FiniteValue,discount_budget
from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.live_ppo_run import _tensorboard

MODES=['inherited_refit','reset_free','time_free','time_nonpositive']


def verify_reward(root,url,arm,step,expected):
    import urllib.request
    from urllib.parse import urlencode
    for _ in range(30):
        try:
            query=urlencode(dict(run=arm,tag='source_rollout/reward'))
            with urllib.request.urlopen(url+'/data/plugin/scalars/scalars?'+query,timeout=2) as f:values=json.load(f)
            if any(x[1]==step and np.isclose(x[2],expected,rtol=1e-6) for x in values):
                p=root/'tensorboard_verified.json';old=json.loads(p.read_text()) if p.exists() else {}
                old[arm]=dict(url=url,actual_historical_reward_loaded=True,offline_epoch_step=step,checked_unix=time.time())
                atomic_json(p,old);return
        except OSError:pass
        time.sleep(1)
    raise RuntimeError('historical current-run reward scalar missing through TensorBoard HTTP')


def main():
    from prepare_short_recovery_campaign import register
    root=Path('runs/frozen_value_refit_v1').resolve();root.mkdir(exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),stage='authenticating',completed_fits=0,declared_fits=8,epochs_per_fit=32,optimizer_steps=0,physics_steps=0,actor_optimizer_steps=0)
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    update();register(root/'ACTIVE_RUN.json','STTW_CONTROL 冻结策略离线Critic拟合对照',root/'status.json');writers={}
    try:
        source=Path('runs/matched_live_ppo_v1').resolve();data=source/'execution/update_0064'
        cfg=json.loads((source/'effective_config.json').read_text());source_plan=json.loads((source/'plan.json').read_text())
        assert digest(source/'plan.json')==json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']
        assert digest(source/'effective_config.json')==source_plan['files_sha256']['effective_config.json']
        complete=json.loads((data/'collection_complete.json').read_text())
        assert digest(data/'physical.npz')==complete['raw_physical_sha256'] and digest(data/'decisions.npz')==complete['raw_decisions_sha256']
        assert digest(source/'training_cases.json')==source_plan['files_sha256']['training_cases.json']
        identity=json.loads((data/'sampling_identity.json').read_text());cases=identity['cases'];assert cases==json.loads((source/'training_cases.json').read_text())[63]
        pair=json.loads((source/'checkpoint_pair_0063.json').read_text())
        assert pair['completed_updates']==63 and pair['plan_sha256']==digest(source/'plan.json')
        assert identity['plan_sha256']==digest(source/'plan.json')
        models={m['arm']:m for m in pair['models']}
        with np.load(data/'physical.npz') as z:
            active=z['active_tick'];score=z['scored_tick_reward'];upper=active.reshape(500,4,120).any(1);rewards=np.where(active,score,0).reshape(500,4,120).sum(1,dtype=np.float64)
        mc=discounted_returns(rewards,upper,gamma=.997)
        with np.load(data/'decisions.npz') as z:obs=z['critic_observation'];valid=z['valid'];old_values=z['old_task_value'];ticks=z['task_tick']
        np.testing.assert_allclose(obs[:,:,-1][valid],(2000-ticks[valid])/2000,atol=1e-7)
        for arm,m in models.items():
            assert digest(m['checkpoint'])==m['checkpoint_sha256']
            assert m['fingerprint']==identity['models'][arm]['state_sha256']
        files=[Path(__file__).resolve(),Path('learning/src/sttw_control/finite_value_fit.py').resolve(),Path('learning/src/sttw_control/live_training_signal_audit.py').resolve(),Path('learning/src/sttw_control/path_command_policy.py').resolve(),Path('learning/src/sttw_control/live_ppo_run.py').resolve(),Path('learning/src/sttw_control/preference_best.py').resolve(),Path('docs/superpowers/plans/2026-10-25-frozen-value-refit.md').resolve()]
        plan=dict(project='STTW_CONTROL',experiment='frozen_value_refit_v1',source=str(source),collection=64,sampling_policy_checkpoint=63,
            models=models,raw_physical_sha256=complete['raw_physical_sha256'],raw_decisions_sha256=complete['raw_decisions_sha256'],sampling_identity_sha256=digest(data/'sampling_identity.json'),
            partition='replicate0 train30, replicate1 development30 per arm; whole episodes, same ancestral initial states',
            torch_version=torch.__version__,modes=MODES,epochs=32,batch_size=1024,lr=.0003,grad_clip=1.,loss='MC MSE in original value units on legal decisions',seed=2026102564,
            physics_steps=0,actor_optimizer_steps=0,policy_adoption='none',scope='Value prediction diagnostic only; no proof of Actor benefit or independent generalization',
            source_sha256={str(f):digest(f) for f in files},code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        for f in files:
            dest=root/'source_at_launch'/f.relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
        if not torch.cuda.is_available():raise RuntimeError('GPU required for this declared offline fit')
        torch.set_num_threads(4);device=torch.device('cuda');url,writers=_tensorboard(root);update(tensorboard_url=url)
        histories=[];final=[];started=time.monotonic()
        for arm_index,arm in enumerate(['plain','support']):
            m=models[arm];checkpoint=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
            assert checkpoint['next_collection']==63 and checkpoint['config_sha256']==config_hash(cfg)
            base=make_torch_path_network(352,1,cfg);base.load_state_dict(checkpoint['task_critic'],strict=True)
            def dataset(rep):
                cols=np.array([c['arm']==arm and c['replicate']==rep for c in cases]);mask=valid&cols[None,:]
                return dict(x=torch.tensor(obs[mask],device=device),y=torch.tensor(mc[mask],dtype=torch.float32,device=device),ticks=ticks[mask],cases=np.flatnonzero(cols).tolist(),old=old_values[mask])
            train,dev=dataset(0),dataset(1);assert len(train['cases'])==len(dev['cases'])==30 and not set(train['cases'])&set(dev['cases'])
            factor=discount_budget(train['x'][:,-1]);rate=max(1e-6,float(-(factor*train['y']).sum()/factor.square().sum()))
            rng=np.random.default_rng(plan['seed']+arm_index);orders=[rng.permutation(len(train['y'])) for _ in range(32)]
            import hashlib
            order_hash=hashlib.sha256(np.concatenate(orders).tobytes()).hexdigest()
            initial={mode:FiniteValue(base,mode,rate=rate).to(device) for mode in MODES}
            with torch.no_grad():
                expected=initial['reset_free'](dev['x']);differences={mode:float((initial[mode](dev['x'])-expected).abs().max()) for mode in MODES[1:]}
                assert max(differences.values())<1e-4
                legacy_difference=float((initial['inherited_refit'](dev['x']).cpu()-torch.tensor(dev['old'])).abs().max())
            source_reward=float(np.mean([rewards[:,i].sum() for i in train['cases']]))
            atomic_json(root/(arm+'_dataset.json'),dict(train_cases=train['cases'],dev_cases=dev['cases'],train_samples=len(train['y']),dev_samples=len(dev['y']),rate=rate,batch_order_sha256=order_hash,matched_initial_max_abs=differences,inherited_value_replay_max_abs=legacy_difference,historical_train_episode_mean_reward=source_reward))
            def metrics(model,ds):
                with torch.no_grad():pred=model(ds['x']).cpu().numpy();target=ds['y'].cpu().numpy()
                return dict(all=value_calibration(pred,target,np.ones(len(pred),bool)),tail=value_calibration(pred,target,ds['ticks']>=1800),last20ms=value_calibration(pred,target,ds['ticks']>=1996),positive_fraction=float(np.mean(pred>0)))
            for mode_index,mode in enumerate(MODES):
                model=initial[mode];optimizer=torch.optim.Adam(model.parameters(),lr=plan['lr']);before=metrics(model,dev)
                for epoch,order in enumerate(orders,1):
                    update(stage='offline_value_fitting',arm=arm,mode=mode,epoch=epoch,elapsed_s=time.monotonic()-started)
                    for begin in range(0,len(order),1024):
                        ids=torch.tensor(order[begin:begin+1024],device=device);optimizer.zero_grad(set_to_none=True)
                        loss=(model(train['x'][ids])-train['y'][ids]).square().mean()
                        if not torch.isfinite(loss):raise ValueError('nonfinite offline value loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();state['optimizer_steps']+=1
                    train_metrics=metrics(model,train);dev_metrics=metrics(model,dev)
                    histories.append(dict(arm=arm,mode=mode,epoch=epoch,train=train_metrics,development=dev_metrics))
                    w=writers[arm];step=mode_index*32+epoch
                    for phase,met in [('train',train_metrics),('development',dev_metrics)]:
                        for window in ['all','tail','last20ms']:w.add_scalar(mode+'/'+phase+'/'+window+'_RMSE',met[window]['rmse'],step)
                        w.add_scalar(mode+'/'+phase+'/positive_fraction',met['positive_fraction'],step)
                    w.add_scalar('source_rollout/reward',source_reward,step);w.flush()
                    if epoch==1:verify_reward(root,url,arm,step,source_reward)
                    atomic_json(root/'history.json',histories)
                target=root/f'{arm}_{mode}.pt';torch.save(dict(mode=mode,rate=rate,state_dict={k:v.cpu() for k,v in model.state_dict().items()},source_checkpoint_sha256=m['checkpoint_sha256']),target)
                final.append(dict(arm=arm,mode=mode,initial_development=before,final_development=dev_metrics,final_train=train_metrics,checkpoint=str(target),checkpoint_sha256=digest(target)))
                atomic_json(root/'results.json',final);update(completed_fits=len(final))
            verify_reward(root,url,arm,128,source_reward)
        atomic_json(root/'receipt.json',dict(completed_fits=8,epochs_per_fit=32,optimizer_steps=state['optimizer_steps'],actor_optimizer_steps=0,physics_steps=0,elapsed_s=time.monotonic()-started,model_adoption='none'))
        update(phase='completed',stage='offline_value_fit_complete_review_pending',completed_fits=8)
    except BaseException as exc:
        (root/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
    finally:
        for writer in writers.values():writer.close()


if __name__=='__main__':main()
