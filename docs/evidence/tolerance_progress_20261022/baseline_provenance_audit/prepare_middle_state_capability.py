"""Freeze hypothetical force predictions on authentic H75 states before scanning.

The middle Actor generated the prefix only. Every suffix holds its captured
governed reference and runs frozen local300; no suffix task-success claim.
"""
from pathlib import Path
import argparse,json,hashlib,pickle,time,os
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-baselines',type=Path)
    args=parser.parse_args();out=args.output.resolve();root=Path(__file__).resolve().parents[4]
    from sttw_control.preference_best import atomic_json
    status=dict(phase='running',stage='loading',pid=os.getpid(),classifier_fits=0,new_physics_steps=0,policy_updates=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(out/'prepare_status.json',status)
    update()
    try:
        import jax
        from scipy.optimize import minimize
        from scipy.special import expit
        from torch.utils.tensorboard import SummaryWriter
        from sttw_control.path_command_training import load_environment
        from sttw_control.tolerance_state_features import unstable_features
        from sttw_control.conditional_tolerance import logistic_objective,predict_conditional
        origin=root/'runs/closed_loop_tolerance_dataset_v1'
        dataset=json.loads((origin/'dataset.json').read_text());rows=dataset['rows']
        y=np.asarray([r['fell'] for r in rows],float);baseline={}
        writer=None if args.reuse_baselines else SummaryWriter(str(origin/'tensorboard/low_middle_prediction_baselines'))
        reused=json.loads(args.reuse_baselines.read_text()) if args.reuse_baselines else None
        for i,name in enumerate(['speed_force','generic_augmented'],1):
            x=np.asarray([r['features'][name] for r in rows]);mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
            design=np.c_[(x-mean)/scale,np.ones(len(x))]
            if reused is not None:
                weights=reused[name]
                np.testing.assert_array_equal(weights['mean'],mean)
                np.testing.assert_array_equal(weights['scale'],scale)
                theta=np.asarray(weights['theta']);_,gradient=logistic_objective(theta,design,y,l2=1.)
                assert np.isfinite(theta).all() and theta[-2]>=0
                baseline[name]=weights
                update(reused_classifier_fits=i,reuse_baselines=str(args.reuse_baselines),
                    baseline_gradient_diagnostic=float(np.max(np.abs(gradient))))
                continue
            bounds=[(None,None)]*design.shape[1];bounds[-2]=(0.,None)
            fit=minimize(lambda t:logistic_objective(t,design,y,l2=1.),np.zeros(design.shape[1]),
                jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
            if not fit.success:raise RuntimeError(str(fit.message))
            baseline[name]=dict(mean=mean.tolist(),scale=scale.tolist(),theta=fit.x.tolist(),
                trained_on='All 198 previously declared historical DEV cases, no new suffix outcomes')
            loss=float(np.mean(np.logaddexp(0,design@fit.x)-y*(design@fit.x)))
            writer.add_scalar('fit/train_log_loss',loss,i);writer.flush();update(classifier_fits=i)
        if writer is not None:writer.close()
        atomic_json(out/'standardized_candidates.json',baseline)
        candidates=json.loads((root/'runs/closed_loop_tolerance_nonlinear_v1/candidate_models.json').read_text())
        atomic_json(out/'physical_scale_candidates.json',candidates)
        declaration=root/'runs/tolerance_recovery_opportunity_v1/declaration.json'
        decl=json.loads(declaration.read_text());env,_=load_environment(decl['config'],decl['source'],'terminal_contract_v2')
        source=root/'runs/reserve_feature_expansion_v1/post_fixed_final_v1/low_new_geometry_confirmation/H75/evaluation'
        manifest=json.loads((source/'manifest.json').read_text())
        assert json.dumps(manifest['physical_identity'],sort_keys=True)==json.dumps(env.physics.bundle.identity,sort_keys=True)
        parent=manifest['declaration']['policies']['H75']
        assert parent['actor_sha256']=='d10642082f74e730ce1e834608575eab04b04b16d7373784646499ad1cd2b0df'
        assert hashlib.sha256(Path(parent['actor']).read_bytes()).hexdigest()==parent['actor_sha256']
        assert parent['identity']['lower_actor_sha256']==dataset['lower_actor_sha256']
        radius=env.physics.spec['limits']['wheel_radius_proxy_m']
        cc=decl['source']['controller'];ac=decl['source']['actuator']
        snapshots=[];cases=[];predictions=[];prefixes=[]
        for route in ['left90_R3p0','right90_R3p0']:
            for speed in [1.6,1.9]:
                for alpha in [0.,1.]:
                    name=f'H75_{route}_v{speed:g}_a{alpha:g}_t3.2_snapshot.pkl';path=source/name
                    snapshot=jax.tree.map(jax.numpy.asarray,pickle.loads(path.read_bytes()))
                    if bool(snapshot.fault|snapshot.domain_exit|snapshot.physical.failed):raise ValueError('invalid declared prefix '+name)
                    measurement,pose,true_speed=env.physics.observe(snapshot.physical.data)
                    sensor=dict(zip(['roll','roll_rate','steer','steer_rate','yaw_rate','rear_rate','front_rate'],map(float,measurement)))
                    ref=np.asarray(snapshot.physical.governor.current_reference).tolist()
                    f=unstable_features(speed_proxy=sensor['rear_rate']*radius,steer=sensor['steer'],roll=sensor['roll'],
                        roll_rate=sensor['roll_rate'],disturbance=float(snapshot.physical.controller.disturbance),controller=cc)
                    sid=f'{route}_v{speed:g}_a{alpha:g}';v=f['model_speed_proxy']
                    lam=f['lambda_u'];b=f['b_u'];T=.3;delta=sensor['steer']
                    lo=max(-ac['steer_rate_limit'],(-ac['steer_limit']-delta)/T);hi=min(ac['steer_rate_limit'],(ac['steer_limit']-delta)/T)
                    q=-np.expm1(-lam*T)/lam;K=.7192241548921705*(-np.expm1(-lam*.1))/lam
                    plus=(-q*min(b*lo,b*hi)-f['z_u'])/K;minus=(f['z_u']+q*max(b*lo,b*hi))/K
                    prefixes.append(dict(state_id=sid,snapshot=str(path),sensor=sensor,held_reference=ref,
                        true_forward_speed=float(true_speed),wheel_speed_proxy=sensor['rear_rate']*radius,
                        eso=float(snapshot.physical.controller.disturbance),alpha=alpha,
                        source_middle_actor_sha256=parent['actor_sha256'],future_middle_actor_active=False))
                    for force in [0.,-32.,32.,-48.,48.,-64.,64.]:
                        side=float(np.sign(force));amp=abs(force)
                        raw=[v,v*v,side*delta,side*sensor['steer_rate'],side*sensor['roll'],side*sensor['roll_rate'],
                            side*float(snapshot.physical.controller.disturbance),ref[0]-v,side*(ref[1]-delta)]
                        features=dict(speed_force=[v,v*v,amp],generic_augmented=raw+[v*side*sensor['roll'],v*side*sensor['roll_rate'],amp])
                        probabilities={}
                        for model,feature in features.items():
                            weights=baseline[model];z=(np.asarray(feature)-weights['mean'])/weights['scale']
                            probabilities['standardized_'+model]=float(expit(z@np.asarray(weights['theta'][:-1])+weights['theta'][-1]))
                        for model,weights in candidates.items():
                            probabilities['physical_scale_'+model]=float(predict_conditional(weights,np.asarray([raw]),np.asarray([amp]))['probability'][0])
                        cap=plus if force>0 else minus if force<0 else min(plus,minus)
                        row=dict(state_id=sid,route=route,requested_speed=speed,alpha=alpha,force_N=force,reference=ref)
                        cases.append(row);snapshots.append(str(path))
                        predictions.append(dict(**row,probability_scores=probabilities,analytic_capacity_N=float(cap),
                            analytic_predict_contact=bool(amp>cap),scope='Fixed-reference conditional prediction, not Actor choice or safety guarantee'))
        assert len(cases)==56 and len(prefixes)==8
        atomic_json(out/'prefixes.json',dict(cases=prefixes,wheel_radius_proxy_m=radius))
        atomic_json(out/'predictions_before_scan.json',dict(cases=predictions,saved_unix=time.time(),new_physics_steps=0))
        files=[Path(__file__),root/'learning/src/sttw_control/conditional_tolerance.py',
            root/'learning/cli/probe_balance_boundary.py',root/'learning/src/sttw_control/balance_probe.py',
            out/'predictions_before_scan.json',out/'standardized_candidates.json',out/'physical_scale_candidates.json',source/'manifest.json']
        files+=list(map(Path,dict.fromkeys(snapshots)))
        plan=dict(project='STTW_CONTROL',purpose='冻结中层真实低速进入状态的固定参考能力确认；不执行后续中层',
            source_declaration=str(declaration),snapshot=snapshots[0],case_snapshots=snapshots,cases=cases,
            forces_N=[r['force_N'] for r in cases],references_m_s_rad=[r['reference'] for r in cases],
            pulse_delay_s=0.,pulse_duration_s=.1,observation_s=4.,maximum_lower_ticks=44800,
            training_steps=0,termination='contact_only_v1',wait_for=str(root/'runs/closed_loop_tolerance_nonlinear_v1/status.json'),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            stop_rule='One fixed 56-case grid; no force refinement, no predictor refits after new labels',
            provenance='Historical DEV nominal H75 prefixes, new constant-reference suffix outcomes, shared ancestor; not final TEST.')
        atomic_json(out/'scan_plan.json',plan)
        update(phase='completed',stage='predictions_frozen_before_new_suffixes',cases=56,prefixes=8,
            classifier_fits=0 if reused is not None else 2,reused_classifier_fits=2 if reused is not None else 0)
        print(json.dumps(dict(prefixes=8,cases=56,maximum_new_lower_ticks=44800),indent=2))
    except BaseException as exc:
        update(phase='error',stage='preparation_error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
