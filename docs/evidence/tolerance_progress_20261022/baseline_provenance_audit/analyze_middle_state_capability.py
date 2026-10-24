"""Paired physical outcomes and frozen forecasts; no refitting after labels."""
from pathlib import Path
import argparse,json,time,os,pickle,hashlib
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve()
    from sttw_control.preference_best import atomic_json
    status=dict(phase='running',stage='reading',pid=os.getpid(),new_physics_steps=0,policy_updates=0,classifier_fits=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(run/'analysis_status.json',status)
    update()
    try:
        from sttw_control.balance_probe import classify_contact_balance
        assert json.loads((run/'status.json').read_text())['phase']=='completed'
        plan=json.loads((run/'scan_plan.json').read_text())
        prediction=json.loads((run/'predictions_before_scan.json').read_text())
        launch=json.loads((run/'scan_launch_receipt.json').read_text())
        assert prediction['saved_unix']<launch['started_unix']
        for path,sha in plan['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        results=json.loads((run/'results.json').read_text())['cases']
        prefixes=json.loads((run/'prefixes.json').read_text())['cases']
        source={p['state_id']:p for p in prefixes};traces={};rows=[];source_hashes=[]
        for case,observed,pred in zip(plan['cases'],results,prediction['cases']):
            assert case['state_id']==pred['state_id'] and case['force_N']==pred['force_N']==observed['force_N']
            file=run/(observed['name']+'.npz')
            with np.load(file) as d:
                label=classify_contact_balance(d,horizon_s=4.)
                assert label['balance_label']==observed['balance_label']
                assert label['active_ticks']==observed['active_ticks']
                active=d['active_tick'].astype(bool);trace={k:d[k][active] for k in d.files}
            np.testing.assert_array_equal(trace['reference'],np.broadcast_to(np.asarray(case['reference'],np.float32),(len(trace['time']),2)))
            source_hashes.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            traces[case['state_id'],case['force_N']]=trace
            rows.append(dict(**case,**label,name=observed['name'],scores=pred['probability_scores'],
                analytic_capacity_N=pred['analytic_capacity_N'],analytic_predict_contact=pred['analytic_predict_contact'],
                impulse_error_Ns=float(np.linalg.norm(np.asarray(observed['impulse_Ns'])-observed['expected_impulse_Ns']))))
        assert len(rows)==56 and len(source)==8
        methods=list(rows[0]['scores'])+['analytic'];metrics={}
        for method in methods:
            metrics[method]={}
            for subset in ['nonzero','zero']:
                rr=[r for r in rows if (r['force_N']!=0)==(subset=='nonzero')]
                y=np.asarray([r['balance_label']=='FLOOR_CONTACT' for r in rr])
                known=np.asarray([r['balance_label'] in ['FLOOR_CONTACT','NO_FLOOR_CONTACT'] for r in rr])
                score=np.asarray([abs(r['force_N'])-r['analytic_capacity_N'] if method=='analytic' else r['scores'][method] for r in rr])
                bad=score>0 if method=='analytic' else score>=.5
                positives=score[y&known];negatives=score[~y&known]
                auc=float(np.mean([float(a>b)+.5*float(a==b) for a in positives for b in negatives])) if len(positives) and len(negatives) else None
                metrics[method][subset]=dict(cases=len(rr),actual_contact=int((y&known).sum()),
                    actual_no_contact=int((~y&known).sum()),unknown=int((~known).sum()),
                    false_safe=int((y&known&~bad).sum()),conservative=int((~y&known&bad).sum()),AUC=auc)
        states=[]
        for sid in source:
            rr=[r for r in rows if r['state_id']==sid];zero=next(r for r in rr if r['force_N']==0)
            directions=[]
            for side in [-1,1]:
                q=sorted([zero]+[r for r in rr if r['force_N']*side>0],key=lambda r:abs(r['force_N']))
                seen_failure=False;reversal=False;last_safe=None;first_fail=None
                for r in q:
                    if r['balance_label']=='FLOOR_CONTACT':
                        seen_failure=True
                        if first_fail is None:first_fail=abs(r['force_N'])
                    elif r['balance_label']=='NO_FLOOR_CONTACT':
                        reversal|=seen_failure
                        if not seen_failure:last_safe=abs(r['force_N'])
                directions.append(dict(side=side,grid_last_safe_before_failure_N=last_safe,
                    first_tested_failure_N=first_fail,nonmonotone_tested_grid=bool(reversal),
                    unforced_contact=zero['balance_label']=='FLOOR_CONTACT',
                    scope='Tested points only; no guarantee for untested amplitudes.'))
            states.append(dict(prefix=source[sid],zero_label=zero['balance_label'],directions=directions,
                disturbed_no_contact=sum(r['force_N']!=0 and r['balance_label']=='NO_FLOOR_CONTACT' for r in rr),
                disturbed_contact=sum(r['force_N']!=0 and r['balance_label']=='FLOOR_CONTACT' for r in rr)))
        atomic_json(run/'analysis.json',dict(rows=rows,states=states,metrics=metrics,source_sha256=source_hashes,
            new_physics_steps=0,policy_updates=0,classifier_fits=0,
            prediction_lead_seconds=launch['started_unix']-prediction['saved_unix'],
            scope='New fixed-reference suffix outcomes from historical DEV H75 states; not full-middle task success or independent TEST.'))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(20,10),layout='constrained')
        decl=json.loads(Path(plan['source_declaration']).read_text());cc=decl['source']['controller']
        colors={0.:'black',-32.:'tab:blue',32.:'tab:orange',-48.:'tab:green',48.:'tab:red',-64.:'tab:purple',64.:'tab:brown'}
        for ax,item in zip(axes.flat,states):
            prefix=item['prefix'];sid=prefix['state_id'];state=pickle.loads(Path(prefix['snapshot']).read_bytes())
            origin=np.asarray(state.physical.data.qpos[:2]);yaw=float(state.physical.yaw_unwrapped)
            speed,steer=prefix['held_reference'];t=np.arange(801)*.005;k=np.tan(steer)*np.cos(cc['caster'])/cc['wheelbase']
            if abs(k)<1e-8:guide=origin+t[:,None]*speed*np.array([np.cos(yaw),np.sin(yaw)])
            else:guide=origin+np.column_stack([(np.sin(yaw+k*speed*t)-np.sin(yaw))/k,-(np.cos(yaw+k*speed*t)-np.cos(yaw))/k])
            ax.plot(*guide.T,':',color='gray',label='held-command guide')
            for r in rows:
                if r['state_id']!=sid:continue
                trace=traces[sid,r['force_N']];xy=trace['actual_xy']
                ax.plot(*xy.T,color=colors[r['force_N']],label=f"{r['force_N']:+g} N")
                ax.scatter(*xy[-1],marker='x' if r['balance_label']=='FLOOR_CONTACT' else 'o',color=colors[r['force_N']],s=20)
            ax.scatter(*origin,color='black',marker='s');ax.set_title(sid,fontsize=9)
            ax.set(xlabel='world x (m)',ylabel='world y (m)');ax.set_aspect('equal')
        axes.flat[0].legend(fontsize=7);fig.suptitle('Frozen local300 and held reference; H75 generated prefix only; x = contact')
        fig.savefig(run/'all_paths.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(8,2,figsize=(12,24),layout='constrained')
        for i,item in enumerate(states):
            sid=item['prefix']['state_id']
            for col,side in enumerate([-1,1]):
                ax=axes[i,col];rr=sorted([r for r in rows if r['state_id']==sid and (r['force_N']*side>0 or r['force_N']==0)],key=lambda r:abs(r['force_N']))
                force=[abs(r['force_N']) for r in rr]
                for method in methods[:-1]:ax.plot(force,[r['scores'][method] for r in rr],'.--',label=method)
                ax.scatter(force,[r['balance_label']=='FLOOR_CONTACT' for r in rr],color='black',marker='x',label='actual contact')
                ax.axhline(.5,color='gray',ls=':');ax.set(title=sid+f' side {side:+d}',xlabel='queried amplitude (N)',ylabel='conditional score',ylim=(-.05,1.05))
        axes[0,0].legend(fontsize=7);fig.savefig(run/'frozen_predictions_vs_outcomes.png',dpi=130);plt.close(fig)
        update(phase='completed',stage='all_cases_and_plots_complete',cases=56)
        print(json.dumps(metrics,indent=2))
    except BaseException as exc:
        update(phase='error',stage='analysis_error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
