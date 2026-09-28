"""Gate-ordered evaluator. Future schedules remain here, outside control API."""
import json,pickle,time
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from .teleop_commands import command_stream,random_schedule
from .teleop_metrics import baseline_metrics,tracking_metrics,core_gate
from .teleop_reporting import save_trace,plot_case,report
from .closed_loop_predictor import ClosedLoopPredictor
from .preference_governor import PreferenceGovernor
from .teleop_budget import BudgetExhausted

def dump_snapshot(path,s):
    with Path(path).open('wb') as f:pickle.dump(jax.device_get(s),f)

def decorate(logs,stream,case,method):
    x={k:np.asarray(v) for k,v in logs.items()};n=len(x['phi'])
    x['time']=(np.arange(n)+1)*.005;x['raw_target']=stream[:n,:2]
    x['case_id']=np.full(n,case);x['method']=np.full(n,method);x['alpha']=np.full(n,-1 if method=='B0' else int(method[1]))
    for key,value in [('mode',0),('g',0.),('raw_feasible',False),('raw_reserve',False),('fallback',False),('recovery_blocked',False)]:
        if key not in x:x[key]=np.full(n,value)
    x['end_code']=np.full(n,'ongoing',dtype='<U24')
    x['end_code'][-1]='physical_failure' if x['physical_failure'][-1] else ('window_end' if n==3200 else 'incomplete')
    return x

class Review:
    def __init__(self,env,prepared,budget,root,config):
        self.env=env;self.prepared=prepared;self.budget=budget;self.root=Path(root);self.config=config
        self.predictor=ClosedLoopPredictor(env,budget);self.metrics={};self.cases={};self.latencies=[]
        self.status=dict(status='in_progress',classification=None,gates={k:'not_run' for k in ['interfaces','prediction_replay','cpu_mjx_audit','A_B0','A_governor','B_preference','B_heading','C_random','ablation','real_time']})
        self.status['gates']['interfaces']='passed';self.status['gates']['prediction_replay']='passed'
        self.traces={};self.snapshots={}
        (self.root/'traces').mkdir(exist_ok=True)

    def restore(self):
        self.status=json.loads((self.root/'status.json').read_text())
        self.metrics=json.loads((self.root/'metrics.json').read_text())
        self.cases=json.loads((self.root/'case_manifest.json').read_text())
        for case,info in self.cases.items():
            for method in info['methods']:
                path=self.root/'traces'/f'{case}_{method}'
                with np.load(path.with_suffix('.npz')) as z:self.traces[(case,method)]={k:z[k] for k in z.files}
                dp=path.with_suffix('.decisions.json')
                if dp.exists():self.latencies += [(d['latency_ms'],d['cold']) for d in json.loads(dp.read_text())]
            sp=self.root/'traces'/f'{case}_audit_snapshot.pkl'
            if sp.exists():
                with sp.open('rb') as f:self.snapshots[case]=pickle.load(f)

    def save(self):
        steady=[v for v,cold in self.latencies if not cold]
        latency=dict(samples=len(steady),compile_first_solve_ms=[v for v,cold in self.latencies if cold],
            prediction_compile_seconds=self.predictor.compile_seconds,real_time_ready=None)
        if steady:
            latency.update({f'p{p}_ms':float(np.percentile(steady,p)) for p in [50,95,99]})
            latency['max_ms']=float(max(steady));latency['real_time_ready']=latency['p99_ms']<=50
            self.status['gates']['real_time']='passed' if latency['p99_ms']<=40 else 'target_not_met'
        self.metrics['runtime']=latency
        report(self.root,self.status,self.metrics,self.cases)

    def baseline_batch(self,cases):
        streams=np.stack([command_stream(self.config['fixed_cases'][case]['target_rows_time_speed_steer']) for case in cases])
        self.budget.charge(plant=3200*len(cases),episodes=len(cases),reason='Gate A B0 six-case batch')
        template=jax.eval_shape(lambda s:self.env._step(s,s.raw,True),self.prepared)[1]
        blank=jax.tree.map(lambda x:jp.zeros(x.shape,x.dtype),template)
        def one(stream):
            def tick(carry,inputs):
                state,audit,first_bad,bad_seen,previous_log=carry;k,row=inputs
                old=state;state=state.replace(raw=row[2:])
                state,l=jax.lax.cond(state.failed,lambda _:(state,previous_log),lambda _:self.env._step(state,state.raw,True),None)
                audit=jax.lax.cond((k==800)&~state.failed,lambda _:state,lambda _:audit,None)
                bad_now=state.failed|(l['peak_roll']>.30)
                first_bad=jax.lax.cond((~bad_seen)&bad_now,lambda _:old,lambda _:first_bad,None)
                return (state,audit,first_bad,bad_seen|bad_now,l),l
            return jax.lax.scan(tick,(self.prepared,self.prepared,self.prepared,jp.bool_(False),blank),(jp.arange(3200),stream))
        start=time.perf_counter();fn=jax.jit(jax.vmap(one))
        args=jp.asarray(streams);compiled=fn.lower(args).compile()
        self.metrics['baseline_batch_compile_s']=time.perf_counter()-start
        start=time.perf_counter();result=compiled(args);jax.block_until_ready(result)
        self.metrics['baseline_batch_execution_s']=time.perf_counter()-start
        (ends,audits,bad,bad_seen,last),logs=jax.device_get(result)
        for i,case in enumerate(cases):
            rawlogs={k:v[i] for k,v in logs.items()}
            failures=np.flatnonzero(rawlogs['physical_failure']);n=int(failures[0]+1) if len(failures) else 3200
            x=decorate({k:v[:n] for k,v in rawlogs.items()},streams[i],case,'B0')
            save_trace(self.root/'traces'/f'{case}_B0',x);self.traces[(case,'B0')]=x
            self.snapshots[case]=jax.tree.map(lambda v:v[i],audits)
            dump_snapshot(self.root/'traces'/f'{case}_audit_snapshot.pkl',self.snapshots[case])
            if bool(bad_seen[i]):dump_snapshot(self.root/'traces'/f'{case}_failure_snapshot.pkl',jax.tree.map(lambda v:v[i],bad))
            rows=self.config['fixed_cases'][case]['target_rows_time_speed_steer']
            self.metrics[case]={'B0':baseline_metrics(x,rows)}
            if not self.metrics[case]['B0']['passed'] and not bool(bad_seen[i]):
                dump_snapshot(self.root/'traces'/f'{case}_failure_snapshot.pkl',jax.tree.map(lambda v:v[i],ends))
            self.cases[case]=dict(rows=rows,methods=['B0'],initial_snapshot='contracts/prepared_snapshot.pkl')
            plot_case(self.root,case,{'B0':x})
        self.save()
        return all(self.metrics[c]['B0']['passed'] for c in cases)

    def episode(self,case,method,rows,ablation=False):
        path=self.root/'traces'/f'{case}_{method}'
        if path.with_suffix('.npz').exists():raise RuntimeError('refusing to overwrite existing episode')
        self.budget.charge(episodes=0 if ablation else 1,ablation=int(ablation),reason=f'begin {case}/{method}')
        stream=command_stream(rows);s=self.prepared;logs=[];decisions=[];gov=PreferenceGovernor(self.predictor,disable_recovery=ablation)
        decision=None;failure_saved=False;raw_conflict=False
        try:
            for k,row in enumerate(stream):
                s=s.replace(raw=jp.asarray(row[2:]))
                if k%10==0:
                    if k%100==0:
                        (self.root/'execution_progress.json').write_text(json.dumps(dict(case=case,method=method,task_time_s=k*.005,phase='ablation' if ablation else 'main',updated_unix=time.time()))+'\n')
                    if method!='B0':
                        decision=gov.decide(s,int(method[1]));s=s.replace(governor=decision.governor)
                        self.latencies.append((decision.latency_ms,decision.cold_compile));raw_conflict|=not decision.raw_feasible
                        decisions.append(dict(t=k*.005,mode=decision.mode,latency_ms=decision.latency_ms,cold=decision.cold_compile,
                            raw_feasible=decision.raw_feasible,raw_reserve=decision.raw_reserve,selected_feasible=decision.selected_feasible,
                            fallback=decision.fallback_used,recovery_blocked=decision.recovery_blocked,
                            counts=decision.candidate_counts,predictor_ticks=decision.predictor_ticks,metrics=decision.cost_summary))
                        if decision.fallback_used and not failure_saved:
                            dump_snapshot(path.with_name(path.name+'_failure_snapshot.pkl'),s)
                            d=gov.last_diagnostics
                            np.savez_compressed(path.with_name(path.name+'_failure_candidates.npz'),goal=d['candidates'].goal,bypass=d['candidates'].bypass,valid=d['candidates'].valid,index=d['index'],**d['results'])
                            failure_saved=True
                    self.budget.charge(plant=min(10,len(stream)-k),reason=f'{case}/{method} ticks {k}:{k+10}')
                goal=s.raw if method=='B0' else jp.asarray(decision.goal);bypass=True if method=='B0' else decision.bypass
                before=s;s,l=self.env.step(s,goal,bypass)
                # Real-time recovery hold uses physical control ticks, never a future event.
                good=(jp.abs(l['e_psi_unwrapped'])<=.03)&(jp.abs(l['actual_forward_speed']-s.raw[0])<=.10)&(jp.abs(l['actual_delta']-s.raw[1])<=.03)
                hold=jp.where(good,s.governor.recovery_hold_ticks+1,0)
                s=s.replace(governor=s.governor.replace(recovery_hold_ticks=hold))
                l=jax.device_get(l)
                l.update(mode=0 if decision is None else decision.mode,g=float(s.governor.recovery_gain),
                    raw_feasible=False if decision is None else decision.raw_feasible,raw_reserve=False if decision is None else decision.raw_reserve,
                    fallback=False if decision is None else decision.fallback_used,recovery_blocked=False if decision is None else decision.recovery_blocked)
                logs.append(l)
                if bool(s.failed):
                    if not failure_saved:dump_snapshot(path.with_name(path.name+'_failure_snapshot.pkl'),before)
                    break
        except (BudgetExhausted,FloatingPointError):
            if logs:self._finish_episode(case,method,rows,stream,logs,decisions,path,raw_conflict)
            raise
        return self._finish_episode(case,method,rows,stream,logs,decisions,path,raw_conflict)

    def _finish_episode(self,case,method,rows,stream,logs,decisions,path,raw_conflict):
        x=decorate({k:np.asarray([l[k] for l in logs]) for k in logs[0]},stream,case,method)
        save_trace(path,x);path.with_suffix('.decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
        self.traces[(case,method)]=x
        self.cases.setdefault(case,dict(rows=np.asarray(rows).tolist(),methods=[],initial_snapshot='contracts/prepared_snapshot.pkl'))['methods'].append(method)
        self.metrics.setdefault(case,{})[method]=tracking_metrics(x)
        self.metrics[case][method]['raw_conflict_seen']=raw_conflict
        plot_case(self.root,case,{m:v for (c,m),v in self.traces.items() if c==case});self.save()
        return x

    def cpu_audit(self):
        from .teleop_env import TeleopEnv
        cpu=TeleopEnv(backend='cpu');results={}
        for case in ['N2','N3']:
            if len(self.traces[(case,'B0')]['time'])<=800:
                results[case]={'status':'unavailable: baseline failed before selected snapshot'};continue
            source=self.snapshots[case];c=cpu.from_mjx(source,self.env);j=source
            self.budget.charge(predictor=480,contracts=2,reason=f'{case} CPU/MJX independent 1.2s physics audit')
            cpu_logs=[];mjx_logs=[]
            for _ in range(240):
                c,cl=cpu.step(c,c.raw,True);j,jl=self.env.step(j,j.raw,True)
                cpu_logs.append(jax.device_get(cl));mjx_logs.append(jax.device_get(jl))
            fields=['actual_forward_speed','actual_delta','phi','phi_dot','yaw_unwrapped','actual_xy']
            results[case]={key:float(np.max(np.abs(np.asarray([l[key] for l in cpu_logs])-np.asarray([l[key] for l in mjx_logs])))) for key in fields}
            results[case]['cpu_failure']=bool(c.failed);results[case]['mjx_failure']=bool(j.failed)
            np.savez_compressed(self.root/'contracts'/f'{case}_cpu_mjx.npz',**{f'{engine}_{key}':np.asarray([l[key] for l in logs]) for engine,logs in [('cpu',cpu_logs),('mjx',mjx_logs)] for key in fields})
        self.metrics['cpu_mjx_differences']=results;self.status['gates']['cpu_mjx_audit']='recorded (no bitwise equivalence claimed)';self.save()

    def failure_diagnosis(self,case):
        """Single fixed-snapshot diagnostic; no new full episode or reset."""
        from .preference_governor import coarse_grid,refine_grid,select,CONFLICT
        source=self.root/'traces'/f'{case}_failure_snapshot.pkl'
        if not source.exists():return
        with source.open('rb') as f:s=pickle.load(f)
        raw=np.asarray(s.raw)
        coarse=coarse_grid(raw,np.asarray(s.governor.last_goal),np.asarray(s.governor.current_reference),raw[1],CONFLICT,0.)
        cr,_=self.predictor.predict(s,coarse)
        fine=refine_grid(coarse,cr,raw,CONFLICT,0.);fr,_=self.predictor.predict(s,fine)
        r={k:np.concatenate([cr[k],fr[k]]) for k in cr}
        goals=np.concatenate([coarse.goal,fine.goal]);bypass=np.concatenate([coarse.bypass,fine.bypass]);valid=np.concatenate([coarse.valid,fine.valid])
        out=self.root/'failure_diagnosis';out.mkdir(exist_ok=True)
        np.savez_compressed(out/f'{case}_candidates.npz',goal=goals,bypass=bypass,valid=valid,**r)
        ids=np.flatnonzero(r['feasible']);finite=np.flatnonzero(r['finite'])
        if not len(ids):ids=finite
        if not len(ids):self.metrics['failure_diagnosis']={'classification':'numerical_failure'};return
        choices=dict(raw=0,best_steer=int(ids[np.argmin(r['steer_rmse'][ids])]),best_speed=int(ids[np.argmin(r['speed_rmse'][ids])]),selected_alpha0=select(r,0,CONFLICT),selected_alpha1=select(r,1,CONFLICT))
        audit={}
        for label,index in choices.items():
            self.budget.charge(predictor=240,contracts=1,reason=f'{case} failing snapshot {label} trace')
            end,logs=self.predictor.replay(s,jp.asarray(goals[index]),bool(bypass[index]));jax.block_until_ready(end)
            x={k:np.asarray(v) for k,v in jax.device_get(logs).items()}
            x['time']=(np.arange(240)+1)*.005
            save_trace(out/f'{case}_{label}',x)
            audit[label]=dict(index=int(index),goal=goals[index].tolist(),bypass=bool(bypass[index]),feasible=bool(r['feasible'][index]),
                steer_rmse=float(r['steer_rmse'][index]),speed_rmse=float(r['speed_rmse'][index]),
                residual_clipped_ticks=int(np.sum(x['residual_clipped'])),physical_failure=bool(end.failed))
        self.metrics['failure_diagnosis']=dict(case=case,snapshot=str(source.relative_to(self.root)),choices=audit,
            feasible=int(r['feasible'].sum()),working_rejections=int(np.sum(valid&~r['working'])),terminal_rejections=int(np.sum(valid&~r['terminal'])),
            note='diagnostic finite family, no live control, no reset of ESO or physical state')
        (out/f'{case}_summary.json').write_text(json.dumps(self.metrics['failure_diagnosis'],indent=2)+'\n')
