"""Full MJX closed loop, current command held, fixed vmap/scan prediction."""
import time
import jax
import jax.numpy as jp
import numpy as np
from .controller import _system

def mask_validity(result,valid):
    result=dict(result)
    result['feasible']=result['feasible'] & np.asarray(valid)
    result['finite']=result['finite'] & np.asarray(valid)
    return result

class ClosedLoopPredictor:
    def __init__(self,env,budget):
        if env.backend!='mjx':raise ValueError('V1 predictor requires MJX JAX')
        self.env=env;self.budget=budget;self.compile_seconds={};self.warmed=set();self.executables={}
        self.compiled=jax.jit(jax.vmap(self._one,in_axes=(None,0,0)))
        self.replay=jax.jit(self._replay)

    def _replay(self,s,goal,bypass):
        def step(s,_):return self.env._step(s,goal,bypass)
        return jax.lax.scan(step,s,None,length=240)

    def _one(self,s,goal,bypass):
        env=self.env;raw=s.raw
        # Accumulators only: no full 144 x 240 physics trajectories on disk/device.
        # [steer_sq, speed_sq, heading_sq, smooth, peak_roll, peak_rate,
        #  max_violation, sum_violation_sq, heading_at_H, all_finite, command_violation]
        sums=jp.zeros(11).at[9].set(1.)
        def tick(carry,i):
            state,a=carry;prev=state.actuator.previous
            state,l=env._step(state,goal,bypass)
            roll=l['peak_roll'];rate=l['peak_roll_rate']
            vr=jp.maximum((roll-.30)/.06,0.);vd=jp.maximum((rate-1.5)/1.5,0.)
            performance=i<160
            smooth=.02*jp.sum(((l['final_command']-prev)/jp.array([3.,60.]))**2)+.001*jp.sum((l['applied_residual']/jp.array([1.5,10.]))**2)
            a=a.at[:4].add(jp.where(performance,jp.array([(l['actual_delta']-raw[1])**2,
                (l['actual_forward_speed']-raw[0])**2,(l['e_psi_unwrapped']/.1)**2,smooth]),jp.zeros(4)))
            a=a.at[4].set(jp.maximum(a[4],roll)).at[5].set(jp.maximum(a[5],rate))
            a=a.at[6].set(jp.maximum(a[6],jp.maximum(vr,vd))).at[7].add(vr**2+vd**2)
            a=a.at[8].set(jp.where(i==159,l['e_psi_unwrapped'],a[8]))
            a=a.at[9].set(a[9]*(~l['nonfinite']))
            command_bad=jp.any(jp.abs(l['final_command'])>jp.array([3.,60.]))
            a=a.at[10].set(jp.maximum(a[10],command_bad))
            return (state,a),None
        (end,a),_=jax.lax.scan(tick,(s,sums),jp.arange(240))
        m,_,_=env.observe(end.data)
        *_,ratio,_=_system(m[5]*.1,end.controller.gains,env.cc)
        eq_error=jp.abs(m[0]-m[2]/ratio)
        terminal_rate=jp.abs(m[1])
        finite=(a[9]>0)&jp.all(jp.isfinite(a))&jp.isfinite(eq_error)
        working=(a[4]<=.30)&(a[5]<=1.5)
        terminal=(terminal_rate<=.30)&(eq_error<=.08)
        feasible=finite&~end.failed&working&terminal&(a[10]==0)
        violation=a[6]+a[7]/240+jp.maximum((terminal_rate-.30)/.30,0)**2+jp.maximum((eq_error-.08)/.08,0)**2
        return dict(steer_rmse=jp.sqrt(a[0]/160),speed_rmse=jp.sqrt(a[1]/160),
            heading_cost=a[2]/160+2*(a[8]/.1)**2,smoothness=a[3]/160,
            peak_roll=a[4],peak_roll_rate=a[5],terminal_rate=terminal_rate,equilibrium_error=eq_error,
            finite=finite,physical_failure=end.failed,working=working,terminal=terminal,
            command_ok=a[10]==0,feasible=feasible,reserve=feasible&(a[4]<=.24)&(a[5]<=.8),
            violation=violation,terminal_qpos=end.data.qpos,terminal_qvel=end.data.qvel)

    def predict(self,s,candidates):
        n=len(candidates.valid)
        self.budget.charge(predictor=n*240,reason=f'prediction batch {n} including padding')
        goals=jp.asarray(candidates.goal);bypass=jp.asarray(candidates.bypass)
        cold=n not in self.warmed
        if cold:
            start=time.perf_counter()
            self.executables[n]=self.compiled.lower(s,goals,bypass).compile()
            self.compile_seconds[str(n)]=time.perf_counter()-start
            self.warmed.add(n)
        result=self.executables[n](s,goals,bypass)
        jax.block_until_ready(result)
        result={k:np.asarray(v) for k,v in jax.device_get(result).items()}
        return mask_validity(result,candidates.valid),cold
