"""Independent filter recurrence check and frozen-command replay; zero physics."""
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from simple_reference_filter import sampled_latent
from sttw_control.path_command_policy import correction_tick
from sttw_control.preference_best import atomic_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=Path('runs/nominal_preference_transfer_v1/zero/plan.json').resolve();plan=json.loads(source.read_text());cfg=plan['config']
    dt=cfg['timing']['lower_dt_s'];decay=np.exp(-dt/cfg['residual_interface']['filter_tau_s'][1])
    @jax.jit
    def replay(nominal,rates):
        def one(carry,x):
            filtered,offset,latent=carry;index,r,rate=x
            z,target,inverse_clip=sampled_latent(rate,cfg)
            latent=z
            filtered,offset,governed,flags=correction_tick(filtered,offset,latent,r,cfg)
            return (filtered,offset,latent),(governed,offset,flags['rate_clipped'],flags['reference_clipped'],inverse_clip,target)
        return jax.lax.scan(one,(j.zeros(2),j.zeros(2),j.zeros(2)),(j.arange(len(nominal)),nominal,rates))[1]
    # Constant rate: at the reference stream's200Hz update the target is exact. This
    # compares the real mapping/filter implementation to an independent scalar
    # first-order recurrence, not another copy of the target construction.
    n=100;rate=.03;r=np.arange(1,n+1)*dt*rate;nom=np.column_stack([np.full(n,2.5),r]).astype('float32')
    result=jax.device_get(replay(j.asarray(nom),j.full(n,rate)))
    expected=[];y=0.
    for x in r:y=decay*y+(1-decay)*x;expected.append(y)
    error=float(np.max(np.abs(result[0][:,1]-expected)))
    if error>2e-7 or result[2].any() or result[3].any():raise ValueError('Independent unbounded recurrence fixture failed')
    receipts={str(x):digest(x) for x in [source,Path(__file__).resolve(),Path(__file__).with_name('simple_reference_filter.py').resolve()]};rows=[]
    for case in plan['cases']:
        file=Path('runs/nominal_preference_transfer_v1/zero/evaluation')/(case['name']+'.npz');receipts[str(file.resolve())]=digest(file)
        with np.load(file) as z:nominal=z['nominal'];rates=z['nominal_rates'][:,1]
        replayed=jax.device_get(replay(j.asarray(nominal),j.asarray(rates)))
        expected=[];y=nominal[0,1]-dt*rates[0]
        for x in nominal[:,1]:y=decay*y+(1-decay)*x;expected.append(y)
        rows.append(dict(name=case['name'],speed=case['actual_requested_speed_m_s'],samples=len(nominal),
                         actual_vs_ideal200Hz_max_rad=float(np.max(np.abs(replayed[0][:,1]-expected))),
                         max_offset_rad=float(np.max(np.abs(replayed[1][:,1]))),
                         correction_slew_clipped_ticks=int(replayed[2].sum()),reference_clipped_ticks=int(replayed[3].sum()),
                         inverse_map_clipped_ticks=int(replayed[4].sum()),
                         max_requested_target_rad=float(np.max(np.abs(replayed[5])))))
    out=a.output.resolve();out.mkdir(exist_ok=False)
    atomic_json(out/'check.json',dict(independent_recurrence_error_rad=error,cases=rows,input_sha256=receipts,new_physics_ticks=0,training_steps=0,scope='Frozen nominal-command replay only. Does not predict robot behavior; closed-loop follower commands change after applying this comparator.'))
    print(json.dumps(dict(independent_error_rad=error,cases=rows)))


if __name__=='__main__':main()
