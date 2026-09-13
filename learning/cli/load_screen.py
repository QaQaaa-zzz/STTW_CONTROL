#!/usr/bin/env python3
"""Sequential baseline rear-load screening; stop increasing on failure/non-settling."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
from sttw_control.env import RecoveryEnv,config_from_dict
from sttw_control.evaluation import evaluate
from sttw_control.media import state_series,plot_states
from sttw_control.speed_recovery import paired_speed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);a=parser.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['output']);root.mkdir(exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];nom=None
    def status(phase,**kwargs):(root/'status.json').write_text(json.dumps(dict(phase=phase,results=rows,**kwargs),indent=2)+'\n')
    try:
        for torque in plan['loads_nm']:
            status('running',load_nm=torque)
            raw=dict(plan['task']);raw.update(random_events=None,disturbance_rear_torque=torque,disturbance_force=0.,disturbance_steer_rate=0.)
            c=config_from_dict(raw);env=RecoveryEnv(c,backend='cpu');dest=root/f'load_{torque:g}'
            summary=evaluate(env,dest,seed=plan['seed']);tr=dict(np.load(dest/'trace.npz'));v=state_series(tr,raw)['speed']
            plots=dest/'plots';plots.mkdir();plot_states(tr,raw,plots)
            if nom is None:nom=(tr['time'],v)
            x=paired_speed(tr['time'],v,*nom,start=c.disturbance_start,end=c.disturbance_start+c.disturbance_duration,target=c.speed_reference,failed=summary['physical_failure'])
            rows.append(dict(load_nm=torque,**x['summary']))
            np.savez_compressed(dest/'paired_speed.npz',time=tr['time'],true_speed=v,extra_speed=x['extra_speed'],deficit=x['deficit'])
            (root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            if summary['physical_failure'] or not x['summary']['final_speed_band_held']:
                status('stopped',reason='failed or did not hold speed band; no higher load attempted');break
            del env; jax.clear_caches()
        else:status('complete')
        (root/'INDEX.md').write_text('# Rear load baseline screen\n\nPaired against same-seed nominal; no learned policy or new training.\n\n'+'\n'.join(f"- {r['load_nm']:g} Nm: [trajectory](load_{r['load_nm']:g}/plots/trajectory.png), max extra drop {r['maximum_extra_drop_m_s']}, deficit {r['deficit_integral_m']}, speed settling {r['speed_settling_after_event_s']}" for r in rows)+'\n')
    except Exception as exc:status('error',error=str(exc));raise

if __name__=='__main__':main()
