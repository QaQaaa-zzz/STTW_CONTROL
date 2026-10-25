"""Read-only availability audit of the inherited .8/.4rad/s, .6s gate.
No counterfactual control benefit is inferred from these factual trajectories.
"""
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from sttw_control.preference_best import digest,atomic_json
from sttw_control.reference_reserve_gate import initial_gate,gate_step,GateConfig


def main():
    root=Path('runs/entry_recovery_factorial_v1').resolve()
    source=root/'analysis/results.json';rows=json.loads(source.read_text());output=[]
    @jax.jit
    def run(rate):
        def step(state,value):
            end,_,receipt=gate_step(state,j.zeros(2),rho=j.float32(0),roll_rate=value,mode=2,dt=.02)
            return end,j.stack([receipt['active'],receipt['trigger']])
        return jax.lax.scan(step,initial_gate(),rate)[1]
    for row in rows:
        file=Path(row['raw_file']);assert digest(file)==row['raw_sha256']
        with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        # A decision at tick t can read the measured state after t-1, never t.
        # Tick0 is omitted because per-tick arrays omit the exact initial sensor.
        # This is an availability audit, not an assertion about a full new run.
        ticks=np.arange(4,len(raw['active_tick']),4)
        ticks=ticks[raw['active_tick'][ticks].astype(bool)]
        rates=raw['phi_dot'][ticks-1];receipt=np.asarray(run(j.asarray(rates)))
        hits=ticks[receipt[:,1].astype(bool)];active=receipt[:,0].astype(bool)
        first=None if not len(hits) else int(hits[0]);contacts=np.flatnonzero(raw['active_tick']&raw['contact_failure'])
        contact=None if not len(contacts) else int(contacts[0])
        output.append(dict(case_id=row['case_id'],mode=row['mode'],alpha=row['alpha'],event_id=row['event_id'],
            force_N=row['force_N'],first_trigger_tick=first,
            first_trigger_actor_legal=None if first is None else bool(raw['actor_action_valid'][first]),
            onset_delay_s=None if first is None or not row['force_N'] else (first-row['pulse_onset_task_tick'])*.005,
            contact_tick=contact,trigger_to_contact_s=None if contact is None or first is None else (contact-first)*.005,
            active_duration_on_factual_trace_s=float(active.sum()*.02),
            maximum_abs_measured_roll_rate_rad_s=float(np.max(abs(rates))),
            original_physical_outcome=row['physical_outcome']))
    atomic_json(root/'analysis/inherited_rate_detector_audit.json',dict(source=str(source),source_sha256=digest(source),
        gate=vars(GateConfig()),decision_dt_s=.02,omitted_initial_decision=True,physics_steps=0,optimizer_updates=0,cases=output,
        limitation='Factual-trigger availability only; gate-induced states and short B takeover survival/task outcomes remain untested. No threshold fitted.'))
    print('checked',len(output),'AA',[(r['event_id'],r['alpha'],r['first_trigger_tick']) for r in output if r['mode']=='AA'])


if __name__=='__main__':main()
