"""Export measured event states from single-lane ditch evaluations, without fitting a model."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cases=json.loads((a.run/'protocol.json').read_text())['cases'];manifest=[]
for name in cases:
    src=a.run/name;cfg=json.loads((src/'config.json').read_text())
    summary=json.loads((src/'summary.json').read_text())['cases'][0]
    lane=src/summary['candidate']['name']
    d=np.load(lane/'verified_diagnostics.npz');phys=np.load(lane/'physics_contact.npz');policy=np.load(lane/'trajectory.npz')
    t=d['time'];stage=d['stage'][:,0];attempt=d['attempt'][:,0];target=d['target'][:,0];loaded=d['loaded'][:,0]
    if not np.array_equal(t,phys['time']) or len(t)!=len(phys['root_state_wxyz']):
        raise ValueError(f'{name}: contact and physical clocks differ')
    events=[]
    def add(event,index):
        if index is None:return
        events.append(dict(event=event,physics_index=int(index),time_s=float(t[index])))
    def first(mask):
        found=np.flatnonzero(mask)
        return int(found[0]) if len(found) else None
    entry=np.load(src/'approach_entry.npz');entry_time=float(entry['time_s'])
    if entry_time < t[0]:
        raise ValueError('Exporter requires a recorded live approach before jump entry')
    # Entry/handback occur at a policy boundary before the next physical step.
    add('jump_control_entry',int(np.searchsorted(t,entry_time,side='right')-1))
    after=t>entry_time
    launch=first(after & attempt)
    add('flight_attempt_confirmed',launch)
    add('clearance_confirmed',first(after & (stage>=2)))
    add('first_far_bank_contact',first(after & (np.arange(len(t)) >= (launch if launch is not None else len(t))) & target.any(-1)))
    add('landing_capture',first(after & (stage>=3)))
    hand=src/'handoff.json'
    if hand.exists():
        h=json.loads(hand.read_text());add('sttw_handoff',int(np.searchsorted(t,h['time_s'],side='right')-1))
    add('window_end' if summary['terminal_code']==9 else 'termination',len(t)-1)
    events.sort(key=lambda e:e['time_s']);indices=np.array([e['physics_index'] for e in events])
    # Root convention is wxyz as recorded. No future state is placed in actor inputs.
    np.savez_compressed(a.output/(name+'.npz'),event_time=t[indices],event_root_wxyz=phys['root_state_wxyz'][indices,0],
        event_joints=phys['joint_state'][indices,0],joint_names=phys['joint_names'],
        physics_time=t,contact_mode=loaded[:,0].astype(np.int8)+2*loaded[:,1].astype(np.int8),
        far_bank_contact=target,policy_observation=policy['observation'][:,0],policy_input_time=policy['time']-policy['time'][0],
        jump_entry_observation=entry['observation'][0],jump_entry_root_xyzw=entry['root'][0])
    record=dict(case=name,source=str(src),group='same_reset_seed_'+str(cfg['evaluation_seed']),events=events,
        missing_events=[e for e in ['flight_attempt_confirmed','clearance_confirmed','first_far_bank_contact','landing_capture','sttw_handoff'] if e not in [v['event'] for v in events]],
        terminated_code=summary['terminal_code'],bounded_continuation=summary['ground_continued_to_horizon'],
        approach=cfg['ground_handoff']['approach'],terrain=cfg['ditch_terrain'])
    manifest.append(record)
(a.output/'manifest.json').write_text(json.dumps(dict(schema='ditch_measured_events_v1',cases=manifest,
    scope='Nominal DEV interface dataset, no fit/test split or generalization claim. All same-reset candidates share one group.',
    contact_mode='Bit0 wheel0 loaded, bit1 wheel1 loaded; no-load is not by itself an active jump.',
    event_semantics='Attempt/stage labels reuse existing physical scorer; confirmed flight is debounced, not exact first loss of support. First far-bank contact uses collider-filtered target flag after confirmed attempt. Capture is not one-second stable support.',
    timing='Event root quaternions are wxyz; jump_entry_root_xyzw follows the plant physical() xyzw convention. Events and states aligned at solved physics step. Policy observations are PRE-step at time-0.02. Entry/handoff align with preceding solved step.',
    limitations='Controller ESO/filter hidden states not serialized; actor observations retain existing history. Commands remain in source ground_commands.npz; policy action array is proposals, not ground executed actions. Local height crops and planned action features not yet exported; not a complete Markov training dataset.'),indent=2))
print(json.dumps({r['case']:{e['event']:e['time_s'] for e in r['events']} for r in manifest},indent=2))
