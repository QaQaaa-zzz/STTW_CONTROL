"""Bounded Phase 2A collection of distinct same-root returns across fresh DEV starts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import run,save

p=argparse.ArgumentParser(description=__doc__)
for name in ('anchor','lateral','output','lock'):
    p.add_argument(name,type=Path)
a=p.parse_args()
out=a.output.resolve()
out.mkdir()
(out/'gpu.lock').symlink_to(a.lock.resolve())
cfg=json.loads((a.anchor/'screen.json').read_text())
original=dict(np.load(cfg['oracle_branch_reference']))
neutral=Path(cfg['initial_snapshot_path']).parent.parent/'neutral.npz'
# Freeze the command bank independently of results on held-out layouts.
available=[(original,54,-1.91,'bend098'),(original,59,-1.95,'bend100')]
for profile in (54,59):
    b=dict(np.load(a.lateral/f'profile{profile}.npz'))
    for lane,shift in ((10,.25),(24,.55)):
        available.append((b,lane,-1.95,f'bend{profile}_shift{shift}'))
variants=cfg['ground_handoff']['handoff_variants']
seeds=list(range(21901,21925))
roots={seed:np.random.default_rng(seed).uniform([-.05,-.03,-.015],[.05,.03,.015]).tolist() for seed in seeds}
save(out/'plan.json',dict(role='Development, all new initial states. Original and transfer test cases excluded.',
    map=cfg['world_map'],roots=roots,commands=[v[-1] for v in available],trigger_deltas_m=[-.02,0,.02],
    budget='Up to six batches of four new roots, eighteen unique commands per root; 72 scored plus 12 fixed batch companions per batch. Stop after 8 roots if zero qualified returns. Stop early after two additional roots each yield two distinct locked handbacks. No training.',
    gate='Freeze physics, Source, alpha 0, 1.8 m/s, left/right tasks, and a single 0.55 m map. Deduplicate full trajectory to handback within root.'))
commands=[(ref,lane,trigger+delta,name) for ref,lane,trigger,name in available for delta in (-.02,0,.02)]
records=[]
pairs=0
for idx in range(0,len(seeds),4):
    group=out/f'group_{idx//4}'
    group.mkdir()
    plans=(commands*4)+[commands[0]]*12
    ids=[seeds[idx+i//18] for i in range(72)]+[seeds[idx]]*12
    h=cfg['ground_handoff']
    v=[]
    for template,root,(_,__,trigger,___) in zip(variants,ids,plans,strict=True):
        y,yaw,roll=roots[root]
        v.append(dict(template,root_variant=root,data_split='DEV',start_y_m=y,start_yaw_rad=yaw,start_roll_rad=roll,activation_offset_m=trigger))
    base={k:val for k,val in cfg.items() if k not in ('initial_snapshot_path','oracle_branch_reference','oracle_initial_speed_m_s','oracle_initial_alpha')}
    base.update(ground_handoff=dict(h,handoff_variants=v,reference_path=str(neutral)),snapshot_loaded_far=False,verify_active_jump=False)
    run(base,group/'neutral','source',.005,out)
    refpath=str(group/'routes.npz')
    np.savez_compressed(refpath,s=original['s'],**{k:np.stack([b[k][lane] for b,lane,_,__ in plans]) for k in ('xy','heading','curvature')})
    case=group/'screen'
    full=dict(cfg,initial_snapshot_path=str(group/'neutral/source_initial_snapshot.pkl'),oracle_branch_reference=refpath,
              ground_handoff=dict(h,handoff_variants=v,reference_path=refpath),snapshot_loaded_far=True)
    physical=run(full,case,'source',10,out)
    audits=json.loads((case/'jump_validation.json').read_text())['cases']
    unique={root:set() for root in set(ids)}
    for lane,(r,j,plan) in enumerate(zip(physical[:72],audits[:72],plans[:72],strict=True)):
        assert r['case_dir']==j['case']
        load=[t for t in (j['solid_bank_landing_s'],j['both_far_after_flight_s']) if t is not None]
        hand=r['handoff_s']
        qualifies=bool(load and hand is not None and min(load)<=hand+1e-5)
        y=yaw=None
        if qualifies:
            with np.load(case/r['case_dir']/'trajectory.npz') as z:
                k=np.flatnonzero(z['trace'][:,-1]==2)[0]
                unique[ids[lane]].add(z['qpos'][:k+1].tobytes())
                y,yaw=map(float,z['trace'][k,[2,6]])
        records.append(dict(group=idx//4,lane=lane,root=ids[lane],command=plan[-1],trigger_offset=plan[2],
                   flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in j['events']),
                   loaded_s=min(load) if load else None,handoff_s=hand,qualified=qualifies,y_h_m=y,yaw_h_rad=yaw,termination=r['termination']))
    pairs+=sum(len(hands)>=2 for hands in unique.values())
    with (out/'prefixes.csv').open('w') as f:
        writer=csv.DictWriter(f,records[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    if pairs>=2 or (idx==4 and not any(r['qualified'] for r in records)):
        break
save(out/'terminal.json',dict(scored=len(records),qualified=sum(r['qualified'] for r in records),same_root_pairs=pairs))
