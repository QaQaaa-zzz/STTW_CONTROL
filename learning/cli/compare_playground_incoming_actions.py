"""Pair the existing two incoming-action history conventions at STTW to Source entry."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import run,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','out','lock'):
        p.add_argument(key,type=Path)
    args=p.parse_args()
    out=args.out.resolve()
    out.mkdir()
    (out/'gpu.lock').symlink_to(args.lock.resolve())
    groups=[('discovery',args.root/'task_reversal_bend_prefixes',(54,59)),
            ('new_roots',args.root/'task_reversal_new_domain/group_0',range(72))]
    save(out/'plan.json',dict(map='wide_oracle_055',role='DEV interface diagnostic, no change to the primary matrix',
        budget='Two fixed batches, 2 previously successful controls and 72 previously screened new-root cases. No training or repeated parameter search.',
        variable='rb_history_from_ground_targets False to True; replay identical initial snapshots, frozen skills, physics, routes, triggers and locked handback criteria.',
        decision='Report added/retained/lost measured flights, loaded landings and locked handbacks. No adoption based on a single new or duplicated root.'))
    rows=[]
    for label,group,lanes in groups:
        cfg=json.loads((group/'screen.json').read_text())
        assert not cfg.get('rb_history_from_ground_targets',False)
        case=out/label
        actual=run(dict(cfg,rb_history_from_ground_targets=True,snapshot_loaded_far=False),case,'source',10,out)
        original=json.loads((group/'screen/status.json').read_text())['results']
        both=[json.loads((fold/'jump_validation.json').read_text())['cases'] for fold in (group/'screen',case)]
        for lane in lanes:
            base,r=original[lane],actual[lane]
            assert base['case_dir']==r['case_dir'] and base['rb_entry_s']==r['rb_entry_s']
            record=dict(group=label,lane=lane,root=r['variant']['root_variant'],case=r['case_dir'],entry_s=r['rb_entry_s'])
            with np.load(group/'screen'/r['case_dir']/'trajectory.npz') as before,np.load(case/r['case_dir']/'trajectory.npz') as after:
                k=np.flatnonzero(np.isclose(before['trace'][:,0],r['rb_entry_s'],atol=1e-6,rtol=0)).item()
                for field in ('qpos','qvel'):
                    np.testing.assert_array_equal(before[field][:k+1],after[field][:k+1])
                record['entry_action_change_l2']=float(np.linalg.norm(after['rb_state'][k,-4:]-before['rb_state'][k,-4:]))
            for method,result,audits in zip(('original','ground_actions'),(base,r),both):
                audit=audits[lane]
                assert audit['case']==result['case_dir']
                load=[t for t in (audit['solid_bank_landing_s'],audit['both_far_after_flight_s']) if t is not None]
                hand=result['handoff_s']
                record.update({method+'_'+key:value for key,value in dict(
                    flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in audit['events']),
                    loaded=bool(load),qualified=bool(load and hand is not None and min(load)<=hand+1e-5),
                    termination=result['termination']).items()})
            rows.append(record)
        with (out/'paired.csv').open('w') as f:
            writer=csv.DictWriter(f,rows[0],lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    save(out/'terminal.json',dict(scored=len(rows),qualified=sum(r['ground_actions_qualified'] for r in rows)))


if __name__=='__main__':
    main()
