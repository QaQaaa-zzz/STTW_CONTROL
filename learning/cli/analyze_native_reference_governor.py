"""Describe a sealed engineering forecast without claiming closed-loop control."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    plan=json.loads((root/'plan.json').read_text())
    receipt=json.loads((root/'engineering/receipt.json').read_text())
    assert receipt['phase']=='completed' and receipt['actual_controller_episodes']==0
    with np.load(root/'engineering/prediction_traces.npz') as z:trace={k:z[k] for k in z.files}
    with np.load(root/'engineering/prediction_summary.npz') as z:summary={k:z[k] for k in z.files}
    with np.load(root/'engineering/candidates.npz') as z:grid={k:z[k] for k in z.files}
    active=trace['active_tick'].reshape(26,-1)
    xy=trace['actual_xy'].reshape(26,-1,2)
    reward=trace['scored_tick_reward'].reshape(26,-1)
    costs=-np.where(active,reward,0).sum(axis=1,dtype=np.float64)
    np.testing.assert_allclose(costs,summary['original_user_cost'],rtol=3e-5,atol=3e-5)
    assert int(active.sum())==receipt['new_prediction_lower_ticks']
    chosen=receipt['selected_index']
    assert chosen==int(np.argmin(summary['original_user_cost']))
    source=Path(plan['source'])
    original=json.loads((source/'analysis/review.json').read_text())
    case=plan['first_case']
    cached=next(r for r in original['records'] if r['family']=='zero' and
                (r['route'],r['speed'],r['alpha'])==(case['route'],case['speed'],case['alpha']))
    assert sha(cached['raw_file'])==cached['raw_sha256']
    with np.load(cached['raw_file']) as z:cached_xy=z['actual_xy'][:200]
    zero=plan['zero_candidate_index']
    cached_error=float(np.max(np.linalg.norm(xy[zero]-cached_xy,axis=1)))
    rows=[dict(index=i,latent=grid['latents'][i].tolist(),physical_offset_target=grid['physical_offsets'][i].tolist(),
        observed_lower_ticks=int(active[i].sum()),original_user_cost=float(costs[i]),
        native_physical_failure=bool(summary['native_physical_failure'][i]),native_domain_exit=bool(summary['native_domain_exit'][i])) for i in range(26)]
    output=root/'analysis';output.mkdir(exist_ok=False)
    result=dict(rows=rows,selected_index=chosen,selected_physical_offset_target=rows[chosen]['physical_offset_target'],
        cached_zero_prefix_xy_max_l2_difference_m=cached_error,cached_zero_raw_sha256=cached['raw_sha256'],
        exact_replay_gate=False,new_prediction_lower_ticks=int(active.sum()),new_actual_controller_episodes=0,
        compile_s=receipt['compile_s'],execution_s=receipt['execution_s'],
        actual_controller_benefit='not_tested',future_force='zero forecast only',
        source_sha256={str(p.resolve()):sha(p) for p in [root/'plan.json',root/'engineering/receipt.json',
            root/'engineering/prediction_traces.npz',root/'engineering/prediction_summary.npz',Path(__file__)]})
    (output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for i in range(26):
        p=xy[i][active[i]]
        axes[0].plot(*p.T,color='.7',lw=.7,label='All26 simulated candidates' if i==0 else None)
        axes[0].scatter(*p[-1],color='.65',s=8)
    for i,color,style,label in [(zero,'black','--','Zero target forecast'),(chosen,'#0072B2','-','Minimum forecast cost')]:
        axes[0].plot(*xy[i][active[i]].T,color=color,ls=style,lw=1.7,label=label)
    axes[0].plot(*cached_xy.T,':',color='#D55E00',lw=1.2,label='Cached executed zero prefix')
    with np.load(source/'evaluation'/(case['route']+'_path.npz')) as z:
        axes[0].plot(*z['xy'].T,':',color='.3',label='Original geometric path')
    all_points=xy.reshape(-1,2);lo=all_points.min(axis=0)-.1;hi=all_points.max(axis=0)+.1
    axes[0].set_xlim(lo[0],hi[0]);axes[0].set_ylim(lo[1],hi[1])
    axes[0].set_aspect('equal',adjustable='box');axes[0].set(xlabel='World X (m)',ylabel='World Y (m)')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    axes[1].plot(range(26),costs,'o',color='.4',label='Original native V3 forecast cost')
    axes[1].scatter([chosen],[costs[chosen]],marker='*',s=100,color='#0072B2',label='Selected')
    axes[1].set(xlabel='Fixed candidate index',ylabel='Original forecast cost',ylim=(0,max(costs)*1.1))
    axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    fig.suptitle('Engineering prediction only: native task0 / alpha0 / requested2.6m/s /1s forecast\nAll26 predictions, not closed-loop governor execution; future force assumed zero')
    fig.savefig(output/'forecast_xy_and_cost.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_sha256']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
