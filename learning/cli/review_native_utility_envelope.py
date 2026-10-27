"""Finite-library utility geometry of12 already executed native trajectories.

Lines vary evaluation weights only. They do not interpolate robot motions or
predict the conditional policy at unexecuted preference inputs.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(lines):
    cuts={0.,1.}
    for a,b in itertools.combinations(lines,2):
        slope=a['slope']-b['slope']
        if slope:
            x=(b['intercept']-a['intercept'])/slope
            if 0<x<1:cuts.add(x)
    cuts=sorted(cuts);intervals=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        mid=(left+right)/2;values=[v['intercept']+v['slope']*mid for v in lines]
        best=min(values)
        winners=[v['execution_alpha'] for v,cost in zip(lines,values) if abs(cost-best)<1e-10]
        intervals.append(dict(left=left,right=right,best_execution_alphas=winners))
    return intervals


def main(root):
    plan=json.loads((root/'plan.json').read_text());source=Path(plan['source'])
    assert digest(source)==plan['source_sha256']
    data=json.loads(source.read_text());rows=data['records'];assert len(rows)==12
    for path,sha in data['source_sha256'].items():assert digest(path)==sha
    contexts=sorted({(r['route'],r['speed']) for r in rows});assert len(contexts)==4
    results=[]
    for route,speed in contexts:
        group=sorted([r for r in rows if (r['route'],r['speed'])==(route,speed)],key=lambda r:r['execution_alpha'])
        assert [r['execution_alpha'] for r in group]==[0.,.5,1.]
        assert all(r['complete'] and r['goal1m_before_stop'] and not r['physical_failure'] and not r['domain_exit'] for r in group)
        lines=[dict(execution_alpha=r['execution_alpha'],intercept=r['scores']['0.0']['scored'],
                    slope=r['scores']['1.0']['scored']-r['scores']['0.0']['scored']) for r in group]
        intervals=envelope(lines)
        matched=[]
        for r in group:
            alpha=r['execution_alpha'];costs=[v['intercept']+v['slope']*alpha for v in lines]
            matched.append(dict(user_alpha=alpha,matched_cost=costs[[0.,.5,1.].index(alpha)],
                best_of3_cost=min(costs),gap=costs[[0.,.5,1.].index(alpha)]-min(costs),
                best_execution_alpha=lines[costs.index(min(costs))]['execution_alpha']))
        results.append(dict(route=route,speed=speed,lines=lines,intervals=intervals,matched=matched,
            best_interval_width={str(a):sum(v['right']-v['left'] for v in intervals if a in v['best_execution_alphas']) for a in [0.,.5,1.]}))
    report=dict(contexts=results,source_sha256=plan['source_sha256'],cli_sha256=digest(__file__),
        new_physics_ticks=0,new_training_steps=0,unique_actual_traces=12,
        proof='Fixed V3 trace: endpoint speed/primary channels capped before mixing, path factor affine; remaining costs/termination independent of evaluation alpha. J(alpha)=(1-alpha)J(0)+alpha J(1).',
        scope='Four correlated seen DEV contexts. Hindsight best-of3 lower envelope, not deployment selector, continuous policy Pareto optimality, interpolated dynamics or disturbance capability.')
    (root/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,c in zip(axes.flat,results):
        for line,color,style in zip(c['lines'],['#0072B2','#D55E00','#009E73'],['-','--','-.']):
            ax.plot([0,1],[line['intercept'],line['intercept']+line['slope']],color=color,ls=style,label=f"Executed alpha={line['execution_alpha']:g}")
        xs=sorted({v[k] for v in c['intervals'] for k in ('left','right')})
        ax.plot(xs,[min(v['intercept']+v['slope']*x for v in c['lines']) for x in xs],':',color='black',lw=2,label='Hindsight best of3')
        for m in c['matched']:ax.scatter(m['user_alpha'],m['matched_cost'],color='black',s=20,marker='s')
        ax.set(title=f"{c['route']} / requested{c['speed']:g}m/s",xlabel='User evaluation alpha',ylabel='Original full10s cost',xlim=(0,1));ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('12 cached actual native H75 traces / four seen DEV contexts / F0\nWeights vary, motions stay fixed; squares=matched actual0/.5/1; no new execution or safety claim')
    for ext in ('png','pdf'):fig.savefig(root/('utility_envelope.'+ext),dpi=160)
    plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
