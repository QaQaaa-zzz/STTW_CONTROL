"""Recompute fixed 25/48/75 PPO comparisons from authenticated complete panels."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def key(r):return tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
def pair(a,b,label):
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    assert len(aa)==len(a) and len(bb)==len(b) and aa.keys()==bb.keys()
    return dict(n=len(a),baseline=sum(r[label] for r in a),candidate=sum(r[label] for r in b),gains=sum(not aa[k][label] and bb[k][label] for k in aa),losses=sum(aa[k][label] and not bb[k][label] for k in aa))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.root
    plan=read(root/'runs/preference_full_tail_length75_v1/evaluation_plan.json');run=root/'runs/preference_full_tail_length75_post_v1';state=read(run/'status.json')
    assert state['phase']=='completed' and state['completed_cases']==940
    result=dict(scope='used DEV, one training lineage, fixed75 primary; not independent final test or deployment qualification',status=state,sources={},panels={})
    count=ticks=0
    for panel,spec in plan['panels'].items():
        base=Path(spec['source'])
        for f,h in spec['source_hashes'].items():assert sha(base/f)==h
        fresh=run/panel/'evaluation';s=read(fresh/'status.json');assert s['phase']=='completed'
        a=read(base/'evaluation/results.json')['cases'];b=read(fresh/'results.json')['cases'];assert len(a)==spec['cases_per_policy'] and len(b)==2*len(a)
        assert (run/panel/'analysis/REPORT.md').exists()
        count+=len(b);ticks+=s['active_lower_ticks'];result['sources'][str(fresh/'results.json')]=sha(fresh/'results.json')
        result['panels'][panel]={}
        for pol in ('ppo48','ppo75'):
            g=[r for r in b if r['policy']==pol]
            result['panels'][panel][pol]={label:pair(a,g,label) for label in ('task_recovered','performance_success')}
            result['panels'][panel][pol]['by_alpha']=[dict(alpha=al,**pair([r for r in a if r['alpha']==al],[r for r in g if r['alpha']==al],'task_recovered')) for al in sorted({r['alpha'] for r in a})]
    assert count==state['completed_cases'] and ticks==state['active_lower_ticks']
    result['record_totals']={pol:{k:sum(result['panels'][q][pol]['task_recovered'][k] for q in result['panels']) for k in ('n','baseline','candidate','gains','losses')} for pol in ('ppo48','ppo75')}
    result['totals_warning']='Panel records include repeated nominal controls; totals are bookkeeping, not independent sample counts.'
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['record_totals'],indent=2))
if __name__=='__main__':main()
