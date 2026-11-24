"""Finalize the frozen PCRP decision experiment after its four required services succeed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from dispatch_paired import ALLOCATION, ROOT, STATE, REPO, PYTHON, bark, read, record, save


BRANCH = 'experiment/sttw-gravel'
REMOTE_REF = 'refs/heads/' + BRANCH
PROTOCOL = Path('docs/evidence/pcrp_matched_controls/decision_experiments')
RESULTS = PROTOCOL / 'results'
METHODS = Path('docs/METHODS_AND_RESULTS.md')
INDEXES = (PROTOCOL / 'README.md', METHODS)
CHECKSUMS = PROTOCOL / 'SHA256SUMS'
ALLOWED = [str(p) for p in (RESULTS, *INDEXES, CHECKSUMS)]
OUT = ROOT / 'final_delivery' / 'results'
STATUS = ROOT / 'final_status.json'
EVENT = 'PCRP_DECISION_20261122'
REPORT_FILES = ('README.md', 'decision_report/README.md')
EXPECTED = dict(e1_candidates=7824, e2_physical_lanes=1240, e1c_ridge_trials=64, e1c_controls=32)


def git(cwd, *args, input=None, binary=False):
    result = subprocess.run(['git', '-C', str(cwd), *map(str, args)], check=True,
                            input=input, stdout=subprocess.PIPE, text=not binary)
    return result.stdout if binary else result.stdout.strip()


def mark(stage, **fields):
    state = read(STATUS) if STATUS.exists() else {}
    state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    save(STATUS, state)


def check_export(folder):
    manifest = read(folder / 'evidence_manifest.json')
    if manifest.get('stage') != 'complete' or any(manifest.get(k) != v for k,v in EXPECTED.items()):
        raise ValueError('Exported episode counts differ from the frozen decisions')
    for name in (*REPORT_FILES, 'e1/rollouts.csv'):
        if not (folder/name).stat().st_size:
            raise ValueError(f'Empty evidence index {name}')
    if any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('The repository evidence must contain files within the export directory')


def gather():
    for stage, expected in ALLOCATION.items():
        state = read(STATE / f'{stage}.json')
        if (state.get('stage') != 'complete' or state.get('physical_rollouts') != expected
                or not Path(state['report']).is_file()):
            raise ValueError(f'{stage} has not completed its registered rollouts and report')
    pin = read(STATE / 'final_pin.json')
    worktree = Path(pin['worktree'])
    if git(worktree,'rev-parse','HEAD') != pin['commit']:
        raise ValueError('The final analysis worktree differs from its registered commit')
    mark('research_complete', final_pin=pin, scientific_rollouts=9232, diagnostic_rollouts=672)
    bark('final', 'research_complete',
         'PCRP 必做阶段已完成。科学工况 9,232 条，原排列诊断 672 条。正在核算常驻复验实耗，汇总判决并交付。')
    record('必做阶段全部完成', 'E2、E1a、E1b、E1c 的阶段报告和物理回合预算均已核实。原始记录 '+str(ROOT))
    env = dict(os.environ, PYTHONPATH=str(worktree/'learning/src'), PYTHONDONTWRITEBYTECODE='1',
               JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    def cli(name, *args):
        subprocess.run([PYTHON,'-u',str(worktree/'learning/cli'/name),'--run',str(ROOT),*map(str,args)],
                       cwd=worktree, env=env, check=True)
    if not OUT.exists():
        mark('summarizing')
        cli('summarize_pcrp_decision.py')
        summary=read(ROOT/'decision_report/summary.json')
        check_accounting(summary)
        OUT.parent.mkdir(exist_ok=True)
        mark('exporting')
        with tempfile.TemporaryDirectory(dir=OUT.parent,prefix='export-') as tmp:
            staged = Path(tmp)/'results'
            cli('export_pcrp_decision_evidence.py','--out',staged)
            check_export(staged)
            staged.rename(OUT)
    check_export(OUT)
    summary=read(ROOT/'decision_report/summary.json')
    check_accounting(summary)
    verdict=summary['verdict_zh'].strip()
    if not verdict:
        raise ValueError('Missing final E1/E2 verdict')
    return verdict


def check_accounting(summary):
    probes=summary.get('resident_probe_rollouts',0)
    main=summary['main_rollouts']
    cap=read(ROOT/'paired_authorization.json')['main_experiment_rollout_cap']
    if (summary['stage']!='complete' or summary.get('scientific_rollouts',9232)!=9232 or
            probes<0 or main!=9232+probes or main>cap or summary['diagnostic_rollouts']!=672 or
            summary['all_rollouts']!=main+672):
        raise ValueError('Unexpected final research summary or rollout accounting')


def publish_files(tree,verdict):
    """Write the export and one result link into each applicable index in this tree."""
    target=tree/RESULTS
    if target.exists():
        # A repeat is valid only if all content in the existing published package matches.
        files={f.relative_to(OUT) for f in OUT.rglob('*') if f.is_file()}
        current={f.relative_to(target) for f in target.rglob('*') if f.is_file() or f.is_symlink()}
        if current!=files or any((target/f).is_symlink() or
                 hashlib.sha256((target/f).read_bytes()).digest()!=hashlib.sha256((OUT/f).read_bytes()).digest() for f in files):
            raise ValueError(f'Existing {RESULTS} contains different evidence')
    else:
        shutil.copytree(OUT,target)
    for rel in INDEXES:
        doc=tree/rel
        link=os.path.relpath(RESULTS/'README.md',rel.parent)
        decision_link=os.path.relpath(RESULTS/'decision_report/README.md',rel.parent)
        counts=read(ROOT/'decision_report/summary.json')
        line=f'必做科学工况 9,232 条，常驻复验 {counts.get("resident_probe_rollouts",0)} 条，原批诊断 672 条，新增训练 0 次。{verdict} [逐案例 XY 与指标]({link}) · [两项判决及证据]({decision_link})'
        old=doc.read_text()
        if line in old:
            continue
        if rel==METHODS:
            heading='## PCRP 同车道相位检验与逐案例搜索'
            location=old.index('\n',old.index(heading))
        else:
            location=old.index('\n')
        doc.write_text(old[:location]+'\n\n'+line+old[location:])
    # Preserve the launch-snapshot manifest, replacing only its README checksum.
    sums=tree/CHECKSUMS
    lines=sums.read_text().splitlines(keepends=True)
    indexes=[i for i,v in enumerate(lines) if v.rstrip('\n').endswith('  README.md')]
    if len(indexes)!=1:
        raise ValueError('Launch protocol README checksum is missing or ambiguous')
    lines[indexes[0]]=hashlib.sha256((tree/INDEXES[0]).read_bytes()).hexdigest()+'  README.md\n'
    sums.write_text(''.join(lines))


def stage_commit(tree, base, verdict):
    publish_files(tree,verdict)
    git(tree,'add','--',*(str(p) for p in (*INDEXES,CHECKSUMS)))
    ignored=git(tree,'ls-files','--others','--ignored','--exclude-standard','--',RESULTS)
    git(tree,'add',*(['-f'] if ignored else []),'--',RESULTS)
    changed=git(tree,'diff','--cached','--name-only','-z',binary=True).decode().split('\0')
    if any(p and p not in ALLOWED and not p.startswith(str(RESULTS)+'/') for p in changed):
        raise ValueError('Commit contains paths outside the registered result package and indexes')
    # Check every exported path and byte, including numeric files covered by any Git ignore/filter.
    exported=sorted(str(RESULTS/f.relative_to(OUT)) for f in OUT.rglob('*') if f.is_file())
    paths=git(tree,'ls-files','--stage','-z','--',RESULTS,binary=True).decode().split('\0')
    staged={}
    for row in filter(None,paths):
        meta,path=row.split('\t',1)
        mode,obj,stage=meta.split()
        if mode not in ('100644','100755') or stage!='0':
            raise ValueError(f'Unexpected Git index mode at {path}')
        staged[path]=obj
    if sorted(staged)!=exported:
        raise ValueError('Git index does not contain exactly the full evidence export')
    raw=git(tree,'hash-object','--no-filters','--stdin-paths',input='\n'.join(exported)+'\n').splitlines()
    if raw!=[staged[path] for path in exported]:
        raise ValueError('Git filters changed the bytes of the exported evidence')
    for name in REPORT_FILES:
        blob=git(tree,'show',':'+str(RESULTS/name),binary=True)
        if blob!=(OUT/name).read_bytes():
            raise ValueError('Staged report differs from the exporter: '+name)
    if not any(changed):
        return base
    git(tree,'commit','-m','Publish the completed PCRP paired decision experiments')
    commit=git(tree,'rev-parse','HEAD')
    if git(tree,'show','-s','--format=%P',commit)!=base:
        raise ValueError('Unexpected parent of the results-only commit')
    return commit


def verify_commit(tree, sha, expected_tree, verdict):
    if git(tree,'rev-parse',f'{sha}:{RESULTS}')!=expected_tree:
        raise ValueError('Committed result tree differs from the prepared evidence')
    for name in REPORT_FILES:
        if git(tree,'show',f'{sha}:{RESULTS/name}',binary=True)!=(OUT/name).read_bytes():
            raise ValueError('Committed report differs from the exporter: '+name)
    sums=git(tree,'show',f'{sha}:{CHECKSUMS}').splitlines()
    protocol_readme=git(tree,'show',f'{sha}:{INDEXES[0]}',binary=True)
    if hashlib.sha256(protocol_readme).hexdigest()+'  README.md' not in sums:
        raise ValueError('Protocol README does not match its published SHA-256')
    for rel in INDEXES:
        data=git(tree,'show',f'{sha}:{rel}')
        if f'新增训练 0 次。{verdict} [逐案例 XY 与指标]' not in data:
            raise ValueError(f'{rel} omits the measured verdict at {sha}')
        for filename in REPORT_FILES:
            if '](%s)'%os.path.relpath(RESULTS/filename,rel.parent) not in data:
                raise ValueError(f'{rel} omits {filename} at {sha}')


def checkout(tmp, tag, base):
    # Share objects with the main checkout but give this job its own index, FETCH_HEAD and files.
    tree=Path(tmp)/tag
    git(REPO,'worktree','add','--detach','--no-checkout',tree,base)
    git(tree,'sparse-checkout','set','--no-cone','--stdin',input=f'/{METHODS}\n/{PROTOCOL}/\n')
    git(tree,'checkout','--detach',base)
    return tree


def deliver():
    verdict=gather()
    if git(REPO,'branch','--show-current')!=BRANCH:
        raise ValueError(f'Main checkout is no longer on {BRANCH}')
    # Keep a failed transaction's detached commit and directory for inspection.
    tmp=Path(tempfile.mkdtemp(prefix='sttw-pcrp-decision-final-'))
    mark('publishing', transaction=str(tmp), export=str(OUT))
    remote=checkout(tmp,'remote',read(STATE/'final_pin.json')['commit'])
    allowed=('https://github.com/QaQaaa-zzz/STTW_CONTROL.git','git@github.com:QaQaaa-zzz/STTW_CONTROL.git',
             'ssh://git@github.com/QaQaaa-zzz/STTW_CONTROL.git')
    if any(git(remote,'remote','get-url',*options,'origin') not in allowed for options in ((),('--push',))):
        raise ValueError('Origin fetch or push URL differs from the selected STTW repository')
    git(remote,'fetch','--no-tags','origin',REMOTE_REF)
    base=git(remote,'rev-parse','FETCH_HEAD')
    git(remote,'checkout','--detach',base)
    published=stage_commit(remote,base,verdict)
    expected_tree=git(remote,'rev-parse',f'{published}:{RESULTS}')
    verify_commit(remote,published,expected_tree,verdict)
    mark('remote_prepared',remote_parent=base,remote_result_commit=published,results_tree=expected_tree)
    if published!=base:
        git(remote,'-c','http.proxy=','-c','http.version=HTTP/1.1','-c','http.postBuffer=52428800','push','origin',f'{published}:{REMOTE_REF}')
    # Fetch the actual named branch and read both full reports out of its Git tree.
    git(remote,'fetch','--no-tags','origin',REMOTE_REF)
    verified=git(remote,'rev-parse','FETCH_HEAD')
    if git(remote,'ls-remote','--exit-code','origin',REMOTE_REF).split()[0]!=verified:
        raise RuntimeError('Remote branch changed during post-push verification')
    git(remote,'merge-base','--is-ancestor',published,verified)
    verify_commit(remote,verified,expected_tree,verdict)
    mark('remote_verified',remote_verified_commit=verified,remote_branch=BRANCH)
    parent=git(REPO,'rev-parse','HEAD')
    local=checkout(tmp,'local',parent)
    local_sha=stage_commit(local,parent,verdict)
    verify_commit(local,local_sha,expected_tree,verdict)
    mark('local_prepared',local_result_commit=local_sha)
    if local_sha!=parent:
        # Refuse overlapping local changes and preserve the independent result commit on failure.
        git(REPO,'merge','--ff-only','--no-edit',local_sha)
    if git(REPO,'rev-parse','HEAD')!=local_sha:
        raise ValueError('Main branch advanced during local integration')
    verify_commit(REPO,local_sha,expected_tree,verdict)
    for name in REPORT_FILES:
        if (REPO/RESULTS/name).read_bytes()!=(OUT/name).read_bytes():
            raise ValueError('Main checkout report mismatch: '+name)
    for tree in (remote,local):git(REPO,'worktree','remove',tree)
    tmp.rmdir()
    report=f'https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/{verified}/{RESULTS}/README.md'
    counts=read(ROOT/'decision_report/summary.json')
    msg=f'PCRP 判决实验已完成并核验远端 {verified}，本地 {local_sha}；科学工况 9232，常驻复验 {counts.get("resident_probe_rollouts",0)}，主预算实耗 {counts["main_rollouts"]}，另列672，0训练。{verdict} 逐案结果 {report}'
    record('最终报告发布并核验',msg+' 外盘 '+str(ROOT/'decision_report/README.md'))
    bark('final','complete',msg)
    mark('complete',remote_report=report,verdict_zh=verdict,main_rollouts=counts['main_rollouts'],diagnostic_rollouts=672,
         completion_receipt=str(STATE/'final_complete_receipt.json'))


def error(message):
    mark('error',error=message)
    try:
        record('最终汇总或发布异常',message+' 状态 '+str(STATUS))
    finally:
        bark('final','error','PCRP 汇总或发布遇到问题：'+message+'。请查看 '+str(STATUS))
        mark('error',error_receipt=str(STATE/'final_error_receipt.json'))


if __name__=='__main__':
    action=sys.argv[1:]
    if action in ([],['run']):
        try:
            deliver()
        except (Exception, KeyboardInterrupt) as exc:
            error(f'{type(exc).__name__}: {exc}')
            raise
    elif action==['stopped']:
        last=read(STATUS) if STATUS.exists() else {}
        if os.environ.get('SERVICE_RESULT')!='success' or last.get('stage')!='complete':
            error(last.get('error') or f"systemd {os.environ.get('SERVICE_RESULT','unknown')} / {os.environ.get('EXIT_STATUS','')}")
    else:
        raise SystemExit('usage: finish_paired.py [run|stopped]')
