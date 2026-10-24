"""Archive named historical middle policies without retargeting original files."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(root):
    repo = Path.cwd().resolve()
    distill = Path('/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021')
    dual = Path('/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_coverage_v2_resume_20261011')
    dm = json.loads((distill/'manifest.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    (root/'objects').mkdir()
    objects, archives, records = {}, {}, []

    def put(file, expected=None):
        file = Path(file).resolve()
        identity = sha(file)
        if expected and expected != identity:
            raise ValueError('Identity differs: '+str(file))
        dest = root/'objects'/identity
        if not dest.exists():
            shutil.copyfile(file, dest)
            if sha(dest) != identity:
                raise ValueError('Copy verification failed')
            dest.chmod(0o444)
        item = objects.setdefault(identity, dict(path=str(dest.relative_to(root)), bytes=dest.stat().st_size, origins=[]))
        if str(file) not in item['origins']:
            item['origins'].append(str(file))
        return dict(original=str(file), object=identity, sha256=identity)

    def metadata(file):
        file = Path(file)
        return put(file) if file.exists() else None

    def code(commit):
        commit = subprocess.check_output(['git', 'rev-parse', commit+'^{commit}'], text=True).strip()
        if commit not in archives:
            target = root/('code_'+commit+'.tar.gz')
            raw = subprocess.check_output(['git', 'archive', '--format=tar', commit, 'learning'])
            with target.open('xb') as stream:
                with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
                    zipped.write(raw)
            archives[commit] = dict(path=target.name, sha256=sha(target), scope='Tracked learning source/config/CLI/tests at declared Git commit')
            target.chmod(0o444)
        return commit

    def add(identity, label, parents, model, manifest_path, *, checkpoint_kind='PPO_state', extra=None, status='historical_reference'):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        actual = manifest.get('declaration', manifest)
        commit = manifest.get('code_revision', actual.get('code_revision'))
        record = dict(id=identity, label=label, parents=parents, status=status,
            actor=put(model['actor'], model.get('actor_sha256')),
            checkpoint=put(model['checkpoint'], model.get('checkpoint_sha256')),
            checkpoint_kind=checkpoint_kind, update=model.get('source_update'),
            manifest=put(manifest_path), code_commit=code(commit), evidence=[], runtime_overlays=[])
        for file in extra or []:
            stored = metadata(file)
            if stored:
                record['evidence'].append(stored)
        plan = manifest.get('plan', {})
        for name in ['config', 'declaration', 'parent_manifest']:
            if isinstance(plan.get(name), str):
                stored = metadata(plan[name])
                if stored:
                    record['evidence'].append(stored)
        config = manifest.get('config', actual.get('config'))
        if isinstance(config, dict):
            config_path = root/(identity+'_resolved_config.json')
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
            record['resolved_config'] = dict(path=config_path.name, sha256=sha(config_path))
            config_path.chmod(0o444)
        # Preserve exact declared runtime files, including formerly untracked code.
        declarations = [manifest]
        if isinstance(plan.get('declaration'), str):
            declarations.append(json.loads(Path(plan['declaration']).read_text()))
        for declaration in declarations:
            for file, expected in declaration.get('source_sha256', {}).items():
                if Path(file).suffix != '.py':
                    continue
                if Path(file).exists() and sha(file)==expected:
                    record['runtime_overlays'].append(put(file, expected))
                else:
                    record['runtime_overlays'].append(dict(original=file, expected_sha256=expected,
                        status='HISTORICAL_DECLARATION_ONLY; declared Git archive preserved, live file differs'))
        records.append(record)

    dual_ids = []
    for teacher in dm['teachers']:
        identity = 'STTW_MIDDLE_DUAL_COVERAGE_U200_A'+str(teacher['alpha'])
        dual_ids.append(identity)
        add(identity, '双网络：独立α'+str(teacher['alpha'])+'端点', [], teacher, dual/'manifest.json',
            extra=[dual/f"alpha{teacher['alpha']}/best_task_model.json", dual/f"alpha{teacher['alpha']}/metrics.jsonl"])
    bc_id = 'STTW_MIDDLE_DISTILLED_BC'
    add(bc_id, '双端点蒸馏的单网络，尚未PPO微调', dual_ids,
        dict(actor=str(distill/'distillation/student_actor.pkl'), checkpoint=str(distill/'distillation/student.pt')),
        distill/'manifest.json', checkpoint_kind='supervised_actor_not_PPO_resume',
        extra=[distill/'distillation_gate.json', distill/'distillation_reward_http_verified.json'])
    single = 'STTW_MIDDLE_DISTILLED_V3_U325'
    for update, identity, label, parents in [
        (325, single, '蒸馏＋V3课程PPO的开发候选u325', [bc_id]),
        (400, 'STTW_MIDDLE_DISTILLED_V3_U400_FINAL', '同一400轮训练终点，保留波动/负结果', [single])]:
        add(identity, label, parents,
            dict(actor=str(distill/f'training/checkpoints/actor_{update:04d}.pkl'), checkpoint=str(distill/f'training/checkpoints/update_{update:04d}.pt'), source_update=update),
            distill/'manifest.json', extra=[distill/'training/manifest.json', distill/'training/best_stage_3.json', distill/'training/last_completed.json', distill/'training/metrics.jsonl'])
    for name, identity, label, parents in [
        ('preference_recovery_train_v1', 'STTW_MIDDLE_SNIPPET_PPO25_U27', '原短恢复训练支路：2预热＋25PPO', [single]),
        ('preference_full_tail_v1', 'STTW_MIDDLE_FULLTAIL_PPO25_U27', '完整尾段训练：2预热＋25PPO', [single]),
        ('preference_full_tail_length75_v1', 'STTW_MIDDLE_FULLTAIL_PPO75_U77', 'H75：完整尾段25PPO后精确续到75PPO', ['STTW_MIDDLE_FULLTAIL_PPO25_U27'])]:
        run = repo/'runs'/name/'uniform'
        last = json.loads((run/'last_completed.json').read_text())
        update = last['update']
        add(identity, label, parents, dict(actor=str(run/f'checkpoints/actor_{update:04d}.pkl'),
            checkpoint=last['checkpoint'], checkpoint_sha256=last['checkpoint_sha256'], source_update=update),
            run/'manifest.json', extra=[run/'effective_config.json', run/'plan.json', run/'metrics.jsonl', run/'last_completed.json'],
            status='frozen_primary_comparator' if update==77 else 'historical_reference')
    teaching = repo/'runs/preference_witness_selection_v1/witness_918501'
    add('STTW_MIDDLE_WITNESS_S918501_SUP400', '恢复见证教学分支，非H75后继', [single],
        json.loads((teaching/'training/candidate_record.json').read_text()), teaching/'training/manifest.json',
        checkpoint_kind='supervised_actor_only_not_PPO_resume',
        extra=[teaching/'declaration.json', teaching/'training/metrics.jsonl', teaching/'training/final_fit.json'])
    lower = dm['source']['candidate']
    common = {k:put(lower[k], lower.get(k+'_sha256', lower.get('config_file_sha256') if k=='config' else None)) for k in ['actor','checkpoint','config']}
    common['source'] = dm['source']
    # Capture the present learning tree too, including untracked reusable modules.
    runtime = root/'freeze_time_learning_tree.tar.gz'
    with tarfile.open(runtime, 'w:gz') as tar:
        for file in sorted((repo/'learning').rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix in ['.py','.json','.toml','.yaml','.yml','.txt','.md']:
                tar.add(file, arcname=str(file.relative_to(repo)))
    runtime.chmod(0o444)
    git = root/'freeze_time_git_state.txt'
    git.write_text(subprocess.check_output(['git','status','--short'],text=True)+'\n'+subprocess.check_output(['git','diff','--binary','--','learning'],text=True))
    git.chmod(0o444)
    document = dict(schema='sttw_frozen_middle_lineage_v1', created_on='2026-10-24',
        original_files_untouched=True, no_training_or_physics=True,
        current_primary_id='STTW_MIDDLE_FULLTAIL_PPO75_U77', nodes=records,
        common_frozen_lower=common, code_archives=archives, objects=objects,
        freeze_time_source=dict(path=runtime.name, sha256=sha(runtime), role='Actual current tree incl untracked modules; not evidence every historical run executed these bytes'),
        freeze_time_git_state=dict(path=git.name,sha256=sha(git)),
        limits=['Freezing is preservation, not adoption or a new success claim.',
                'Git archives preserve declared versions; exact runtime overlays separately identify recorded files.',
                'External physics assets/environment dependencies remain referenced by original source identity.',
                'Exact learner resume is not frame-continuous physical state resume.'])
    path = root/'manifest.json'
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    path.chmod(0o444)
    return document


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for identity, obj in manifest['objects'].items():
        if sha(root/obj['path']) != identity:
            raise ValueError('Frozen object changed '+identity)
    nodes = manifest['nodes']
    ids = {node['id'] for node in nodes}
    for node in nodes:
        if any(parent not in ids for parent in node['parents']):
            raise ValueError('Missing lineage parent')
        for field in ['actor','checkpoint']:
            if sha(node[field]['original']) != node[field]['sha256']:
                raise ValueError('Original changed '+node['id'])
        if node.get('resolved_config') and sha(root/node['resolved_config']['path']) != node['resolved_config']['sha256']:
            raise ValueError('Frozen config changed')
    for obj in [*manifest['code_archives'].values(),manifest['freeze_time_source'],manifest['freeze_time_git_state']]:
        if sha(root/obj['path']) != obj['sha256']:
            raise ValueError('Source archive changed')
    print(json.dumps(dict(nodes=len(nodes),objects=len(manifest['objects']),manifest_sha256=sha(root/'manifest.json'),
        all_frozen_and_original_hashes_match=True,physical_steps=0,training_steps=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if not args.verify:
        freeze(args.root.resolve())
    verify(args.root.resolve())
