"""Run an explicit slice of declared roots, batching downstream tasks per root."""
import argparse
import os
import json
import subprocess
import sys
import time
from pathlib import Path
os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH','')


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--group', default='nominal')
p.add_argument('--start', type=int, default=0)
p.add_argument('--count', type=int, required=True)
p.add_argument('--analysis-python', required=True)
p.add_argument('--parallel-tasks', action='store_true')
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
cases = manifest[a.group]
if a.start < 0 or a.count <= 0 or a.start + a.count > len(cases):
    p.error('slice must be inside the declared manifest group')
selected = cases[a.start:a.start + a.count]
if a.parallel_tasks and any(c.get('source_only', False) for c in selected):
    p.error('source-only collection does not use parallel downstream tasks')
status_path = a.manifest.parent / f'{a.group}_collection_status.json'
started = time.time()
completed = []
for case in selected:
    output = Path(case['output'])
    # The collector owns output creation; never reuse or overwrite a prior run.
    log_path = output.parent / (output.name + '_runtime.log')
    status_path.write_text(json.dumps(dict(status='running', origin_id=case['origin_id'],
        started_unix=started, completed=completed), indent=2) + '\n')
    command = [sys.executable, 'learning/cli/collect_connection_candidates.py',
        '--config', case['config'], '--output', str(output)]
    if case.get('source_snapshot'):
        command.extend(['--source-snapshot', case['source_snapshot']])
    if case.get('source_only', False):
        command.append('--source-only')
    if a.parallel_tasks:
        command.append('--parallel-tasks')
    with log_path.open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status_path.write_text(json.dumps(dict(status='failed', origin_id=case['origin_id'],
            returncode=result.returncode, completed=completed), indent=2) + '\n')
        sys.exit(result.returncode)
    outcome = json.loads((output / 'status.json').read_text())
    # Failed upstream roots remain in the declared denominator, with their XY trace.
    plot = [a.analysis_python, 'learning/cli/plot_connection_candidates.py', '--run', str(output)]
    if case.get('source_only', False) or outcome['status'] == 'source_failed':
        plot.append('--source-only')
    subprocess.run(plot, check=True)
    if 'envelope_half_width_m' in case and outcome['status'] == 'complete':
        spec = json.loads(Path(case['config']).read_text())
        for task in spec['tasks']:
            subprocess.run([a.analysis_python, 'learning/cli/evaluate_connection_envelope.py',
                '--task-run', str(output / task['task_id']),
                '--half-width', str(case['envelope_half_width_m'])], check=True)
    completed.append(dict(origin_id=case['origin_id'], output=str(output), outcome=outcome))
status_path.write_text(json.dumps(dict(status='completed_slice', start=a.start, count=a.count,
    started_unix=started, finished_unix=time.time(), completed=completed), indent=2) + '\n')
