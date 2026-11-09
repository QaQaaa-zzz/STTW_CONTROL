"""Freeze the first twenty preassigned training roots at the short station."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = json.loads(a.source.read_text())
roots = list(dict.fromkeys(c['origin_id'] for c in source['cases']))[:20]
selected = [c for c in source['cases'] if c['origin_id'] in roots and c['distance_m'] == 5.6]
assert len(roots) == 20 and len(selected) == 120
assert all(c['split'] == 'train' for c in selected)
a.output.mkdir(parents=True, exist_ok=False)
cases = []
for case in selected:
    spec = json.loads(Path(case['config']).read_text())
    spec.update(output=str(a.output / case['case_id']),
                role='A2 short-station TRAIN; natural full prefix and frozen update2 suffix; not independent TEST')
    path = a.output / (case['case_id'] + '.json')
    path.write_text(json.dumps(spec, indent=2) + '\n')
    cases.append(dict(case, config=str(path)))
manifest = dict(source, cases=cases, status='prepared_not_started',
    source_manifest=str(a.source), source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    roots=roots, optimizer_updates=0, physics_lane_steps_max=120*8800,
    protocol='20 preassigned train roots x 6 actions, one 5.6m platform layout, 22s full-chain window; no snapshot restore, no root replacement, retain upstream failures',
    scope='A2 short120 with observed downstream labels; A0 snapshot audit deferred by user throughput priority; A1/A3/A4 not launched',
    estimate_wall_hours=[1, 3])
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(cases=len(cases), roots=len(roots), output=str(a.output))))
