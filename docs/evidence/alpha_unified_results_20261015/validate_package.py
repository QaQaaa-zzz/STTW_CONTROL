"""Validate published numerical grain, unaltered samples, images and links."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
from PIL import Image
P=Path(__file__).parent
jsonout=P/'package_validation.json';jsonout.write_text('{"state":"checking"}\n')
def rows(name):return list(csv.DictReader((P/name).open()))
audit=rows('reward_audit.csv');conditions=rows('condition_metrics.csv');gaps=rows('preference_metrics.csv');stages=rows('checkpoint_summary.csv')
assert len(audit)==len(conditions)==300 and len(gaps)==20 and len(stages)==10
assert len({(r['arm'],r['update'],r['alpha'],r['case']) for r in conditions})==300
assert max(float(r['max_reward_error']) for r in audit)<2e-6
assert all(r['steps']=='4000' and r['goal']=='True' for r in conditions)
assert len([r for r in conditions if r['arm']=='warm' and r['update']=='150' and r['strict_hold_1s']=='False'])==15
assert len([r for r in conditions if r['arm']=='warm' and r['update']=='200' and r['strict_hold_1s']=='False'])==1
assert all(r['strict_hold_1s']=='True' for r in conditions if r['arm']=='scratch' and int(r['update'])>=100)
assert sum(r['strict_hold_1s']=='False' for r in conditions if r['arm']=='scratch' and r['update']=='0')==30
assert sum(r['strict_hold_1s']=='False' for r in conditions if r['arm']=='scratch' and r['update']=='50')==10
assert [r['update'] for r in stages if r['selected']=='True' and r['arm']=='warm']==['0']
assert [r['update'] for r in stages if r['selected']=='True' and r['arm']=='scratch']==['100']
verify=json.loads((P/'verification.json').read_text());assert hashlib.sha256((P/'curve_data.npz').read_bytes()).hexdigest()==verify['curve_npz_sha256']
curve=np.load(P/'curve_data.npz');checked=0
for r in audit:
    if int(r['update']) not in ([0,200] if r['arm']=='warm' else [100,200]):continue
    token={'0.0':'0','0':'0','0.25':'0p25','0.5':'0p5','0.75':'0p75','1.0':'1','1':'1'}[r['alpha']]
    prefix=f"{r['arm']}_u{r['update']}_a{token}_{r['case']}__";source=dict(np.load(r['source_npz']))
    for key in curve.files:
        if key.startswith(prefix):
            field=key[len(prefix):];value=curve[key];assert np.isfinite(value).all()
            if field=='reward':np.testing.assert_allclose(value,source['scored_tick_reward'],rtol=0,atol=2e-6)
            else:np.testing.assert_array_equal(value,source[field])
    assert len(curve[prefix+'time'])==4000;checked+=1
assert checked==120
pngs=list(P.glob('*.png'));assert len(pngs)==34
for p in pngs:
    with Image.open(p) as im:im.verify()
links=0
for p in P.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http:','https:','#')):continue
        assert (p.parent/target.split('#')[0]).exists(),target;links+=1
for p in P.glob('*.csv'):assert b'\r' not in p.read_bytes()
assert not any(p.suffix.lower() in ('.pt','.pkl','.zip','.pdf') for p in P.iterdir())
result=dict(state='passed',condition_rows=300,unique_conditions=300,preference_rows=20,candidate_rows=10,unmodified_full_resolution_policy_curves=checked,png_decodes=len(pngs),local_markdown_links=links,csv_line_endings='LF',source_reward_max_abs_error=verify['reward_max_abs_error'],new_physics=0,new_training_updates=0,files=len(list(P.iterdir())),bytes=sum(p.stat().st_size for p in P.iterdir() if p.is_file()),no_models_or_full_logs=True)
jsonout.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
