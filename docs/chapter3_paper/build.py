"""Compile the Chapter 3 English draft and inventory its rendered inputs."""
import argparse,json,re,subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine',default=str(Path.home()/'.local/bin/tectonic'))
a=p.parse_args();here=Path(__file__).resolve().parent
source=(here/'MANUSCRIPT.tex').read_text()
paths=[here/'../evidence/chapter3_event_outputs',here/'../evidence/chapter3_offroad_event_training',here/'../evidence/chapter3_offroad_distance_online/plots']
figures=re.findall(r'\\includegraphics\[[^]]+\]\{([^}]+)\}',source)
resolved=[next((d/name for d in paths if (d/name).is_file()),None) for name in figures]
assert all(resolved),'Missing figure input'
subprocess.run([a.engine,'MANUSCRIPT.tex','--outdir','.', '--keep-logs'],cwd=here,check=True)
info=subprocess.check_output(['pdfinfo',str(here/'MANUSCRIPT.pdf')],text=True)
pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
log=(here/'MANUSCRIPT.log').read_text();assert 'Overfull' not in log,'Inspect overflowing manuscript layout'
assert pages<=8,'Draft exceeds RA-L maximum format length'
(here/'build_receipt.json').write_text(json.dumps(dict(pages=pages,format='US Letter, official ieeeconf initial-submission template',figures=[str(x.relative_to(here.parent)) for x in resolved],overfull_boxes=False,submission_ready=False,scope='Compilation and layout checks; not scientific or human approval'),indent=2)+'\n')
print(f'Built {pages}-page draft with {len(figures)} figures; no overfull boxes.')
