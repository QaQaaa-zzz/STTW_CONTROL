from pathlib import Path
import json,csv
from sttw_control.route_pebbles import stone_layout
r=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_selector_layout_test')
for seed in [20261111,20261112,20261113,20261114]:
 c=json.loads((r/f'layout{seed}_height10mm_scene.json').read_text());c.pop('reference_layout_csv');stones=stone_layout(c);stones[:,1]*=.25
 with (r/f'layout{seed}.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','s_m','n_m','radius_s_m','radius_n_m'])
  for i,row in enumerate(stones):w.writerow([i,*row])
