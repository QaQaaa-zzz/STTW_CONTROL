from pathlib import Path
import json
r=Path(__file__).parent
original=json.loads(Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_dense_corridor_ridge10/dense_corridor/geometry_audit.json').read_text())
records=[]
for h in [0,5,10,20]:
 a=json.loads((r/'terrains'/f'height{h:02d}mm'/'geometry_audit.json').read_text())
 assert a['center_radius_sha256']==original['center_radius_sha256']
 assert a['stones']==original['stones'] and a['upward_triangles'] and a['zero_area_triangles']==0
 assert abs(a['height_max_m']-h/1000)<.00101
 records.append(dict(height_mm=h,**a))
(r/'geometry_validation.json').write_text(json.dumps(records,indent=2))
print('Four matched layouts verified: same centers/radii; height is the changed geometry parameter',flush=True)
