"""Freeze new maps and roots before any TEST execution; reuse for E2/E3."""
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-config',type=Path,required=True)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=json.loads(a.source_config.read_text());world=base['world_map_spec'];rng=np.random.default_rng(2026111401)
roots=[dict(root_variant=1001+i,start_y_m=float(y),start_yaw_rad=float(yaw),start_roll_rad=float(roll)) for i,(y,yaw,roll) in enumerate(rng.uniform([-.05,-.03,-.015],[.05,.03,.015],(4,3)))]
# Model-specific DEV calibration is frozen before generating any new outcome.
cal=json.loads(a.calibration.read_text());frozen={}
for name in ['M0','M1','M2']:
    rows=[r for r in cal if r['model']==name]
    frozen[name]=dict(edges=rows[0]['edges'],axis_margin_m=np.max([r['axis_margin_m'] for r in rows],axis=0).tolist(),source=str(a.calibration))
(a.output/'calibration.json').write_text(json.dumps(frozen,indent=2)+'\n')
entries=[]
for i in range(50):
    folder=a.output/f'map{i:02d}';folder.mkdir();w=copy.deepcopy(world);w['map_id']=f'chapter3_test_{i:02d}'
    w['terrain']['seed']=20261200+i
    for ridge,offset in zip(w['terrain']['ridges'],rng.uniform(-.4,.4,2)):
        ridge['x_interval_m']=[float(x+offset) for x in ridge['x_interval_m']]
    half=float(rng.uniform(1.3,1.7));w['bounds_xy_m']=[-5.,25.,-half,half]
    pre=float(rng.uniform(5.,5.9));post=float(rng.uniform(13.,13.35));yp=float(rng.uniform(-.15,.15));yq=float(rng.uniform(-.08,.08));width=float(rng.uniform(.10,.18))
    w['terrain']['forbidden_rectangles']=[[pre,pre+.2,yp-width/2,yp+width/2],[post,post+.2,yq-.06,yq+.06]];w['terrain']['obstacle_height_m']=.5
    (folder/'world.json').write_text(json.dumps(w,indent=2)+'\n');c=copy.deepcopy(base)
    for field in ['world_map_spec','ditch_terrain','snapshot_times_s','evaluation_lanes']:c.pop(field,None)
    template=copy.deepcopy(base['ground_handoff']['handoff_variants'][0]);variants=[]
    for root in roots:
        for speed in [1.8,1.9]:
            v=copy.deepcopy(template);v.update(root,data_split='TEST',requested_speed_m_s=speed,approach_speed_m_s=speed,continuation_speed_m_s=speed);variants.append(v)
    c['ground_handoff']['handoff_variants']=variants;c.update(world_map=str(folder/'world.json'),planned_activation_offset_m=-1.75,task_evaluation=True,role='Frozen TEST map/root; no tuning on outcomes')
    (folder/'config.json').write_text(json.dumps(c,indent=2)+'\n');entries.append(dict(map_id=i,config=str(folder/'config.json'),world_sha256=hashlib.sha256((folder/'world.json').read_bytes()).hexdigest()))
protocol=dict(role='TEST, new terrain seeds/layouts and complete starting roots, shared map family with development',seed=2026111401,entries=entries,roots=roots,paired_tasks=200,root_independence='Four fixed new roots reused across50 maps; statistical resampling unit is map, not trajectory window',source_config=str(a.source_config),calibration_source=str(a.calibration),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physics='Unchanged RB225/H75 alpha0, locked ground hips/knees,1ms physics/5ms control/20ms policy;2cm/4cm ridges,1-4mm stones,ditch0.3m,2m flat banks',E3=dict(primary='M1 versus M0: admitted-plan physical failure fraction and paired full-task success',secondary='M2 versus M1 and M0; penalized and common-success paired completion times',methods='Same18 route/speed candidates,TRAIN approach/RB prefix, predicted departure with terminal hold and domain checks; max-over-DEV regional margins; static selection before initialization',budget='50 layouts x4 roots x3 methods=600 conditions; identical selected plans execute once and remain paired',intervals='Map-group bootstrap95%,10000 resamples,seed2026111402; report numerators/denominators and no-plan separately',stop='No tuning or retraining using TEST outcomes; preserve all failures;120min execution cap separate from30min assistant wait limit'),scope='First frozen E3 component test. E1/E2 and rolling validity remain required; no claim that this static comparison completes outline P4.',status='frozen_before_execution')
(a.output/'catalog.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(dict(maps=len(entries),roots=roots,method_conditions=600),indent=2))
