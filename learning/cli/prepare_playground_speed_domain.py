"""Freeze neighboring-speed full-chain validation without opening planner commands."""
import argparse,copy,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--start-sweep',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase-switches',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=json.loads((a.source/'resolved_config.json').read_text());source_variants=base['ground_handoff']['handoff_variants'];entries=[]
source_manifest=json.loads((a.start_sweep/'manifest.json').read_text());baseline=Path(source_manifest['entries'][1]['evaluation']);baseline_config=json.loads((baseline/'resolved_config.json').read_text());assert abs(baseline_config['ground_handoff']['approach']['jump_start_x_m']-8.55)<1e-5
pairs=[(1.8,1.9),(1.9,1.8)] if a.phase_switches else [(s,s) for s in [1.7,1.8,1.9]]
for speed,after in pairs:
    name=f'speed{speed:.1f}_to{after:.1f}' if a.phase_switches else f'speed{speed:.1f}';folder=a.output/name;folder.mkdir();c=copy.deepcopy(base)
    for key in ['world_map_spec','ditch_terrain','snapshot_times_s']:c.pop(key,None)
    c.update(role='P3 neighboring-speed DEV/TRAIN coverage; not independent TEST',planned_activation_offset_m=8.55-10.3,task_evaluation=True,audit_actuator_owner=False)
    variants=[copy.deepcopy(v) for v in source_variants if a.phase_switches or speed!=1.8 or v['root_variant'] not in [0,3]]
    for v in variants:v.update(requested_speed_m_s=speed,approach_speed_m_s=speed,continuation_speed_m_s=after)
    c['ground_handoff']['handoff_variants']=variants;c['evaluation_lanes']=[[route,'h75',i] for i in range(len(variants)) for route in range(9)]
    (folder/'config.json').write_text(json.dumps(c,indent=2)+'\n');entries.append(dict(layout=name,speed_m_s=speed,continuation_speed_m_s=after,config=str(folder/'config.json'),evaluation=str(folder/'evaluation'),episodes=len(c['evaluation_lanes'])))
manifest=dict(role='P3 adjacent speed validation;54TRAIN+18reusedDEV-root conditions per speed',speeds_m_s=[1.7,1.8,1.9],entries=entries,routes=list(range(9)),seconds=10,ground_only=False,physical_episodes=198,reused_episodes=18,baseline_reuse=dict(evaluation=str(baseline),speed_m_s=1.8,roots=[0,3]),budget='198new physical episodes in3GPU batches(72,54,72),0training,one queue',expected_wallclock_s=[240,900],hard_limit_s=1500,stop='First runtime error or25min, no retries',acceptance='All72complete original10s task with full RB and0.5s goal hold for a discrete speed; otherwise keep it out of validated command domain. No continuous interval claim.',fixed='Original rough map/ditch and2m flat buffers, RB225/H75alpha0, initialspeed from resolved start_vx_m_s with existing yaw, activation8.55, sharedphysics/goal unchanged',change='Only ground requested speed before and after jump; native RB unchanged. Initial condition does not equal speed request, so transient tracking is included.',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
if a.phase_switches:
    manifest.update(role='P3 phase-specific speed command DEV/TRAIN validation',speeds_m_s=[1.8,1.9],phase_pairs=pairs,physical_episodes=144,reused_episodes=0,baseline_reuse=None,budget='144physical episodes in2GPU batches,0training',change='Only approach and handback speed requests differ; speed changes on measured RB handback, native RB unchanged',acceptance='Each ordered phase-speed pair must complete all72 original tasks; no arbitrary ground-time switching or interpolation claim.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(a.output/'source').mkdir()
for path in [Path(__file__),Path('learning/src/sttw_control/playground_handoff_evaluation.py')]:shutil.copyfile(path,a.output/'source'/path.name)
print(json.dumps(dict(new=manifest['physical_episodes'],reused=manifest['reused_episodes'],batches=[e['episodes'] for e in entries])))
