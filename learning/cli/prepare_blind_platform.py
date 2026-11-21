"""Freeze a two-map Chapter 3 TEST and transfer the DEV handback prior onto its published geometry."""
import argparse
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from evaluate_platform_review import decisions, fit_hidden, hand_state, initial_features, tensor
from evaluate_playground_handoff import transplant
from predict_platform_departures import physical_raster
from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import save
from sttw_control.playground_history_model import GroundPredictionState, load_history_model
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.playground_terrain_features import TerrainFeatures


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(wide,out):
    old=wide/'review_114';s1=wide/'chapter_mainline/s1'
    parity=pd.read_csv(s1/'common_bank_parity.csv').iloc[0]
    if parity['n']!=2160 or parity.oracle_diff or parity.physical_outcome_diff:
        raise ValueError(f'Original-bank replay baseline requires review: {parity.to_dict()}')
    latency=pd.read_csv(s1/'latency.csv').query("method=='phase2_relative'").set_index('candidates')
    if set(latency.index)!={3,9} or not (latency.n==1000).all():raise ValueError('Incomplete single-robot latency benchmark')
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'test_freeze_manifest.json'
    if manifest.exists():raise FileExistsError(manifest)
    plan=read(old/'plan.json');cal=wide/'primary/roots_0008'
    model=Path(read(old/'online/plan.json')['ground_model'])
    configs={skill:read(cal/skill/'declaration.json') for skill in ('source','rb_run1')}
    archived_bank=Path(plan['initial_banks'][0])
    files=[archived_bank.with_name(f'batch_{i:03d}.npz') for i in range(5)]
    arrays=[dict(np.load(f)) for f in files];bank=out/'initial_bank.npz'
    np.savez(bank,**{k:np.concatenate([a[k] for a in arrays]) for k in ('qpos','qvel')})
    rows=list(range(1001,1041))
    packed=np.load(bank)
    train=np.load(wide/'initial_bank.npz')
    seen={tuple(np.r_[q,v]) for q,v in zip(train['qpos'],train['qvel'])}
    ids={tuple(np.r_[packed['qpos'][r],packed['qvel'][r]]) for r in rows}
    if len(ids)!=40 or seen & ids:raise ValueError('Previously used or duplicate TEST initial state')
    maps={}
    for label,seed,dx,dy in [('test_a',520731,-.12,.035),('test_b',860917,.16,-.045)]:
        world=read(Path(plan['maps']['primary']));world['map_id']=label;terrain=world['terrain']
        terrain['rough_source']['seed']=seed
        walls=[[x0+dx,x1+dx,y0+dy,y1+dy] for x0,x1,y0,y1 in terrain['forbidden_rectangles']]
        terrain['forbidden_rectangles']=walls;terrain['rough_source']['forbidden_rectangles']=walls
        where=out/(label+'.json');save(where,world);maps[label]=dict(path=str(where),sha256=sha(where),seed=seed)
    phys_hashes={s:{k:v for k,v in dc['skills'][s].items() if k in ('actor_sha256','normalizer_sha256','checkpoint_identity_sha256','source_training_run_id')} for s,dc in configs.items()}
    prog=Path(__file__).resolve().parent
    code=[prog/x for x in ('prepare_blind_platform.py','run_blind_platform.py','report_blind_platform.py','report_platform_online_review.py',
                           'run_platform_online_review.py','report_platform_controls.py','evaluate_playground_handoff.py')]
    code += [prog.parent/'src/sttw_control'/x for x in ('platform_online.py','playground_handoff_evaluation.py')]
    inputs=[old/'physical_scores.csv',s1/'latency.csv',s1/'common_bank_summary.csv',model/'best.pt',model/'nominal.npz',
            wide/'primary/departure_bank.npz',cal/'rb_run1.json',cal/'source.json',Path('learning/configs/playground_rb_skill_sources.json')]
    cfg=dict(role='independent frozen TEST',wide=str(wide),maps=maps,roots=rows,bank=str(bank),bank_sha256=sha(bank),
             hashes={str(f):sha(f) for f in code+inputs},weights=dict(phys_hashes,H75=configs['source']['ground_model']['source_actor']),
             predictor='phase2_relative',ground_model=str(model),margin={'source':.2105724513530731,'rb_run1':.1017768085002899},
             timings_ms=latency.p95_ms.to_dict(),latency_ticks={phase:math.ceil(latency.at[n,'p95_ms']/5.) for phase,n in [('start',9),('departure',3)]},
             clock='same physical budget for each method. H75 neutral lead-in then selected approach. Straight published exit until candidate release.',
             alpha=0.,speed=2.,forces_N=[0,5,-5],section_x=[12.7,13.6,15.],root_bootstrap_seed=20261120,
             scope='2 layouts x 2 skills x 3 forces x 1320 worlds. 40 roots x (24 planner assignments + nine physical branches). H1 reuses force-zero oracle. 40 independent root clusters.')
    save(manifest,cfg);(out/'test_freeze_manifest.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n')


@torch.inference_mode()
def project(out, initializer_dir=None):
    plan=read(out/'test_freeze_manifest.json');wide=Path(plan['wide']);raw=wide/'review_114';base=wide/'primary/roots_0008'
    captures=wide.parent/'replication/primary'
    schema=read(captures/'source_capture/model_state_schema.json')
    rows=pd.read_csv(raw/'physical_scores.csv').query("layout=='primary' and section=='near' and verified_jump and root<78")
    bank=dict(np.load(wide/'primary/departure_bank.npz'));refs={x.tobytes():i for i,x in enumerate(bank['xy'])}
    model=load_history_model(Path(plan['ground_model']))
    for layout,m in plan['maps'].items():
        dest=out/layout;dest.mkdir(exist_ok=True);world=read(Path(m['path']))
        terrain=world['terrain'];height=terrain['far_bank_height_m']
        xml=read(Path('learning/configs/playground_rb_skill_sources.json'))['model']['path']
        bundle=SimpleNamespace(xml_path=Path(xml))
        physics=transplant(bundle,terrain,dest,5,read(base/'source.json')['ground_physics'])
        physical_raster(dest/'terrain.mjb',dest/'terrain.npz')
        raster=dict(np.load(dest/'terrain.npz'));raster['height']-=height;model.terrain=TerrainFeatures(**raster,device='cuda')
        walls=[i for i in range(physics.ngeom) if physics.geom(i).name.startswith('planning_obstacle_')]
        obs=tensor(physics.geom_pos[walls]),tensor(physics.geom_size[walls]);env=BodyEnvelope(physics,dimensions=3)
        for skill in ('source','rb_run1'):
            g=rows[rows.skill==skill].reset_index(drop=True)
            if initializer_dir is None:
                mu,scale,weights=fit_hidden(captures/(skill+'_capture'),model,schema)
            else:
                with np.load(initializer_dir/(skill+'_initializer.npz')) as saved:
                    mu,scale,weights=(saved[k] for k in ('mu','scale','weight'))
            np.savez(dest/(skill+'_initializer.npz'),mu=mu,scale=scale,weight=weights)
            ps=[];us=[];vs=[];refids=[];speeds=[]
            for r in g.itertuples():
                with np.load(r.trajectory) as z:
                    pose,u,v=hand_state(z,r.handoff_s,schema);ps.append(pose);us.append(u);vs.append(v)
                    refids.append(refs[z['published_reference_xy'].tobytes()]);speeds.append(z['trace'][round(r.handoff_s/.005)+1,18])
            p,u,prev=map(tensor,(ps,us,vs));request=tensor(speeds)
            feat=(initial_features(np.stack(ps),np.stack(us),np.stack(vs))-mu)/scale
            hidden=tensor(np.column_stack((np.ones(len(g)),feat))@weights)
            ends=np.floor((10-g.handoff_s.to_numpy()-.005+1e-6)/.02).astype(int)+1
            route=(tensor(bank['s']),*(tensor(bank[k][refids]) for k in ('xy','heading','curvature')))
            p[:,2]-=height
            f=model.rollout_state(GroundPredictionState(p,u,prev,hidden),*route,int(ends.max())-1,request)
            f['pose'][:,:,2]+=height
            table=decisions(f['pose'],f['velocity'],f['domain'],request,ends,world,.6,model.nominal.follower['preview_body_x_min_m'],env,obs,plan['section_x'])
            table[['root','approach','departure']]=g[['root','approach','departure']]
            for i,k in enumerate(('x','y','z','qw','qx','qy','qz')):table[k]=np.asarray(ps)[:,i]
            table['speed']=np.asarray(us)[:,0]
            table.to_csv(dest/(skill+'_prior.csv'),index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','prior']);p.add_argument('wide',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args()
    if a.stage=='freeze':freeze(a.wide,a.out)
    else:project(a.out)
