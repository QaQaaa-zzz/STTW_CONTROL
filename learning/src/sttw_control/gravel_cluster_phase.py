"""First actual loaded contact per cap-owner cluster, from native patch points."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .route_pebbles import project_route


def cluster_contact_onsets(raw,contract,cases,stones_per_cluster):
    raw=Path(raw);meta=json.loads((raw/'index.json').read_text());n=contract['settings']['num_envs'];count=cases[0]['lane_count'];origins=np.asarray(meta['origins']);onsets=np.full((n,2,3),np.inf)
    route=np.load(Path(contract['bundle'])/'route.npz');paths=list(dict.fromkeys(c['terrain_path'] for c in cases));terrain_ids=np.array([paths.index(c['terrain_path']) for c in cases for _ in range(count)])
    layouts=[];interpolators=[]
    for path in paths:
        path=Path(path);layouts.append(np.loadtxt(path/'stone_centers.csv',delimiter=',',skiprows=1,usecols=(1,2,3,4),ndmin=2))
        with np.load(path/'terrain.npz') as f:interpolators.append(RegularGridInterpolator((f['grid_x'],f['grid_y']),f['height_field']*.001,bounds_error=False,fill_value=0))
    for chunk in meta['chunks']:
        with np.load(raw/chunk['file']) as f:d={k:f[k] for k in f.files}
        ticks=len(d['physics_step']);rr=np.arange(ticks)[:,None,None]
        for wi,wheel in enumerate(('frontwheel','rearwheel')):
            counts=d[wheel+'_normal_count'][:,:,0];starts=d[wheel+'_normal_start'][:,:,0];width=int(counts.max(initial=0));ix=np.clip(starts[:,:,None]+np.arange(width),0,d[wheel+'_normal_force'].shape[1]-1)
            force=d[wheel+'_normal_force'][rr,ix,0];point=d[wheel+'_normal_point'][rr,ix]-origins[None,:,None,:]
            valid=(np.arange(width)<counts[:,:,None])&(force>.01)&(point[:,:,:,2]>.00025)&d['alive_before'][:,:,None]
            ti,lane,patch=np.nonzero(valid)
            for terrain_id in range(len(paths)):
                use=terrain_ids[lane]==terrain_id;xy=point[ti[use],lane[use],patch[use],:2];hit=interpolators[terrain_id](xy)>.0005;xy=xy[hit];lanes=lane[use][hit];times=d['physics_step'][ti[use][hit]]*meta['physics_dt_s']
                if not len(xy):continue
                ss,nn=project_route(xy,route);best=np.zeros(len(xy));owner=np.zeros(len(xy),dtype=int)
                for stone_id,(s,nc,rs,rn) in enumerate(layouts[terrain_id]):
                    h=np.maximum(0,1-(ss-s)**2/rs**2-(nn-nc)**2/rn**2)**2;better=h>best;owner[better]=stone_id//stones_per_cluster;best=np.maximum(best,h)
                hit=best>0;np.minimum.at(onsets[:,wi,:],(lanes[hit],owner[hit]),times[hit])
    return onsets
