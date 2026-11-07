"""Frozen P1 capability cells and map-only prospective contact-spacing features."""
import json
from pathlib import Path
import numpy as np
from .route_pebbles import cap_height


def preview_features(terrain,speed):
    terrain=Path(terrain);c=json.loads((terrain/'identity.json').read_text());stones=np.loadtxt(terrain/'stone_centers.csv',delimiter=',',skiprows=1,usecols=(1,2,3,4));ss=np.arange(c['s_range_m'][0],c['s_range_m'][1]+.0025,.005);raised=cap_height(ss,np.zeros_like(ss),stones,c)>.0005;starts=ss[np.flatnonzero(raised&~np.r_[False,raised[:-1]])];gaps=np.diff(starts)
    return [c['height_m'],len(stones)/((c['s_range_m'][1]-c['s_range_m'][0])*2*c['half_width_m']),float(len(starts)),float(np.median(gaps)/speed) if len(gaps) else 0.,float(np.std(gaps)/np.mean(gaps)) if len(gaps) else 0.,float(raised.mean())]


def choose_preference(model,task,speed,features,phase):
    candidates=[]
    for alpha in model['alphas']:
        cells=[r for r in model['cells'] if r['task']==task and r['speed_m_s']==speed and r['alpha']==alpha]
        if phase:
            distance=np.linalg.norm((np.array([r['preview_features'] for r in cells])-features)/model['feature_scale'],axis=1);indices=np.argsort(distance,kind='stable')[:model['neighbors']];weights=1/(distance[indices]+model['distance_floor'])**2
        else:indices=np.arange(len(cells));weights=np.ones(len(cells))
        probability=float(np.average([cells[i]['qualified_fraction'] for i in indices],weights=weights));cost=float(np.average([cells[i]['normalized_cost'] for i in indices],weights=weights));candidates.append(dict(alpha=alpha,predicted_qualified_fraction=probability,predicted_cost=cost,neighbor_terrains=[cells[i]['terrain'] for i in indices]))
    best=min(candidates,key=lambda x:(-x['predicted_qualified_fraction'],x['predicted_cost'],x['alpha']))
    return best['alpha'],candidates
