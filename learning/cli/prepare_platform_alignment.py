"""Prepare paired STTW entry paths with fixed entry and exit contracts."""
import argparse
import json
from pathlib import Path

import numpy as np


def aligned_route(grid, arc, front, sign, flat_length):
    def smooth(t):
        t = np.clip(t, 0, 1)
        return t**3 * (10 - 15*t + 6*t*t)
    pre = smooth((grid - (front-flat_length-4.5))/4.5)
    post = (grid-front-2.)/1.75
    y = sign*.22*pre*(1-smooth(post)) + .12*np.sin(np.pi*np.clip(post,0,1))**2
    distance = np.r_[0., np.cumsum(np.hypot(np.diff(grid), np.diff(y)))]
    xy = np.stack([np.interp(arc,distance,col) for col in (grid,y)],1)
    heading = np.arctan2(np.gradient(xy[:,1]),np.gradient(xy[:,0]))
    return xy, heading, np.gradient(heading,arc)


def shape_entry_heading(reference, entry_x, slope):
    """Change the reference tangent at RB entry while keeping its lateral target fixed there."""
    shaped = {key: value.copy() for key,value in reference.items()}
    arc = shaped['s']
    for lane, xy in enumerate(shaped['xy']):
        x, y = xy.T.copy()
        dx = x-entry_x
        taper = np.clip((3.-np.abs(dx))/.8,0.,1.)
        taper = taper**3*(10.-15.*taper+6.*taper**2)
        sign = 1 if lane%2==0 else -1
        y += sign*slope*dx*np.exp(-.5*(dx/1.55)**2)*taper
        distance = arc[0]+np.r_[0.,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
        xy[:,0] = np.interp(arc,distance,x)+np.maximum(arc-distance[-1],0.)
        xy[:,1] = np.interp(arc,distance,y)
        heading = np.arctan2(np.gradient(xy[:,1]),np.gradient(xy[:,0]))
        shaped['heading'][lane] = heading
        shaped['curvature'][lane] = np.gradient(heading,arc)
        prefix = x < entry_x-3.
        for key in ('xy','heading','curvature'):
            shaped[key][lane,prefix] = reference[key][lane,prefix]
    return shaped


def prepare(template, output, roots, stage, fresh_context):
    cfg = json.loads(template.read_text())
    hand = cfg['ground_handoff']
    front = json.loads(Path(cfg['world_map']).read_text())['terrain']['front_x_m']
    with np.load(hand['reference_path']) as ref:
        arc = ref['s']
        original = [tuple(ref[k][r] for k in ('xy','heading','curvature')) for r in (0,2)]
    grid = np.linspace(-5,25,1501)
    two = [aligned_route(grid,arc,front,s,2.) for s in (1,-1)]
    four = [aligned_route(grid,arc,front,s,4.) for s in (1,-1)]
    paths = {'alignment':original+two, 'settle':two+four, 'frozen':[four[0],two[1]]}[stage]
    route_path = output/'routes.npz'
    np.savez_compressed(route_path,s=arc,**dict(zip(('xy','heading','curvature'),map(np.stack,zip(*paths)))))
    hand['reference_path'] = str(route_path)
    hand['handoff_variants'] = [dict(hand['handoff_variants'][0],root_variant=i,bank_row=i) for i in roots]
    cfg.update(evaluation_lanes=[[r,'h75',i] for i in range(len(roots)) for r in range(len(paths))],
               collect_model_state=True, native_rb_entry_context=fresh_context,
               fixed_order_tree_reductions=True, canonical_constraint_rows=True,ordered_solver_reductions=True)
    (output/'screen.json').write_text(json.dumps(cfg,indent=2))
    return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--stage',choices=('alignment','settle','frozen'),default='frozen')
    parser.add_argument('--world',type=Path,help='Frozen physical map, including its goal window')
    parser.add_argument('--fresh-context',action='store_true')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cfg = prepare(args.template,args.output.resolve(),range(args.start,args.start+args.count),args.stage,args.fresh_context)
    if args.world:
        cfg['world_map'] = str(args.world.resolve())
        (args.output/'screen.json').write_text(json.dumps(cfg,indent=2))
