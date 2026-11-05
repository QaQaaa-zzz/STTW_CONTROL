"""Remove redundant zero-height grid lines without changing terrain surfaces."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=np.load(a.source/'terrain.npz');h=source['height_field'];rows,cols=h.shape
active=(h!=0).any(0);keep=active.copy();keep[1:]|=active[:-1];keep[:-1]|=active[1:];keep[[0,-1]]=True
j=np.flatnonzero(keep);i=np.arange(rows) if active.any() else np.array([0,rows-1])
vertices=source['vertices'][(i[:,None]*cols+j[None,:]).ravel()]
base=(np.arange(len(i)-1)[:,None]*len(j)+np.arange(len(j)-1)[None,:]).ravel();triangles=np.stack((np.stack((base,base+len(j)+1,base+1),-1),np.stack((base,base+len(j),base+len(j)+1),-1)),1).reshape(-1,3).astype(np.uint32)
a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'terrain.npz',vertices=vertices,triangles=triangles,height_field=h,source_grid_rows=i,source_grid_columns=j)
identity=json.loads((a.source/'identity.json').read_text());identity.update(source_identity=identity.copy(),source=str(a.source.resolve()),sha256=hashlib.sha256((a.output/'terrain.npz').read_bytes()).hexdigest(),vertices=len(vertices),triangles=len(triangles),compaction='retain every nonzero column and its neighbours, merge only intervening zero-height planes; original vertices and triangle winding retained')
(a.output/'identity.json').write_text(json.dumps(identity,indent=2));print(json.dumps(dict(source=str(a.source),triangles_before=len(source['triangles']),triangles_after=len(triangles),vertices_after=len(vertices))))
