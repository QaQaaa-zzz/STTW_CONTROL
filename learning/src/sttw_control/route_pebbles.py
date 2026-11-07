"""World-fixed embedded caps on a local route patch; no moving stones."""
import numpy as np
from scipy.spatial import cKDTree, Delaunay


def project_route(points, route):
    xy=route['xy'];edge=np.diff(xy,axis=0);tree=cKDTree((xy[:-1]+xy[1:])/2)
    ss=[];nn=[]
    for chunk in np.array_split(points,max(1,int(np.ceil(len(points)/50000)))):
        _,ix=tree.query(chunk,k=min(4,len(edge)))
        d=chunk[:,None,:]-xy[ix];e=edge[ix]
        f=np.clip((d*e).sum(-1)/(e*e).sum(-1),0,1);off=d-f[...,None]*e
        best=(off*off).sum(-1).argmin(1);rows=np.arange(len(chunk));j=ix[rows,best]
        v=off[rows,best];t=edge[j]/np.linalg.norm(edge[j],axis=1)[:,None]
        ss.append(route['s'][j]+f[rows,best]*np.diff(route['s'])[j]);nn.append(t[:,0]*v[:,1]-t[:,1]*v[:,0])
    return np.concatenate(ss),np.concatenate(nn)


def stone_layout(config):
    rng=np.random.default_rng(config['seed']);lo,hi=config['s_range_m'];w=config['half_width_m']
    count=round(config['density_per_m2']*(hi-lo)*2*w)
    stones=np.column_stack((rng.uniform(lo,hi,count),rng.uniform(-w,w,count),rng.uniform(*config['radius_range_m'],(count,2))))
    if 'reference_layout_csv' in config:
        original=np.loadtxt(config['reference_layout_csv'],delimiter=',',skiprows=1,usecols=(1,2,3,4),ndmin=2)
        shared=min(count,len(original));stones[:shared]=original[:shared]
    if config.get('distribution')=='clustered':
        centers=np.asarray(config['cluster_centers_sn_m']);chosen=centers[rng.integers(len(centers),size=count)]
        stones[:,:2]=np.clip(chosen+rng.normal(size=(count,2))*config['cluster_std_sn_m'],[lo,-w],[hi,w])
    return stones


def cap_height(s,n,stones,config):
    height=np.zeros_like(s)
    for si,ni,rs,rn in stones:
        mask=(abs(s-si)<=rs)&(abs(n-ni)<=rn)
        q=(s[mask]-si)**2/rs**2+(n[mask]-ni)**2/rn**2
        height[mask]=np.maximum(height[mask],config['height_m']*np.maximum(0,1-q)**2)
    lo,hi=config['s_range_m'];b=config['blend_length_m']
    fade=lambda u:.5*(1-np.cos(np.pi*np.clip(u,0,1)))
    return height*fade((s-lo)/b)*fade((hi-s)/b)


def build_mesh(config,route):
    stones=stone_layout(config);dx=config['horizontal_scale_m'];dz=config['vertical_scale_m']
    use=(route['s']>=config['s_range_m'][0]-.2)&(route['s']<=config['s_range_m'][1]+.2)
    margin=config['half_width_m']+max(config['radius_range_m'])+.2
    local_xy=route['xy'][use].astype(np.float64)
    lower=np.floor((local_xy.min(0)-margin)*10).astype(np.int64)/10
    upper=np.ceil((local_xy.max(0)+margin)*10).astype(np.int64)/10
    ax=[np.linspace(lower[k],upper[k],round((upper[k]-lower[k])/dx)+1) for k in (0,1)]
    x,y=np.meshgrid(*ax,indexing='ij');points=np.column_stack((x.ravel(),y.ravel()))
    s,n=project_route(points,route);z=np.rint(cap_height(s,n,stones,config)/dz)*dz
    coarse=np.arange(config['bounds_m'][0],config['bounds_m'][1]+.05,.1)
    xx,yy=np.meshgrid(coarse,coarse,indexing='ij');outside=np.column_stack((xx.ravel(),yy.ravel()))
    outside=outside[np.any((outside<lower-1e-7)|(outside>upper+1e-7),axis=1)]
    allxy=np.vstack((points,outside));tri=Delaunay(allxy).simplices.astype(np.int32)
    vertices=np.column_stack((allxy,np.r_[z,np.zeros(len(outside))])).astype(np.float32)
    a,b,c=vertices[tri].transpose(1,0,2);normal=np.cross(b-a,c-a)
    flip=normal[:,2]<0;tri[flip]=tri[flip][:,[0,2,1]]
    return vertices,tri,stones,dict(x=ax[0],y=ax[1],height=z.reshape(x.shape),s=s.reshape(x.shape),n=n.reshape(x.shape),bounds=np.r_[lower,upper])
