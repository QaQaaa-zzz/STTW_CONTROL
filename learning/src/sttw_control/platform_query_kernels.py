"""Fused GPU geometry for single-robot, full-horizon platform queries."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as math


@triton.jit
def _interpolate(s, arc, values, offset, stride: tl.constexpr, idx, count: tl.constexpr):
    # searchsorted(right=True), including repeated arc samples.
    i = tl.minimum(tl.maximum(tl.sum((arc <= offset).to(tl.int32), 0) - 1, 0), count - 2)
    a = tl.load(s+i)
    f = tl.minimum(1., tl.maximum(0., math.div_rn(offset-a, tl.load(s+i+1)-a)))
    return tl.load(values+i*stride+idx)*(1.-f) + tl.load(values+(i+1)*stride+idx)*f


@triton.jit
def _segment(x, y, xy, i):
    a = tl.load(xy+2*i)
    b = tl.load(xy+2*i+1)
    dx = tl.load(xy+2*i+2)-a
    dy = tl.load(xy+2*i+3)-b
    frac = tl.minimum(1.,tl.maximum(0.,math.div_rn((x-a)*dx+(y-b)*dy,dx*dx+dy*dy)))
    px = a+frac*dx
    py = b+frac*dy
    return frac, px, py, (x-px)*(x-px)+(y-py)*(y-py)


@triton.jit
def _features(P,V,S,XY,H,K,ETA,N:tl.constexpr,B:tl.constexpr,
              TIME:tl.constexpr,MIN_SPEED:tl.constexpr,LO:tl.constexpr,HI:tl.constexpr,SCALE:tl.constexpr):
    n = tl.program_id(0)
    p = P + n*12
    out = ETA + n*15
    xy = XY + n*N*2
    heading = H+n*N
    curvature = K+n*N
    x,y=tl.load(p),tl.load(p+1)
    j=tl.arange(0,B)
    dx=tl.load(xy+2*j,j<N,other=0)-x
    dy=tl.load(xy+2*j+1,j<N,other=0)-y
    nearest=tl.argmin(tl.where(j<N,dx*dx+dy*dy,float('inf')),0)
    i0=tl.maximum(0,nearest-1)
    i1=tl.minimum(N-2,nearest)
    u0,x0,y0,d0=_segment(x,y,xy,i0)
    u1,x1,y1,d1=_segment(x,y,xy,i1)
    first=d0<=d1
    i=tl.where(first,i0,i1)
    u=tl.where(first,u0,u1)
    ex=x-tl.where(first,x0,x1)
    ey=y-tl.where(first,y0,y1)
    arc=tl.load(S+j,j<N,other=float('inf'))
    s0=tl.load(S+i)
    at=s0+u*(tl.load(S+i+1)-s0)
    h=_interpolate(S,arc,heading,at,1,0,N)
    look=tl.minimum(HI,tl.maximum(LO,TIME*tl.maximum(tl.load(V+n*9),MIN_SPEED)))*SCALE
    ax=_interpolate(S,arc,xy,at+look,2,0,N)-x
    ay=_interpolate(S,arc,xy,at+look,2,1,N)-y
    w,q,r,t=tl.load(p+3),tl.load(p+4),tl.load(p+5),tl.load(p+6)
    roll=math.atan2(2*(w*q+r*t),1-2*(q*q+r*r))
    pitch_argument=tl.minimum(1.,tl.maximum(-1.,2*(w*r-t*q)))
    yaw=math.atan2(2*(w*t+q*r),1-2*(r*r+t*t))
    sn,cs=math.sin(yaw),math.cos(yaw)
    tl.store(out,tl.load(p+2))
    tl.store(out+1,roll)
    tl.store(out+2,pitch_argument)
    tl.store(out+3,sn)
    tl.store(out+4,cs)
    tl.store(out+5,tl.load(p+7))
    for k in tl.static_range(3):
        tl.store(out+6+k,_interpolate(S,arc,curvature,at+.5*k,1,0,N))
    tl.store(out+9,-math.sin(h)*ex+math.cos(h)*ey)
    tl.store(out+10,math.atan2(math.sin(h-yaw),math.cos(h-yaw)))
    tl.store(out+11,cs*ax+sn*ay)
    tl.store(out+12,-sn*ax+cs*ay)
    tl.store(out+13,1.8)
    tl.store(out+14,look)


def route_features(follower, pose, velocity, s, xy, heading, curvature):
    out = pose.new_empty((len(pose),15))
    _features[(len(pose),)](pose.contiguous(),velocity.contiguous(),s,xy,heading,curvature,out,
        len(s),triton.next_power_of_2(len(s)),follower['lookahead_time_s'],
        follower['lookahead_min_speed_m_s'],follower['lookahead_min_m'],follower['lookahead_max_m'],
        follower.get('lookahead_scale',1.),enable_fp_fusion=False)
    # Use the same CUDA asin implementation as the trained predictor.
    out[:,2].asin_()
    return out


@triton.jit
def _terrain_grid(P, OFFSETS, BOUNDS, GRID):
    n = tl.program_id(0)
    p = P+n*12
    w,x,y,z = tl.load(p+3),tl.load(p+4),tl.load(p+5),tl.load(p+6)
    yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    sn,cs = math.sin(yaw),math.cos(yaw)
    j = tl.arange(0,16)
    ox,oy = tl.load(OFFSETS+2*j,j<15,0.),tl.load(OFFSETS+2*j+1,j<15,0.)
    wx = tl.load(p)+cs*ox-sn*oy
    wy = tl.load(p+1)+sn*ox+cs*oy
    x0,x1,y0,y1=tl.load(BOUNDS),tl.load(BOUNDS+1),tl.load(BOUNDS+2),tl.load(BOUNDS+3)
    tl.store(GRID+n*30+2*j, math.div_rn(2*(wx-x0),x1-x0)-1,j<15)
    tl.store(GRID+n*30+2*j+1, math.div_rn(2*(wy-y0),y1-y0)-1,j<15)


def terrain_features(terrain, pose):
    grid = pose.new_empty((1,len(pose)*15,1,2))
    _terrain_grid[(len(pose),)](pose.contiguous(),terrain.offsets,terrain.bounds,grid,enable_fp_fusion=False)
    sampled = torch.nn.functional.grid_sample(terrain.field,grid,align_corners=True,padding_mode='zeros')
    return sampled[0,:,:,0].T.reshape(len(pose),-1)


@triton.jit
def _cross(ax,ay,az,bx,by,bz):
    return math.fma_rn(ay,bz,-az*by),math.fma_rn(az,bx,-ax*bz),math.fma_rn(ax,by,-ay*bx)


@triton.jit
def _rotate(w,x,y,z,a,b,c):
    i,j,k=_cross(x,y,z,a,b,c)
    i,j,k=_cross(x,y,z,i+w*a,j+w*b,k+w*c)
    return a+2*i,b+2*j,c+2*k


@triton.jit
def _advance(P,U,V,OUT):
    n=tl.program_id(0)
    p=P+n*12
    u=U+n*9
    v=V+n*9
    o=OUT+n*12
    a=(tl.load(u+3)+tl.load(v+3))*.5*.02
    b=(tl.load(u+4)+tl.load(v+4))*.5*.02
    c=(tl.load(u+5)+tl.load(v+5))*.5*.02
    angle=math.sqrt((a*a+c*c)+b*b)
    # Eager tensor/scalar division on CUDA uses a rounded reciprocal.
    product=(angle*(1./6.283185307179586))*3.141592653589793
    sinc=tl.where(product==0.,1.,math.div_rn(math.sin(product),product))
    t=math.cos(angle*.5)
    a,b,c=a*.5*sinc,b*.5*sinc,c*.5*sinc
    w,x,y,z=tl.load(p+3),tl.load(p+4),tl.load(p+5),tl.load(p+6)
    i,j,k=_cross(x,y,z,a,b,c)
    nw=w*t-((x*a+z*c)+y*b)
    nx=w*a+x*t+i
    ny=w*b+y*t+j
    nz=w*c+z*t+k
    norm=math.sqrt((nw*nw+nx*nx)+(ny*ny+nz*nz))
    nw,nx,ny,nz=math.div_rn(nw,norm),math.div_rn(nx,norm),math.div_rn(ny,norm),math.div_rn(nz,norm)
    ax,ay,az=_rotate(w,x,y,z,tl.load(u),tl.load(u+1),tl.load(u+2))
    bx,by,bz=_rotate(nw,nx,ny,nz,tl.load(v),tl.load(v+1),tl.load(v+2))
    tl.store(o,tl.load(p)+.01*(ax+bx))
    tl.store(o+1,tl.load(p+1)+.01*(ay+by))
    tl.store(o+2,tl.load(p+2)+.01*(az+bz))
    tl.store(o+3,nw)
    tl.store(o+4,nx)
    tl.store(o+5,ny)
    tl.store(o+6,nz)
    for jnt in tl.static_range(3):
        tl.store(o+7+jnt,tl.load(p+7+jnt)+(tl.load(u+6+jnt)+tl.load(v+6+jnt))*.5*.02)
    tl.store(o+10,tl.load(p+10))
    tl.store(o+11,tl.load(p+11))


def fused_step(model, pose, xi, previous, hidden, s, xy, heading, curvature, speed, terrain_ids, alpha):
    eta = route_features(model.nominal.follower,pose,xi,s,xy,heading,curvature)
    eta[:,13] = speed
    terrain = terrain_features(model.terrain,pose)
    features = torch.cat((xi,eta,pose[:,10:12],terrain),1)
    hidden = model.memory((features-model.input_mean)/model.input_scale,hidden)
    arx = torch.cat((xi,previous,eta),1)
    nxt = ((arx-model.nominal.mean)/model.nominal.scale) @ model.nominal.weight + model.nominal.bias
    if model.nominal.identity:
        nxt = nxt+xi
    nxt = nxt + model.correction(hidden)*model.correction_scale
    out = torch.empty_like(pose)
    _advance[(len(pose),)](pose.contiguous(),xi.contiguous(),nxt.contiguous(),out,enable_fp_fusion=False)
    return out,nxt,hidden,eta[:,9:12]
