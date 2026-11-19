"""Opt-in canonical ordering of active dense elliptic constraint equations on GPU."""
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import constraint
from mujoco.mjx.third_party.mujoco_warp._src.types import ConstraintType

_make_constraint = constraint.make_constraint


@wp.func
def contact_axis(address: wp.array2d(dtype=int), dim: wp.array(dtype=int), contact: int, row: int):
    for i in range(dim[contact]):
        if address[contact, i] == row:
            return i
    return -1


@wp.func
def row_greater(kind: wp.array2d(dtype=int), owner: wp.array2d(dtype=int),
                geom: wp.array(dtype=wp.vec2i), pos: wp.array(dtype=wp.vec3),
                frame: wp.array(dtype=wp.mat33), dim: wp.array(dtype=int),
                address: wp.array2d(dtype=int), world: int, left: int, right: int):
    lk, rk = int(kind[world,left]), int(kind[world,right])
    if lk != rk:
        return lk > rk
    a, b = int(owner[world,left]), int(owner[world,right])
    if lk < int(ConstraintType.CONTACT_FRICTIONLESS):
        return a > b
    for j in range(2):
        if geom[a][j] != geom[b][j]:
            return geom[a][j] > geom[b][j]
    for j in range(3):
        if pos[a][j] != pos[b][j]:
            return pos[a][j] > pos[b][j]
    for i in range(3):
        for j in range(3):
            if frame[a][i,j] != frame[b][i,j]:
                return frame[a][i,j] > frame[b][i,j]
    return contact_axis(address,dim,a,left) > contact_axis(address,dim,b,right)


@wp.func
def swap_scalar(values: wp.array2d(dtype=float), world: int, a: int, b: int):
    tmp = float(values[world,a])
    values[world,a] = values[world,b]
    values[world,b] = tmp


@wp.kernel
def order_active(nefc: wp.array(dtype=int), dim: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
                 pos: wp.array(dtype=wp.vec3), frame: wp.array(dtype=wp.mat33), address: wp.array2d(dtype=int),
                 kind: wp.array2d(dtype=int), owner: wp.array2d(dtype=int), jac: wp.array3d(dtype=float),
                 position: wp.array2d(dtype=float), margin: wp.array2d(dtype=float), mass: wp.array2d(dtype=float),
                 vel: wp.array2d(dtype=float), aref: wp.array2d(dtype=float), friction: wp.array2d(dtype=float)):
    world = wp.tid()
    for row in range(1,wp.min(nefc[world],kind.shape[1])):
        right = int(row)
        while right > 0:
            left = right - 1
            if not row_greater(kind,owner,geom,pos,frame,dim,address,world,left,right):
                break
            lk, rk = int(kind[world,left]), int(kind[world,right])
            a, b = int(owner[world,left]), int(owner[world,right])
            la, ra = int(-1), int(-1)
            if lk >= int(ConstraintType.CONTACT_FRICTIONLESS):
                la = contact_axis(address,dim,a,left)
            if rk >= int(ConstraintType.CONTACT_FRICTIONLESS):
                ra = contact_axis(address,dim,b,right)
            if la >= 0:
                address[a,la] = right
            if ra >= 0:
                address[b,ra] = left
            kind[world,left] = rk
            kind[world,right] = lk
            owner[world,left] = b
            owner[world,right] = a
            for j in range(jac.shape[2]):
                tmp = float(jac[world,left,j])
                jac[world,left,j] = jac[world,right,j]
                jac[world,right,j] = tmp
            swap_scalar(position,world,left,right)
            swap_scalar(margin,world,left,right)
            swap_scalar(mass,world,left,right)
            swap_scalar(vel,world,left,right)
            swap_scalar(aref,world,left,right)
            swap_scalar(friction,world,left,right)
            right = left


def make_constraint(model, data):
    _make_constraint(model, data)
    c, e = data.contact, data.efc
    wp.launch(order_active, dim=data.nworld, inputs=[
        data.nefc, c.dim, c.geom, c.pos, c.frame, c.efc_address,
        e.type, e.id, e.J, e.pos, e.margin, e.D, e.vel, e.aref, e.frictionloss])


def enable():
    constraint.make_constraint = make_constraint
