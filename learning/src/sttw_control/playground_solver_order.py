# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Fixed reduction order for batched Newton solves with canonical elliptic rows."""
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import math, smooth, solver, types

_native_gradient = solver._update_gradient
_native_update = solver._update_constraint


@wp.kernel
def backward_force(parent: wp.array(dtype=int), force: wp.array2d(dtype=wp.spatial_vector),
                   level: wp.array(dtype=int), out: wp.array2d(dtype=wp.spatial_vector)):
    w, lane = wp.tid()
    if lane == 0:
        for j in range(level.shape[0]):
            child = level[j]
            if child != 0:
                p = parent[child]
                out[w,p] = out[w,p] + force[w,child]


@wp.kernel
def gradient(smooth_force: wp.array2d(dtype=float), constrained: wp.array2d(dtype=float),
             ma: wp.array2d(dtype=float), done: wp.array(dtype=bool),
             grad: wp.array2d(dtype=float), norm: wp.array(dtype=float)):
    w, lane = wp.tid()
    if lane == 0 and not done[w]:
        total = float(0.)
        for j in range(smooth_force.shape[1]):
            g = ma[w,j] - smooth_force[w,j] - constrained[w,j]
            grad[w,j] = g
            total += g*g
        norm[w] = total


@wp.kernel
def initial_search(mgrad: wp.array2d(dtype=float), search: wp.array2d(dtype=float),
                   norm: wp.array(dtype=float)):
    w, lane = wp.tid()
    if lane == 0:
        total = float(0.)
        for j in range(search.shape[1]):
            s = -mgrad[w,j]
            search[w,j] = s
            total += s*s
        norm[w] = total


@wp.kernel
def update_search(method: int, mgrad: wp.array2d(dtype=float), previous: wp.array2d(dtype=float),
                  beta: wp.array(dtype=float), done: wp.array(dtype=bool),
                  search: wp.array2d(dtype=float), norm: wp.array(dtype=float)):
    w, lane = wp.tid()
    if lane == 0 and not done[w]:
        total = float(0.)
        for j in range(search.shape[1]):
            s = -mgrad[w,j]
            if method == types.SolverType.CG:
                s += beta[w]*previous[w,j]
            search[w,j] = s
            total += s*s
        norm[w] = total


@wp.kernel
def ordered_cost(nefc: wp.array(dtype=int), state: wp.array2d(dtype=int), owner: wp.array2d(dtype=int),
                 mass: wp.array2d(dtype=float), loss: wp.array2d(dtype=float),
                 residual: wp.array2d(dtype=float), dim: wp.array(dtype=int),
                 address: wp.array2d(dtype=int), friction: wp.array(dtype=types.vec5),
                 ratio: wp.array(dtype=float), done: wp.array(dtype=bool),
                 gauss: wp.array(dtype=float), cost: wp.array(dtype=float)):
    w = wp.tid()
    if done[w]:
        return
    total = float(0.)
    for r in range(wp.min(nefc[w],state.shape[1])):
        st = state[w,r]
        d, u = mass[w,r], residual[w,r]
        if st == types.ConstraintState.QUADRATIC:
            total += 0.5*d*u*u
        elif st == types.ConstraintState.LINEARNEG or st == types.ConstraintState.LINEARPOS:
            f = loss[w,r]
            signed = wp.where(st == types.ConstraintState.LINEARNEG,u,-u)
            total -= f*(0.5*math.safe_div(f,d)+signed)
        elif st == types.ConstraintState.CONE:
            c = owner[w,r]
            if address[c,0] == r:
                mu = friction[c][0]*ratio[w % ratio.shape[0]]
                tt = float(0.)
                for j in range(1,dim[c]):
                    tangent = residual[w,address[c,j]]*friction[c][j-1]
                    tt += tangent*tangent
                nmt = u*mu - mu*wp.sqrt(tt)
                total += 0.5*math.safe_div(d,mu*mu*(1.+mu*mu))*nmt*nmt
    cost[w] = total + gauss[w]


def update_constraint(model,data,ctx):
    _native_update(model,data,ctx)
    e, c = data.efc, data.contact
    wp.launch(ordered_cost,dim=data.nworld,inputs=[data.nefc,e.state,e.id,e.D,e.frictionloss,
        ctx.Jaref,c.dim,c.efc_address,c.friction,model.opt.impratio_invsqrt,ctx.done,ctx.gauss,ctx.cost])



# Elliptic cone Hessian from Newton Developers' MuJoCo Warp solver.
# One thread owns each world/matrix entry and visits active normal rows in order.
@wp.kernel
def cone_hessian(nefc: wp.array(dtype=int), owner: wp.array2d(dtype=int),
                 opt_impratio_invsqrt: wp.array(dtype=float), dof_tri_row: wp.array(dtype=int),
                 dof_tri_col: wp.array(dtype=int), contact_friction_in: wp.array(dtype=types.vec5),
                 contact_dim_in: wp.array(dtype=int), contact_efc_address_in: wp.array2d(dtype=int),
                 efc_J_in: wp.array3d(dtype=float), efc_D_in: wp.array2d(dtype=float),
                 efc_state_in: wp.array2d(dtype=int), ctx_Jaref_in: wp.array2d(dtype=float),
                 done: wp.array(dtype=bool), ctx_h_out: wp.array3d(dtype=float)):
    worldid, element = wp.tid()
    if done[worldid]:
        return
    dof1id, dof2id = dof_tri_row[element], dof_tri_col[element]
    for efcid0 in range(wp.min(nefc[worldid],owner.shape[1])):
        if efc_state_in[worldid,efcid0] != types.ConstraintState.CONE:
            continue
        conid = owner[worldid,efcid0]
        if contact_efc_address_in[conid,0] != efcid0:
            continue
        condim = contact_dim_in[conid]
        fri = contact_friction_in[conid]
        mu = fri[0] * opt_impratio_invsqrt[worldid]

        mu2 = mu * mu
        dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))

        if dm == 0.0:
          continue

        n = ctx_Jaref_in[worldid, efcid0] * mu
        u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)

        tt = float(0.0)
        for j in range(1, condim):
          efcidj = contact_efc_address_in[conid, j]
          uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]
          tt += uj * uj
          u[j] = uj

        if tt <= 0.0:
          t = 0.0
        else:
          t = wp.sqrt(tt)
        t = wp.max(t, types.MJ_MINVAL)
        ttt = wp.max(t * t * t, types.MJ_MINVAL)

        h = float(0.0)

        for dim1id in range(condim):
          if dim1id == 0:
            efcid1 = efcid0
          else:
            efcid1 = contact_efc_address_in[conid, dim1id]

          efc_J11 = efc_J_in[worldid, efcid1, dof1id]
          efc_J12 = efc_J_in[worldid, efcid1, dof2id]

          ui = u[dim1id]

          for dim2id in range(0, dim1id + 1):
            if dim2id == 0:
              efcid2 = efcid0
            else:
              efcid2 = contact_efc_address_in[conid, dim2id]

            efc_J21 = efc_J_in[worldid, efcid2, dof1id]
            efc_J22 = efc_J_in[worldid, efcid2, dof2id]

            uj = u[dim2id]

            # set first row/column: (1, -mu/t * u)
            if dim1id == 0 and dim2id == 0:
              hcone = 1.0
            elif dim1id == 0:
              hcone = -math.safe_div(mu, t) * uj
            elif dim2id == 0:
              hcone = -math.safe_div(mu, t) * ui
            else:
              hcone = mu * math.safe_div(n, ttt) * ui * uj

              # add to diagonal: mu^2 - mu * n / t
              if dim1id == dim2id:
                hcone += mu2 - mu * math.safe_div(n, t)

            # pre and post multiply by diag(mu, friction) scale by dm
            if dim1id == 0:
              fri1 = mu
            else:
              fri1 = fri[dim1id - 1]

            if dim2id == 0:
              fri2 = mu
            else:
              fri2 = fri[dim2id - 1]

            hcone *= dm * fri1 * fri2

            if hcone != 0.0:
              h += hcone * efc_J11 * efc_J22

              if dim1id != dim2id:
                h += hcone * efc_J12 * efc_J21

        ctx_h_out[worldid, dof1id, dof2id] += h


def dense_gradient(m,d,ctx):
    if m.is_sparse or m.nv > solver._BLOCK_CHOLESKY_DIM or m.opt.solver != types.SolverType.NEWTON:
        return _native_gradient(m,d,ctx)
    e, c = d.efc, d.contact
    wp.launch(gradient,dim=(d.nworld,m.nv),inputs=[d.qfrc_smooth,d.qfrc_constraint,e.Ma,ctx.done,ctx.grad,ctx.grad_dot])
    wp.launch_tiled(solver.update_gradient_JTDAJ_dense_tiled(e.J.shape[2],types.TILE_SIZE_JTDAJ_DENSE,d.njmax),
        dim=d.nworld,inputs=[d.nefc,d.qM,e.J,e.D,e.state,ctx.done,ctx.h],block_dim=m.block_dim.update_gradient_JTDAJ_dense)
    wp.launch(cone_hessian,dim=(d.nworld,m.dof_tri_row.size),inputs=[d.nefc,e.id,m.opt.impratio_invsqrt,
        m.dof_tri_row,m.dof_tri_col,c.friction,c.dim,c.efc_address,e.J,e.D,e.state,ctx.Jaref,ctx.done,ctx.h])
    wp.launch_tiled(solver.update_gradient_cholesky(m.nv),dim=d.nworld,inputs=[ctx.grad,ctx.h,ctx.done,ctx.Mgrad],
        block_dim=m.block_dim.update_gradient_cholesky)


def enable():
    solver._update_gradient = dense_gradient
    smooth._cfrc_backward = backward_force
    solver.solve_init_search = initial_search
    solver.solve_search_update = update_search
    solver.update_gradient_grad = gradient
    solver._update_constraint = update_constraint
