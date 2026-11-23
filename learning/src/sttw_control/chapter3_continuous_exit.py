"""Post-relock receding-horizon CEM on the frozen raised-platform predictor.

The physics runner owns the pending publication, the original default route and
the controller history. A failed search returns the incumbent and admitted=False.
All lanes in one planner instance share a map, progress grid and frozen model.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .playground_planner import CommittedSplineRoutes
from .playground_prediction import euler, rotate


@dataclass
class ExitProposal:
    xy: torch.Tensor
    heading: torch.Tensor
    curvature: torch.Tensor
    admitted: torch.Tensor
    incumbent_admitted: torch.Tensor
    clearance_m: torch.Tensor
    valid_for_s: torch.Tensor
    searched: torch.Tensor | None = None


def committed_progress(progress, lookahead, speed_request, publication_budget_s=.1):
    """Preserve the pursuit preview, one lower update and time for publication."""
    return progress + lookahead + speed_request * (.02 + publication_budget_s)


def _score(query, poses, velocity, domain, section, side, remaining, gate_status, hold):
    """Frozen Stage 2 goal, body clearance and gate predicates with live task history."""
    n, samples = poses.shape[:2]
    tick = torch.arange(samples, device=poses.device)
    angles = euler(poses[:, :, 3:7].reshape(-1, 4)).reshape(n, samples, 3)
    world = rotate(poses[:, :, 3:7].reshape(-1, 4), velocity[:, :, :3].reshape(-1, 3))
    world = world.reshape(n, samples, 3)
    yaw = angles[:, :, 2]
    speed = world[:, :, 0] * yaw.cos() + world[:, :, 1] * yaw.sin()
    x, y = poses[:, :, 0], poses[:, :, 1]
    g = query.goal
    hit = ((x >= g['x_min_m']) & (x <= g['x_max_m']) & (y.abs() <= g['abs_y_max_m'])
           & (world[:, :, 0] >= g['min_vx_m_s']) & ((speed - 2.).abs() <= g['speed_tolerance_m_s'])
           & (domain[:, :, 0].abs() <= g['path_tolerance_m'])
           & (domain[:, :, 1].abs() <= g['heading_tolerance_rad']))
    last_bad = torch.where(hit, -1, tick).cummax(1).values
    duration = (tick - last_bad - 1) * .02 + (last_bad == -1) * hold[:, None]
    horizon = torch.floor(remaining / .02 + 1e-5).long().clamp(0, samples - 1)
    held = hit & (duration + 1e-6 >= g['hold_s']) & (tick <= horizon[:, None])
    end = torch.where(held.any(1), held.long().argmax(1), horizon)
    window = tick <= end[:, None]
    clearances = []
    # Limit temporary swept-body tensors at larger candidate batch sizes.
    for offset in range(0, samples, 32):
        low, high = query.envelope(poses[:, offset:offset + 32])
        gap = torch.maximum(low[:, :, :, None] - (query.center + query.half),
                            (query.center - query.half) - high[:, :, :, None])
        clearances.append(gap.amax(-1).amin((2, 3)))
    clearance = torch.cat(clearances, 1).masked_fill(~window, float('inf')).amin(1)
    valid = ((domain[:, :, 0].abs() <= query.halfwidth)
             & (domain[:, :, 1].abs() < torch.pi / 2)
             & (domain[:, :, 2] >= query.model.nominal.follower['preview_body_x_min_m'])
             & (angles[:, :, 0].abs() <= .7) & (angles[:, :, 1].abs() <= 1.309))
    within = ((~window) | valid).all(1)
    crosses = (x[:, :-1] < section[:, None]) & (x[:, 1:] >= section[:, None]) & window[:, 1:]
    first = crosses.long().argmax(1)
    row = torch.arange(n, device=x.device)
    fraction = (section - x[row, first]) / (x[row, first + 1] - x[row, first]).clamp_min(1e-7)
    at = y[row, first] + fraction * (y[row, first + 1] - y[row, first])
    passed = crosses.any(1) & (at * side >= .06) & (at.abs() <= 3.9)
    gate = (gate_status == 1) | ((gate_status == 0) & passed)
    safe = clearance > query.margin
    goal = within & held.any(1)
    admitted = safe & goal & gate
    score = (16 * admitted.float() + 4 * safe.float() + 2 * gate.float() + goal.float()
             + .01 * clearance.clamp(-1, 1))
    score = torch.nan_to_num(score, nan=-1e6)
    return score, admitted, clearance, end * .02


class ContinuousExitPlanner:
    """Batched independent single-robot searches. Parameters and noise never mix robots."""

    def __init__(self, query, route, *, candidates=None, iterations=2, robots_per_chunk=1,
                 horizon_steps=255, seed=20261122, fused=True, mode='frozen'):
        if mode not in ('frozen','smooth_event'):
            raise ValueError('Unknown continuous-exit mode')
        self.mode = mode
        candidates = (64 if mode=='smooth_event' else 16) if candidates is None else candidates
        if mode=='smooth_event' and candidates!=64:
            raise ValueError('The smooth-event comparison uses 64 proposals per round')
        if candidates < 8 or candidates % 2 or iterations < 1 or robots_per_chunk < 1 or horizon_steps < 1:
            raise ValueError('CEM needs two equal populations of at least four and a positive horizon')
        if not query.relative or query.model.model_kind == 'direct' or query.model.alpha_conditioned:
            raise ValueError('Use the frozen relative-height residual departure predictor')
        self.query = query
        self.s = route.s
        self.candidates = candidates
        self.iterations = iterations
        self.chunk = robots_per_chunk
        self.horizon = horizon_steps
        self.fused = fused
        self.graph = None
        self.inputs = None
        device = route.s.device
        if fused and device.type != 'cuda':
            raise ValueError('Fused inference requires CUDA')
        if fused:
            from .platform_query_kernels import fused_step, route_features
            self.fused_step = fused_step
            self.fused_features = route_features
        # Fixed, independent left/right CEM populations; the same noise is used across paired cases.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.noise = torch.randn(iterations, 2, candidates // 2, 6, generator=generator).to(route.s)
        self.signs = route.s.new_tensor([1., -1.]).reshape(1, 2, 1)
        self.splines = []
        if mode=='smooth_event':
            # Separate fixed-size graphs for monitoring one current path and scoring 64 alternatives.
            self.event_graphs = {}
            return
        one = torch.zeros(1, dtype=torch.long, device=device)
        basic = SimpleNamespace(s=route.s, xy=route.xy[:1], heading=route.heading[:1],
                                curvature=route.curvature[:1], route_ids=one)
        for _ in range(self.chunk):
            self.splines.append(CommittedSplineRoutes(basic, query.goal['x_min_m']))

    def _evaluate(self, q, velocity, previous, xy, heading, curvature, section, side, remaining, gate, hold):
        model = self.query.model
        pose = q.index_select(1, self.query.pose_order)
        observed = torch.cat((pose[:, :8], pose[:, 10:12], velocity, previous), 1)
        hidden = ((observed - self.query.mean) / self.query.scale) @ self.query.weights[1:] + self.query.weights[0]
        height = self.query.relative['height']
        pose[:, 2] -= height
        req = pose.new_full((len(pose),), 2.)
        poses, twists, domains = [pose], [velocity], []
        for _ in range(self.horizon):
            args = (pose, velocity, previous, hidden, self.s, xy, heading, curvature, req, None, None)
            if self.fused:
                pose, nxt, hidden, domain = self.fused_step(model, *args)
            else:
                pose, nxt, hidden, domain = model.step(*args)
            poses.append(pose)
            twists.append(nxt)
            domains.append(domain)
            previous, velocity = velocity, nxt
        if self.fused:
            last = self.fused_features(model.nominal.follower, pose, velocity, self.s, xy, heading, curvature)
        else:
            last = model.nominal.features(pose, velocity, self.s, xy, heading, curvature)
        domains.append(last[:, 9:12])
        poses = torch.stack(poses, 1)
        poses[:, :, 2] += height
        return _score(self.query, poses, torch.stack(twists, 1), torch.stack(domains, 1),
                      section, side, remaining, gate, hold)

    def _solve(self, q, velocity, previous, xy, heading, curvature, committed, section, side, remaining, gate, hold):
        n = self.chunk
        b = self.candidates
        rows = torch.arange(n, device=q.device)
        for i, spline in enumerate(self.splines):
            active = SimpleNamespace(xy=xy, heading=heading, curvature=curvature, route_ids=rows[i:i+1])
            spline.update(active, committed[i])
        means = (.45 * self.signs).expand(n, 2, 6).clone()
        std = torch.full_like(means, .40)
        repeated = [x.repeat_interleave(b, 0) for x in (q, velocity, previous, section, side, remaining, gate, hold)]
        for iteration in range(self.iterations):
            p = (means[:, :, None] + std[:, :, None] * self.noise[iteration]).clamp(-1.25, 1.25)
            p[:, :, 0] = means
            p[:, :, 1] = 0.
            pflat = p.reshape(n, b, 6)
            paths = [spline(pflat[i]) for i, spline in enumerate(self.splines)]
            bank = [torch.cat(parts, 0).contiguous() for parts in zip(*paths)]
            score, admitted, clearance, valid_for = self._run_evaluate((*repeated[:3], *bank, *repeated[3:]))
            score = score.reshape(n, b)
            elites = score.reshape(n, 2, b // 2).argsort(dim=2, descending=True, stable=True)[:, :, :max(2, b // 8)]
            population = p.gather(2, elites[..., None].expand(-1, -1, -1, 6))
            means = population.mean(2)
            std = population.std(2, unbiased=False).clamp_min(.06)
            choice = score.argmax(1)
            incumbent = admitted.reshape(n, b)[:, 1].clone()
            publish = admitted.reshape(n, b)[rows, choice]
            # When no candidate is admitted, continue the current route. When the incumbent
            # itself passes, retain it as in Stage 2. The runner handles expiry/defaults.
            choice = torch.where(incumbent | ~publish, 1, choice)
            which = rows * b + choice
            current = [t[which] for t in bank] + [publish, incumbent, clearance[which], valid_for[which]]
            best_score_this = score[rows, choice]
            if iteration == 0:
                best, best_score = current, best_score_this
            else:
                take = (current[3] & ~best[3]) | ((current[3] == best[3]) & (best_score_this > best_score))
                best = [torch.where(take.reshape(n, *([1] * (v.ndim - 1))), v, old) for v, old in zip(current, best)]
                best_score = torch.where(take, best_score_this, best_score)
        return best

    def _smooth_paths(self, xy, heading, curvature, end, parameters):
        """Three smooth lateral parameters on the mutable interval; all rows use the published arc grid."""
        s = self.s
        x, y = xy.unbind(-1)
        ids = torch.arange(len(s),device=s.device)
        start = x.gather(1,end[:,None])
        length = (self.query.goal['x_min_m']-start).clamp_min(0.)
        u = ((x-start)/length.clamp_min(1e-6)).clamp(0.,1.)
        # The bump and its first two derivatives vanish at both ends.
        bump = (4*u*(1-u))**3
        z = 2*u-1
        a,b,c = parameters.unbind(-1)
        displacement = a[:,:,None]-y[:,None] + b[:,:,None]*z[:,None] + c[:,:,None]*z[:,None].square()
        # A second spatial derivative scales as 1/L². Four metres is the reference full window.
        scale = (length/4.).clamp_max(1.).square()
        yy = y[:,None] + scale[:,None]*bump[:,None]*displacement
        xx = x[:,None].expand_as(yy)
        # Anchor the new arc to the last immutable sample, not the beginning of a possibly resampled route.
        distances = torch.hypot(xx.diff(dim=2),yy.diff(dim=2))
        distances = torch.where(ids[:-1]>=end[:,None,None],distances,0.)
        arc = s[end,None,None]+torch.cat((distances[:,:,:1]*0,distances.cumsum(2)),2)
        target = s.expand_as(arc).contiguous()
        hi = torch.searchsorted(arc,target,right=True).clamp(1,len(s)-1);lo = hi-1
        prev = arc.gather(2,lo)
        frac = ((target-prev)/(arc.gather(2,hi)-prev).clamp_min(1e-9)).clamp(0,1)
        after = ids>end[:,None,None]
        x1 = torch.where(after,torch.lerp(xx.gather(2,lo),xx.gather(2,hi),frac)+(target-arc[:,:,-1:]).clamp_min(0),xx)
        y1 = torch.where(after,torch.lerp(yy.gather(2,lo),yy.gather(2,hi),frac),y[:,None])
        h = torch.atan2(torch.gradient(y1,dim=2)[0],torch.gradient(x1,dim=2)[0])
        k = torch.gradient(h,spacing=(s,),dim=(2,))[0]
        return (torch.stack((x1,y1),3).flatten(0,1).contiguous(),
                torch.where(after,h,heading[:,None]).flatten(0,1).contiguous(),
                torch.where(after,k,curvature[:,None]).flatten(0,1).contiguous())

    def _smooth_score(self,q,velocity,previous,xy,h,k,section,side,remain,gate,hold):
        args = (q,velocity,previous,xy,h,k,section,side,remain,gate,hold)
        if not self.fused:
            raw,passed,clear,valid = self._evaluate(*args)
        else:
            # A monitoring call must not overwrite the graph captured for a full CEM population.
            n = len(q)
            if n not in self.event_graphs:
                inputs = [arg.clone() for arg in args]
                self._evaluate(*inputs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=self._evaluate(*inputs)
                self.event_graphs[n] = inputs,graph,out
            inputs,graph,out = self.event_graphs[n]
            for buf,arg in zip(inputs,args):buf.copy_(arg)
            graph.replay()
            raw,passed,clear,valid = out
        # These raised-platform references advance in world x. Include the grid segment
        # containing the vehicle and every remaining segment, including the committed preview.
        forward_x = xy[:, :, 0].contiguous()
        current = torch.searchsorted(forward_x, q[:, :1].contiguous()).sub(1).clamp_min(0)
        remaining_path = torch.arange(k.shape[1], device=k.device) >= current
        remaining_k = torch.where(remaining_path, k, 0.)
        peak = remaining_k.abs().amax(1)
        finite = torch.isfinite(peak)
        geometry = finite & (peak<=1.)
        # Mean squared curvature over the remaining reference, in m⁻².
        # Admitted paths rank above every rejected one. The penalty only orders geometrically eligible paths.
        score = torch.where(geometry,raw-.05*remaining_k.square().sum(1)/remaining_path.sum(1),-1000.-torch.nan_to_num(peak,nan=1e6))
        return score,passed&geometry,clear,valid

    def _smooth_search(self,q,v,previous,xy,h,k,committed,section,side,remaining,gate,hold):
        n,b = len(q),self.candidates
        rows = torch.arange(n,device=q.device)
        end = torch.searchsorted(self.s,committed.contiguous()).clamp_max(len(self.s)-1)
        left = q.new_tensor([.36,0.,0.]);right = -left
        means = torch.stack((left,right)).expand(n,-1,-1).clone()
        std = q.new_tensor([.35,.25,.20]).expand_as(means).clone()
        repeated = [x.repeat_interleave(b,0) for x in (q,v,previous,section,side,remaining,gate,hold)]
        best_score = q.new_full((n,),-torch.inf)
        selected = [xy.clone(),h.clone(),k.clone(),torch.zeros(n,dtype=torch.bool,device=q.device),q.new_full((n,),-torch.inf),q.new_zeros(n)]
        for it in range(self.iterations):
            p = (means[:,:,None]+std[:,:,None]*self.noise[it,:,:,:3]).clamp(-1.25,1.25)
            # Exact seeds of this NEW family, not the three frozen legacy departure arrays.
            p[:,:,0]=means
            p[:,0,1]=left;p[:,0,2]=0.;p[:,1,1]=right
            bank = self._smooth_paths(xy,h,k,end,p.reshape(n,b,3))
            score,allowed,clear,valid = self._smooth_score(*repeated[:3],*bank,*repeated[3:])
            score = score.reshape(n,2,b//2)
            elite = score.argsort(dim=2,descending=True,stable=True)[:,:,:b//8]
            population = p.gather(2,elite[:,:,:,None].expand(-1,-1,-1,3))
            means = population.mean(2);std = population.std(2,unbiased=False).clamp_min(.06)
            # Rejected candidates can steer CEM, but they can never become a published plan.
            feasible_score = torch.where(allowed.reshape(n,b),score.flatten(1),-torch.inf)
            top,rank = feasible_score.max(1);take = top>best_score;index = rows*b+rank
            current = [a[index] for a in (*bank,allowed,clear,valid)]
            selected = [torch.where(take.reshape((n,)+(1,)*(a.ndim-1)),a,old) for a,old in zip(current,selected)]
            best_score = torch.maximum(best_score,top)
        return selected

    def _event_plan(self, q, velocity, previous, route, ids, fields):
        n=len(q);committed,section,side,remaining,gate,hold=fields
        bank=[t[ids].clone() for t in (route.xy,route.heading,route.curvature)]
        admitted=torch.zeros(n,device=q.device,dtype=torch.bool)
        clear=q.new_empty(n);valid=q.new_empty(n)
        for start in range(0,n,self.chunk):
            ix=torch.arange(start,start+self.chunk,device=q.device).clamp_max(n-1)
            _,ok,c,t=self._smooth_score(q[ix],velocity[ix],previous[ix],*[a[ix] for a in bank],
                                        section[ix],side[ix],remaining[ix],gate[ix],hold[ix])
            size=min(n-start,self.chunk);sl=slice(start,start+size)
            admitted[sl]=ok[:size];clear[sl]=c[:size];valid[sl]=t[:size]
        incumbent=admitted.clone()
        # A wrong first crossing cannot be undone. A still-valid incumbent needs no new geometry.
        searched=~incumbent & (gate!=-1)
        todo=torch.where(searched)[0]
        for start in range(0,len(todo),self.chunk):
            ix=todo[torch.arange(start,start+self.chunk,device=q.device).clamp_max(len(todo)-1)]
            out=self._smooth_search(q[ix],velocity[ix],previous[ix],*[a[ix] for a in bank],
                                    committed[ix],section[ix],side[ix],remaining[ix],gate[ix],hold[ix])
            size=min(len(todo)-start,self.chunk);ix=ix[:size];found=out[3][:size]
            for dest,value in zip(bank,out[:3]):dest[ix]=value[:size]
            admitted[ix]=found
            clear[ix]=torch.where(found,out[4][:size],clear[ix])
            valid[ix]=torch.where(found,out[5][:size],valid[ix])
        return ExitProposal(*bank,admitted,incumbent,clear,valid,searched)

    def _run_evaluate(self, args):
        if not self.fused:
            return self._evaluate(*args)
        if self.graph is None:
            self.inputs = [arg.clone() for arg in args]
            self._evaluate(*self.inputs)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self._evaluate(*self.inputs)
        else:
            for buffer, arg in zip(self.inputs, args):
                buffer.copy_(arg)
        self.graph.replay()
        return self.output

    @torch.inference_mode()
    def plan(self, q, velocity, previous, route, committed_s, section, side, remaining_s,
             *, gate_status=0, hold_s=0., rows=None):
        """q is native qpos; twists are 9-D body states separated by 20 ms.

        The route bank uses a common progress grid. rows contains its route IDs
        for these q rows. gate_status is 0 before crossing, +1/-1 after a correct/
        wrong first crossing. hold_s is the actual unbroken goal hold at observation.
        """
        n = len(q)
        ids = route.route_ids if rows is None else rows
        fields = [torch.as_tensor(t, device=q.device, dtype=q.dtype).reshape(-1).expand(n)
                  for t in (committed_s, section, side, remaining_s, gate_status, hold_s)]
        if self.mode=='smooth_event':
            return self._event_plan(q,velocity,previous,route,ids,fields)
        result = []
        for start in range(0, n, self.chunk):
            index = torch.arange(start, start + self.chunk, device=q.device).clamp_max(n - 1)
            chosen = ids[index]
            args = [v[index] for v in (q, velocity, previous)]
            args += [v[chosen] for v in (route.xy, route.heading, route.curvature)]
            args += [v[index] for v in fields]
            out = self._solve(*args)
            count = min(self.chunk, n - start)
            result.append([v[:count].clone() for v in out])
        return ExitProposal(*(torch.cat(values, 0) for values in zip(*result)))
