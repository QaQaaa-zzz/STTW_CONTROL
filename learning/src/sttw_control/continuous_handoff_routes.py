"""Use each vehicle's published approach to construct its three frozen exit geometries at handback."""
from copy import copy
import torch


class ContinuousHandoffRoutes:
    def __init__(self, initial, goal_x):
        self.goal_x = float(goal_x)
        lanes = len(initial.route_ids)
        self.lanes = torch.arange(lanes, device=initial.s.device)
        self.branch_rows = lanes + 3*self.lanes[:, None] + torch.arange(3, device=initial.s.device)
        self.route = copy(initial)
        self.route.route_ids = self.lanes.clone()
        for key in ('xy', 'heading', 'curvature'):
            data = getattr(initial, key)[initial.route_ids]
            setattr(self.route,key,torch.cat((data,data.repeat_interleave(3,dim=0))))
        self.published_tick = torch.full_like(self.lanes, -1)
        self.handoff_join = torch.full_like(self.lanes, -1)
        self.geometry_ms = torch.zeros_like(self.lanes,dtype=torch.float64)

    def publish(self, adapter, proposal, committed, ready, deadline, eligible, expected_version):
        """Publish to approach lanes through the ordinary H75 same-grid prefix and clock check."""
        middle = adapter.middle
        if not torch.equal(self.route.s, proposal.s):
            raise ValueError('The proposed approach must use the currently published progress grid')
        now = middle.tick*.005
        guard = (middle.projection['progress'] + middle.follower['lookahead'] + .02*middle.v_user).clamp_max(self.route.s[-1])
        on_time = eligible & (ready<=deadline) & (now<=deadline) & (ready<=now+1e-8)
        accepted = on_time & (committed>=guard)
        ids = self.lanes[accepted]
        receipt = dict(tick=middle.tick,version=adapter.plan_version,
                       late=self.lanes[eligible & ~on_time].tolist(),
                       commitment_expired=self.lanes[on_time & ~accepted].tolist(),published=ids.tolist())
        if not len(ids):return receipt
        proposal_ids = proposal.route_ids
        active = copy(self.route)
        for key in ('xy','heading','curvature'):
            table = getattr(self.route,key).clone()
            table[ids] = getattr(proposal,key)[proposal_ids[ids]]
            setattr(active,key,table)
        # Other lanes (including vehicles that have already jumped) keep their full references intact.
        receipt.update(adapter.update_plan(active,torch.where(accepted,committed,self.route.s[-1]),now,expected_version))
        self.route = middle.route
        self.published_tick[ids] = middle.tick
        return receipt

    def at_handback(self, ids, position, progress):
        """Write actual left/straight/right references into private lane rows; active selection stays external."""
        route = self.route
        rows = route.route_ids[ids]
        s = route.s.double()
        xy = route.xy[rows].double()
        x, original_y = xy.unbind(-1)
        after_x = torch.searchsorted(x.contiguous(),(position[:,0].double()+.05)[:,None]).squeeze(1)
        join = torch.maximum(after_x,torch.searchsorted(s,progress.double().contiguous())+1).clamp_max(len(s)-1)
        grid = torch.arange(len(s),device=s.device)
        x0 = x.gather(1,join[:,None])
        target = x.new_tensor([.36,0.,-.36])
        # Same quintic as departure_route, independently for the three known directions.
        def blend(u):
            u=u.clamp(0.,1.)
            return u**3*(10.-15.*u+6.*u**2)
        w = blend((x-x0)/1.6)*(1.-blend((x-15.)/(self.goal_x-15.)))
        y = original_y[:,None] + (target[None,:,None]-original_y[:,None])*w[:,None]
        distance = torch.hypot(x.diff(dim=-1)[:,None],y.diff(dim=-1))
        distance = torch.where(grid[:-1]>=join[:,None,None],distance,0.)
        arc = s[join][:,None,None] + torch.cat((torch.zeros_like(distance[:,:,:1]),distance.cumsum(-1)),-1)
        # All three arcs share the fixed output sample grid. np.interp clamps outside the domain.
        query = s.expand_as(arc).contiguous()
        upper = torch.searchsorted(arc,query,right=True).clamp_(1,len(s)-1)
        lower = upper-1
        arc_lo = arc.gather(2,lower)
        fraction = ((query-arc_lo)/(arc.gather(2,upper)-arc_lo).clamp_min(1e-24)).clamp(0.,1.)
        def interp(v):
            a=v.gather(2,lower)
            return a+fraction*(v.gather(2,upper)-a)
        post = grid>join[:,None,None]
        xx = torch.where(post,interp(x[:,None].expand_as(y))+(query-arc[:,:,-1:]).clamp_min(0.),x[:,None])
        yy = torch.where(post,interp(y),original_y[:,None])
        heading = torch.atan2(torch.gradient(yy,dim=-1)[0],torch.gradient(xx,dim=-1)[0])
        curvature = torch.gradient(heading,spacing=(s,),dim=(-1,))[0]
        exits = self.branch_rows[ids]
        route.xy[exits] = torch.stack((xx,yy),-1).to(route.xy)
        route.heading[exits] = torch.where(post,heading,route.heading[rows,None]).to(route.heading)
        route.curvature[exits] = torch.where(post,curvature,route.curvature[rows,None]).to(route.curvature)
        self.handoff_join[ids] = join
