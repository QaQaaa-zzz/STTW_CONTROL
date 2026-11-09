"""Batched, finite-horizon off-road jump reward. All lengths SI, angles radians.

One instance per batch of independent lanes. Call at a FIXED sample period,
preferably every physics substep. Roll/pitch and v/vn refer to the local
terrain frame. near/target are actual supporting contacts on assigned patches.
For a ditch, s is longitudinal distance from the near lip for each wheel,
clearance is wheel-bottom distance above the local/virtual obstacle envelope.
The caller owns collision/fall detection and hard finite-horizon expiry.
Use PPO gamma=1, no per-transition clipping, no bootstrap at either task end.
"""
import math
import torch


class JumpReward:
    def __init__(self, n, device, dt=.02, width=.30, hold_s=1., approach_distance=1.8, rough_support=False):
        if min(dt, width, hold_s, approach_distance) <= 0:
            raise ValueError("Positive time and length scales required")
        self.dt, self.width = dt, width
        self.approach_distance = approach_distance
        self.flight_samples = math.ceil(.04 / dt - 1e-8) + 1
        self.hold_samples = math.ceil(hold_s / dt - 1e-8) + 1
        self.hold_s = hold_s
        self.rough_support = rough_support
        self._state = {}
        for name in ('near_ticks', 'air_ticks', 'touch_ticks', 'hold_ticks', 'stage'):
            self._state[name] = torch.zeros(n, device=device, dtype=torch.long)
        for name in ('armed', 'attempt', 'cleared', 'landed', 'done', 'success', 'seen', 'hard_touch', 'restart_window'):
            self._state[name] = torch.zeros(n, device=device, dtype=torch.bool)
        for name in ('phi', 'hold_start', 'quality_sum', 'lift_start', 'prev_vn'):
            self._state[name] = torch.zeros(n, device=device)
        self._state['prev_s'] = torch.zeros((n, 2), device=device)
        self._state['crossed_middle'] = torch.zeros((n, 2), device=device, dtype=torch.bool)
        self._state['touched'] = torch.zeros((n, 2), device=device, dtype=torch.bool)
        if rough_support:
            for name in ('far_age', 'near_age', 'wheel_duty'):
                self._state[name] = torch.zeros((n,2), device=device, dtype=torch.long)
            for name in ('both_free_age', 'any_duty', 'quiet_duty'):
                self._state[name] = torch.zeros(n, device=device, dtype=torch.long)
            self._state['near_seen'] = torch.zeros((n,2), device=device, dtype=torch.bool)
        self.__dict__.update(self._state)
        self.values = torch.tensor([0., 20., 50., 70., 100.], device=device)

    def reset(self, ids):
        for field in self._state.values():
            field[ids] = 0

    @torch.no_grad()
    def step(self, *, s, clearance, loaded, near, target, lane, v, vn, roll, pitch,
             roll_rate, pitch_rate, failed, timeout, clearance_upper=None):
        live = ~self.done
        previous_stage = self.stage.clone()
        previous_phi = self.phi.clone()
        scalars = torch.stack((v, vn, roll, pitch, roll_rate, pitch_rate), -1)
        upper = clearance if clearance_upper is None else clearance_upper
        finite = (torch.isfinite(upper).all(-1) & torch.isfinite(s).all(-1) & torch.isfinite(clearance).all(-1)
                  & torch.isfinite(scalars).all(-1))
        s, clearance = torch.nan_to_num(s), torch.nan_to_num(clearance)
        v, vn, roll, pitch, roll_rate, pitch_rate = torch.nan_to_num(scalars).unbind(-1)
        # At this sample rate a wheel must actually visit the obstacle interior.
        # A discontinuity is a bad transition, never proof of crossing.
        discontinuity = self.seen & ((s-self.prev_s).abs() > 12*self.dt+.002).any(-1)
        bad = failed.bool() | ~finite | discontinuity
        valid = live & ~bad
        loaded, near, target, lane = (x.bool() for x in (loaded, near, target, lane))
        near = near & loaded & (s <= 0)
        target = target & loaded & lane & (s >= self.width)
        grounded_near = near.all(-1) & lane.all(-1)
        near_roll = grounded_near
        if self.rough_support:
            self.near_seen |= near
            self.near_age.copy_(torch.where(near, 0, self.near_age+1))
            near_roll = (self.near_seen.all(-1) & (self.near_age*self.dt <= .08+1e-7).all(-1)
                         & (s <= 0).all(-1) & lane.all(-1) & (upper <= .05).all(-1))
        self.near_ticks.copy_(torch.where(near_roll & valid, self.near_ticks+1, 0))
        self.armed |= (self.near_ticks >= self.flight_samples) & near.any(-1)

        lip_window = (s[:, 0] >= -1.2) & (s[:, 0] < self.width)
        flying = (~loaded.any(-1) & (clearance > .015).all(-1)
                  & lane.all(-1) & (v > 0) & valid)
        # Only the onset must lie in the launch window; it may cross the lip during debounce.
        air_candidate = self.armed & flying & (lip_window | (self.air_ticks > 0))
        self.lift_start.copy_(torch.where(air_candidate & (self.air_ticks == 0), s.mean(-1), self.lift_start))
        self.air_ticks.copy_(torch.where(air_candidate, self.air_ticks+1, 0))
        launch = (self.air_ticks >= self.flight_samples) & ~self.attempt
        self.attempt |= launch
        self.armed &= ~launch

        in_middle = (s >= .2*self.width) & (s <= .8*self.width)
        good_middle = ~loaded & (clearance >= .005) & lane
        # Track the front wheel even during the short bilateral-flight debounce,
        # and during the front-first departure while the rear wheel still pushes.
        tracking = (self.armed | self.attempt) & valid
        self.crossed_middle |= in_middle & good_middle & tracking[:, None]
        wrong_contact = (loaded & ~near & ~target).any(-1)
        abort = ((self.attempt & near.any(-1)) | (tracking & ~lane.all(-1))
                 | (tracking & ((in_middle & ~good_middle).any(-1) | wrong_contact))
                 | (self.cleared & (s < self.width).any(-1)))
        # Re-arming requires fresh near support; cancel stale flight when leaving the strip.
        self.attempt &= ~abort
        self.armed &= ~abort
        self.air_ticks.masked_fill_(abort, 0)
        self.crossed_middle &= ~(abort | grounded_near)[:, None]
        self.cleared &= ~abort
        self.landed &= ~abort
        self.touched &= ~abort[:, None]
        self.hard_touch &= ~abort
        self.cleared |= (self.attempt & self.crossed_middle.all(-1)
                         & (s >= self.width).all(-1) & valid)
        first_target = self.attempt[:, None] & target & ~self.touched & valid[:, None]
        self.touched |= first_target
        # Pre-contact normal closing speed is a root-speed proxy, not a contact impulse.
        self.hard_touch |= first_target.any(-1) & (torch.minimum(self.prev_vn, vn) < -3.5)
        double_support = self.cleared & target.all(-1) & valid & ~self.hard_touch
        rolling = double_support
        if self.rough_support:
            self.far_age.copy_(torch.where(target, 0, self.far_age+1))
            self.both_free_age.copy_(torch.where(target.any(-1), 0, self.both_free_age+1))
            rolling = (self.cleared & self.touched.all(-1) & valid & ~self.hard_touch
                       & (self.far_age*self.dt <= .08+1e-7).all(-1)
                       & (self.both_free_age*self.dt <= .04+1e-7)
                       & (target | (upper <= .05)).all(-1) & lane.all(-1))
        captured = (rolling & (v >= .5) & (v <= 2.8) & (vn.abs() <= .8)
                    & (roll.abs() <= .30) & (pitch.abs() <= .30)
                    & (roll_rate.abs() <= 2.5) & (pitch_rate.abs() <= 2.5))
        self.touch_ticks.copy_(torch.where(captured, self.touch_ticks+1, 0))
        self.landed |= self.touch_ticks >= self.flight_samples

        # Update stage on valid samples only. A collision on the same sample
        # always takes precedence. The best demonstrated stage is paid only once.
        toward_lip = self.attempt & (s[:, 0] >= 0) & (s.mean(-1)-self.lift_start >= .05)
        for passed, k in ((toward_lip, 1), (self.cleared, 2), (self.landed, 3)):
            self.stage.copy_(torch.where(valid & passed & (self.stage < k), k, self.stage))
        quiet = (vn.abs() <= .5) & (roll_rate.abs() <= 1.5) & (pitch_rate.abs() <= 1.5)
        stable = rolling & (v >= 1.) & (v <= 2.4) & (roll.abs() <= .15) & (pitch.abs() <= .15)
        self.hold_ticks.masked_fill_(self.restart_window, 0)
        self.quality_sum.masked_fill_(self.restart_window, 0.)
        if self.rough_support:
            stable &= (vn.abs() <= 1.) & (roll_rate.abs() <= 4.) & (pitch_rate.abs() <= 4.)
            # Each candidate window spans one full second. Discard a completed
            # window that lacked duty, actual travel, or final physical support.
            self.wheel_duty.masked_fill_(self.restart_window[:, None], 0)
            self.any_duty.masked_fill_(self.restart_window, 0)
            self.quiet_duty.masked_fill_(self.restart_window, 0)
            self.wheel_duty.copy_(torch.where(stable[:,None], self.wheel_duty+target.long(), 0))
            self.any_duty.copy_(torch.where(stable, self.any_duty+target.any(-1).long(), 0))
            self.quiet_duty.copy_(torch.where(stable, self.quiet_duty+quiet.long(), 0))
        else:
            stable &= quiet
        center = s.mean(-1)
        begin = stable & (self.hold_ticks == 0)
        self.hold_start.copy_(torch.where(begin, center, self.hold_start))
        self.hold_ticks.copy_(torch.where(stable, self.hold_ticks+1, 0))
        def interval_quality(value, low, high, scale):
            distance = torch.relu(low-value) + torch.relu(value-high)
            return torch.exp(-(distance/scale).square())
        q_entry = interval_quality(v, 1.6, 2.2, .45) * torch.exp(-(roll/.25).square())
        q_land = (interval_quality(v, 1.2, 2., .4) + torch.exp(-(roll/.15).square())
                  + torch.exp(-(pitch/.15).square()) + torch.exp(-(vn/.5).square())) / 4
        self.quality_sum.copy_(torch.where(stable, self.quality_sum+q_land, 0.))
        # One second's worth of actual forward displacement as well as speed.
        # Fifty-one 50 Hz samples span exactly one elapsed second.
        arrived = (self.landed & stable & (self.hold_ticks >= self.hold_samples)
                   & (center-self.hold_start >= .9*self.hold_s) & valid)
        if self.rough_support:
            h = self.hold_ticks
            arrived &= ((100*self.wheel_duty >= 35*h[:,None]).all(-1) & (100*self.any_duty >= 65*h)
                        & (100*self.quiet_duty >= 90*h) & target.any(-1))
        self.restart_window.copy_((self.hold_ticks >= self.hold_samples) & ~arrived)
        self.success |= arrived
        self.stage.masked_fill_(arrived, 4)
        ended = live & (bad | arrived | timeout.bool())
        # Success replaces the stage balance; the terminal quality is in [0,10].
        balance = self.values[self.stage] - self.values[previous_stage]
        quality = 10*self.quality_sum / self.hold_ticks.clamp_min(1)
        outcome = torch.where(bad, -100., torch.where(arrived, quality,
                              torch.where(timeout.bool(), -95., 0.)))
        approach = ((s[:, 0]+self.approach_distance)/self.approach_distance).clamp(0., 1.)
        across = ((s.min(-1).values+1.2)/(1.2+self.width)).clamp(0., 1.)
        phi_now = torch.where(self.cleared, 3*q_land+3*(self.hold_ticks/self.hold_samples).clamp(max=1),
                    torch.where(self.attempt, 6*across, 6*approach*q_entry))
        self.phi.copy_(torch.where(valid & ~ended, phi_now, 0.))
        reward = torch.where(live, balance+outcome+self.phi-previous_phi, 0.)
        self.done |= ended
        self.prev_s.copy_(s)
        self.prev_vn.copy_(vn)
        self.seen |= live
        return reward, self.done.clone(), self.success.clone(), self.stage.clone()
