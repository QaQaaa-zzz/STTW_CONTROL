"""Lookup of physically observed exit outcomes for a vetoed platform handback."""
import torch


def handback_features(q, v):
    # Position, quaternion vector (positive-w convention), steer and free-body velocities.
    orientation = q[:, 4:7] * torch.where(q[:, 3:4] < 0, -1., 1.)
    return torch.cat((q[:, :2], orientation, q[:, 8:9], v[:, :6]), dim=1)


class VetoExitBank:
    def __init__(self, bank, device, keep=slice(None)):
        self.state, self.route, self.wall, self.done = (
            torch.as_tensor(bank[k][keep], device=device) for k in ('state', 'route', 'wall', 'done'))
        self.scale = self.state.std(0, unbiased=False).clamp_min(.015)

    def choose(self, state, route, section, incumbent):
        distance = ((state[:, None] - self.state) / self.scale).square().sum(-1)
        distance = distance.masked_fill(route[:, None] != self.route, torch.inf)
        dist, near = distance.topk(5, largest=False)
        valid = dist.isfinite().unsqueeze(-1)
        walls = (self.wall[near] * valid).sum(1)
        wins = (self.done[near, section[:, None]] * valid).sum(1)
        # Each wall count outranks all five completion votes. Keep incumbent on equal votes.
        score = wins - 6 * walls
        old = incumbent[:, None]
        improves = (walls < walls.gather(1, old)) & (wins >= wins.gather(1, old))
        # The current fallback is the default when no alternative dominates its votes.
        score = score.masked_fill(~improves, -100)
        score.scatter_(1, old, (wins - 6 * walls).gather(1, old))
        best = score.argmax(1)
        retain = score.gather(1, incumbent[:, None]).squeeze(1) == score.amax(1)
        return torch.where(retain, incumbent, best)
