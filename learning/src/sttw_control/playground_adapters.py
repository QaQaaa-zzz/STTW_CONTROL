"""Ground plan publication and controller-state capture for the shared runner."""
from copy import deepcopy


class GroundAdapter:
    def __init__(self, middle, controller):
        self.middle=middle
        self.controller=controller
        self.plan_version=0

    def update_plan(self, route, committed_until, valid_from_time, expected_previous_version):
        """Publish at the current 20 ms boundary; retain the fixed speed/alpha domain."""
        if expected_previous_version!=self.plan_version:
            raise ValueError('stale previous plan version')
        if abs(valid_from_time-self.middle.tick*.005)>1e-8:
            raise ValueError('plan publication time differs from the execution clock')
        self.middle.update_plan(route,committed_until)
        self.plan_version+=1
        return dict(version=self.plan_version,previous_version=expected_previous_version,
                    valid_from_time=valid_from_time,tick=self.middle.tick)

    def snapshot(self):
        """Capture controller state on its current device, excluding frozen weights.

        This is ControllerState only. The runner must additionally capture its
        complete physics data, RB FIFO/actions, brake targets and phase timers.
        """
        m=self.middle
        return dict(version=self.plan_version,route=m.route,
            route_config=deepcopy(m.route.config),
            middle=deepcopy({k:v for k,v in vars(m).items() if k not in ('route','policy')}),
            policy=deepcopy({k:v for k,v in vars(m.policy).items() if k not in ('weights','config')}),
            controller=deepcopy(vars(self.controller)))

    def restore(self, state):
        m=self.middle;policy=m.policy
        m.__dict__.clear()
        m.__dict__.update(deepcopy(state['middle']),route=state['route'],policy=policy)
        m.route.config=deepcopy(state['route_config'])
        weights,config=policy.weights,policy.config
        policy.__dict__.clear()
        policy.__dict__.update(deepcopy(state['policy']),weights=weights,config=config)
        self.controller.__dict__.clear()
        self.controller.__dict__.update(deepcopy(state['controller']))
        self.plan_version=state['version']
