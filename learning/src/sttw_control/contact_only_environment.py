"""Explicit diagnostic termination adapter; physics/control evolution is unchanged."""

class ContactOnlyPhysics:
    """Retain a roll-domain crossing as telemetry and stop on real contact/fault."""
    def __init__(self,original):
        self.original=original

    def __getattr__(self,name):
        return getattr(self.original,name)

    def _step(self,physical,*args,**kwargs):
        end,log=self.original._step(physical,*args,**kwargs)
        contact=log['contact_failure']|self.original.helpers._contact_failure(end.data)
        end=end.replace(failed=physical.failed|log['nonfinite']|contact)
        log=dict(log,contact_failure=contact,physical_failure=end.failed)
        return end,log


def load_environment_contact_only(config,source,terminal_contract):
    from .path_command_training import load_environment
    env,initial=load_environment(config,source,terminal_contract)
    env.physics=ContactOnlyPhysics(env.physics)
    return env,initial
