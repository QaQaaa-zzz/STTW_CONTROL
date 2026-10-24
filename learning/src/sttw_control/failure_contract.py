"""Separate task termination from a finite physical no-contact observation.

This opt-in accounting primitive does not change an environment or controller.
Callers supply episode-latched events and a fixed physical observation horizon.
A domain exit requires a separately specified continuation controller/observation;
these flags alone do not create valid continuation or valid bootstrap data.
"""
import jax.numpy as j


def split_failure_events(*, contact_failure, engineering_fault,
                         tracking_domain_exit, legacy_roll_crossed, time_limit):
    contact, fault, domain, roll, timeout = j.broadcast_arrays(*[
        j.asarray(x, dtype=j.bool_) for x in [contact_failure, engineering_fault,
            tracking_domain_exit, legacy_roll_crossed, time_limit]])
    physical_done = contact | fault | timeout
    task_done = domain | physical_done
    return dict(task_done=task_done, physical_done=physical_done,
                task_bootstrap=~task_done, risk_bootstrap=~physical_done,
                fell=contact & ~fault, unknown=fault,
                survived_window=timeout & ~contact & ~fault,
                legacy_roll_crossed=roll, tracking_domain_exit=domain)
