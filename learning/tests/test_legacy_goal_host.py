"""History preservation of the new host's geometry-only handoff."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as j
from flax import struct
import numpy as np
import pytest

file=Path(__file__).parents[1]/'src/sttw_control/legacy_goal_run.py'
spec=importlib.util.spec_from_file_location('_legacy_goal_host_test',file)
host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)


@struct.dataclass
class Projection:
    progress: object
    cross_track: object


@struct.dataclass
class State:
    physical: object
    projection: object
    follower: object
    history: object
    offsets: object
    tick: object
    alpha: object
    domain_exit: object


@struct.dataclass
class Physical:
    data: object
    observer: object


@struct.dataclass
class Carry:
    state: object
    gate: object
    clean_blocks: object


def fixture():
    state=State(Physical(j.array([4.,2.,.3]),j.array([1.,2.,3.])),Projection(j.float32(8),j.float32(2)),
        j.array([7.,8.]),j.arange(12).reshape(3,4),j.array([.12,-.1]),j.int32(600),j.float32(.75),j.bool_(True))
    return Carry(state,j.array([9.,10.]),j.int32(16))


def test_rebinding_changes_only_geometry_and_keeps_handback_history():
    before=fixture()
    env=SimpleNamespace(config={},physics=SimpleNamespace(observe=lambda data:(None,data,j.float32(1.2))))
    project=lambda path,xy,previous,velocity,cfg:Projection(j.float32(0),xy[1]-path[1])
    pursuit=lambda path,s,pose,v,user,cfg:j.array([user,path[0]+pose[2]])
    one=host.make_geometry_rebind(env,project,pursuit)
    after=one(before,j.array([5.,1.]),j.float32(1.8))
    assert float(after.state.projection.progress)==0
    np.testing.assert_allclose(after.state.follower,[1.8,5.3])
    host.assert_geometry_only(before,after)
    assert int(after.clean_blocks)==16 and bool(after.state.domain_exit)


def test_history_reset_is_rejected_even_when_pose_and_geometry_are_valid():
    before=fixture()
    altered=before.replace(state=before.state.replace(history=j.zeros_like(before.state.history)))
    with pytest.raises(ValueError,match='non-geometric'):
        host.assert_geometry_only(before,altered)


def test_clean_counter_reset_is_rejected_by_same_carry_audit():
    before=fixture()
    with pytest.raises(ValueError,match='non-geometric'):
        host.assert_geometry_only(before,before.replace(clean_blocks=j.int32(0)))


def test_inactive_observation_fault_cannot_be_reported_as_completed():
    # The legacy lower scan can return all inactive logs after an upper
    # observation fault; inspecting active log flags alone misses this stop.
    flags=host.unexplained_boundary_stop([True,False,False],[False,True,False],[False,True,False])
    np.testing.assert_array_equal(flags,[True,False,False])


def test_unexplained_physical_stop_stays_unknown_until_contact_is_observed():
    np.testing.assert_array_equal(host.unexplained_boundary_stop([False],[True],[False]),[True])
