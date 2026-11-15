"""Opt-in fixed-order GPU body-tree reductions for snapshot numerical diagnosis."""
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import smooth
from mujoco.mjx.third_party.mujoco_warp._src.types import vec10


@wp.kernel
def subtree_com(body_parentid: wp.array(dtype=int),
                values: wp.array2d(dtype=wp.vec3),
                tree: wp.array(dtype=int),
                output: wp.array2d(dtype=wp.vec3)):
    world, node = wp.tid()
    if node == 0:
        for i in range(tree.shape[0]):
            body = tree[i]
            if body != 0:
                parent = body_parentid[body]
                output[world, parent] = output[world, parent] + values[world, body]


@wp.kernel
def composite_inertia(body_parentid: wp.array(dtype=int),
                      values: wp.array2d(dtype=vec10),
                      tree: wp.array(dtype=int),
                      output: wp.array2d(dtype=vec10)):
    world, node = wp.tid()
    if node == 0:
        for i in range(tree.shape[0]):
            body = tree[i]
            parent = body_parentid[body]
            if parent != 0:
                output[world, parent] = output[world, parent] + values[world, body]


def enable():
    # Each invocation handles one tree level. Worlds remain parallel on the GPU.
    smooth._subtree_com_acc = subtree_com
    smooth._crb_accumulate = composite_inertia
