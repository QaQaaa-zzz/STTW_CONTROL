"""Complete same-model Playground branch snapshots, kept on device in memory."""
from copy import deepcopy
import jax


def capture(data, ground_adapter, rb_state, runtime, tick, context):
    # Copy every MJX leaf, including contact/constraint and warm-start fields.
    return dict(physics=jax.tree.map(lambda x:x.copy(),data),
                ground=ground_adapter.snapshot(),
                rb=jax.tree.map(lambda x:x.copy(),rb_state),
                runtime=deepcopy(runtime),tick=tick,context=deepcopy(context))


def restore(snapshot, ground_adapter):
    ground_adapter.restore(snapshot['ground'])
    return (jax.tree.map(lambda x:x.copy(),snapshot['physics']),
            jax.tree.map(lambda x:x.copy(),snapshot['rb']),
            deepcopy(snapshot['runtime']))
