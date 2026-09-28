"""Explicit precision migration of the complete immutable MJX snapshot.

No reset, forward call, physics step or controller update occurs here. Original
floating point VALUES and every integer index are retained exactly. Float64
reduces batching arithmetic drift; it is not a relaxed acceptance tolerance.
"""
import jax
import jax.numpy as jp
import numpy as np


def promote_snapshot(snapshot):
    if not jax.config.x64_enabled:raise ValueError('float64 migration requires JAX_ENABLE_X64=true before model loading')
    converted=jax.tree.map(lambda x:jp.asarray(x,dtype=jp.float64) if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.floating) else x,snapshot)
    # JAX64 MJX collision generation produces these int64 arrays. Keep other
    # integer/bool state as-is, including controller ticks and solver metadata.
    contact=converted.data._impl.contact
    contact=contact.replace(geom1=jp.asarray(contact.geom1,jp.int64),
        geom2=jp.asarray(contact.geom2,jp.int64),geom=jp.asarray(contact.geom,jp.int64))
    converted=converted.replace(data=converted.data.replace(_impl=converted.data._impl.replace(contact=contact)))
    old=jax.tree.leaves(snapshot);new=jax.tree.leaves(converted)
    if len(old)!=len(new) or not all(np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True) for a,b in zip(old,new)):
        raise ValueError('precision migration changed snapshot values')
    return converted
