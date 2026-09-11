"""Local compilation reuse without changing user-supplied JAX settings."""
from pathlib import Path
import jax


def configure_compilation_cache(fallback=None):
    if not jax.config.jax_enable_compilation_cache:
        return None
    if jax.config.jax_compilation_cache_dir is None:
        directory=Path(fallback) if fallback is not None else Path(__file__).resolve().parents[3]/'runs'/'.jax_compilation_cache'
        jax.config.update('jax_compilation_cache_dir',str(directory.resolve()))
    return jax.config.jax_compilation_cache_dir
