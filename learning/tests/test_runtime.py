import jax
from sttw_control.runtime import configure_compilation_cache


def test_default_cache_and_explicit_user_override(tmp_path):
    old=jax.config.jax_compilation_cache_dir
    enabled=jax.config.jax_enable_compilation_cache
    try:
        jax.config.update('jax_enable_compilation_cache',True)
        jax.config.update('jax_compilation_cache_dir',None)
        fallback=tmp_path/'fallback'
        assert configure_compilation_cache(fallback)==str(fallback.resolve())
        explicit=str(tmp_path/'user_cache')
        jax.config.update('jax_compilation_cache_dir',explicit)
        assert configure_compilation_cache(fallback)==explicit
        jax.config.update('jax_enable_compilation_cache',False)
        assert configure_compilation_cache(fallback) is None
        assert jax.config.jax_compilation_cache_dir==explicit
    finally:
        jax.config.update('jax_compilation_cache_dir',old)
        jax.config.update('jax_enable_compilation_cache',enabled)
