"""
An amazing circuit generator
"""

__version__ = "0.2.0"
try:
    import jax
    from jax import config

    config.update("jax_enable_x64", True)
    use_jax = True  # The wFunction's source of truth on jax availability.
    # Never import or use Jax without first checking use_jax.
except:
    use_jax = False

from .Generate_circuit import (
    Generate_f_circuit,
    Generate_f_gate,
)
from .scalarQubitization import qubitize_scalar as qubitize_scalar
