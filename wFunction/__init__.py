"""
An amazing circuit generator
"""

__version__ = "0.2.0"


from .Generate_circuit import (
    Generate_f_circuit,
    Generate_f_gate,
    )
from .scalarQubitization import qubitize as qubitize_scalar

