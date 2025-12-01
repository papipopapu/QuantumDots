"""
QuantumDots Library

Core modules for quantum dot simulations:
- hamiltonian: Second quantization framework for constructing Hamiltonians
- density: Lindblad master equation and density matrix tools
- qutipDots: QuTiP-based numerical simulation tools
"""

from .hamiltonian import (
    c_internal,
    Space,
    calc_Hamiltonian,
    calc_vac,
    conjugate,
    delta_site,
    split_c_list
)

from .density import (
    get_Lamb_matrix,
    get_density_equation,
    get_density_equation_vect
)

from .qutipDots import (
    f_destroy,
    f_create,
    eqdot_state,
    get_Lambda,
    get_Liouville,
    red_H_idx,
    red_H,
    get_state_index
)
