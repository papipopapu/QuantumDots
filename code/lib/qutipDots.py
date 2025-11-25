"""
QuTiP-based quantum dot simulation module.

This module provides numerical tools for simulating quantum dot systems
using the QuTiP (Quantum Toolbox in Python) library. It implements:

- Fermionic operators with proper anticommutation relations
- State construction for multi-site quantum dot systems
- Hamiltonian reduction to relevant subspaces
- Liouvillian construction for open system dynamics

The module is designed to work with QuTiP's solvers for both
unitary evolution and Lindblad master equation dynamics.
"""

import numpy as np
from qutip import tensor, basis, sigmaz, destroy, identity, create, jmat

from typing import List
from numpy.typing import ArrayLike


def f_destroy(N: int, i: int):
    """
    Create a fermionic annihilation operator for site i in an N-site system.
    
    Fermionic operators must satisfy the anticommutation relations:
        {c_i, c_j†} = δ_ij
        {c_i, c_j} = 0
    
    This is achieved using the Jordan-Wigner transformation:
        c_i = (⊗_{j<i} σ_z^j) ⊗ σ_-^i ⊗ (⊗_{j>i} I^j)
    
    The σ_z string provides the necessary sign for fermionic statistics.
    
    Args:
        N: Total number of fermionic sites
        i: Index of the site (0-indexed)
        
    Returns:
        Qobj: The fermionic annihilation operator c_i
    """
    # Jordan-Wigner transformation: σ_z string for sites j < i
    return tensor([sigmaz()] * i + [destroy(2)] + [identity(2)] * (N - i - 1))


def f_create(N: int, i: int):
    """
    Create a fermionic creation operator for site i in an N-site system.
    
    This is simply the Hermitian conjugate of the annihilation operator.
    
    Args:
        N: Total number of fermionic sites
        i: Index of the site (0-indexed)
        
    Returns:
        Qobj: The fermionic creation operator c_i†
    """
    return f_destroy(N, i).dag()


def eqdot_state(occupations: List[bool]):
    """
    Create a Fock state for a quantum dot system.
    
    Constructs a tensor product state where each site is either
    occupied (1) or empty (0).
    
    Args:
        occupations: List of occupation numbers [n_1, n_2, ..., n_N]
                     where n_i = 0 or 1
                     
    Returns:
        Qobj: The Fock state |n_1, n_2, ..., n_N⟩
    """
    N = len(occupations)
    return tensor([basis(2, occupations[i]) for i in range(N)])


def get_Lambda(Gamma: ArrayLike) -> ArrayLike:
    """
    Calculate the Lamb shift matrix from transition rates.
    
    For quantum dots coupled to reservoirs, the Lamb shift matrix
    describes the decay of coherences due to the finite coupling
    to the environment.
    
    Λ_mn = (1/2) Σ_{k≠n} (Γ_km + Γ_kn)
    
    Args:
        Gamma: Transition rate matrix (NumPy array)
        
    Returns:
        ArrayLike: The Lamb shift matrix
    """
    D = Gamma.shape[0]
    Lambda = np.zeros((D, D))
    
    for m in range(D):
        for n in range(D):
            Lambda_mn = 0
            for k in range(D):
                if k != n:
                    Lambda_mn += 1/2 * (Gamma[k, m] + Gamma[k, n])
            Lambda[m, n] = Lambda_mn
    
    return Lambda


def get_Liouville(Gamma: ArrayLike, H: ArrayLike) -> ArrayLike:
    """
    Construct the Liouvillian superoperator for the master equation.
    
    The Liouvillian L is defined such that:
        d(vec(ρ))/dt = L · vec(ρ)
    
    where vec(ρ) is the vectorized density matrix (column-stacking).
    
    The Liouvillian includes:
    1. Coherent evolution: -i[H, ρ]
    2. Dissipation: Population transfer and coherence decay
    
    Args:
        Gamma: Transition rate matrix
        H: System Hamiltonian (NumPy array)
        
    Returns:
        ArrayLike: The Liouvillian superoperator (D² × D² complex matrix)
    """
    D = Gamma.shape[0]
    Liouville = np.zeros((D*D, D*D), dtype=np.complex128)
    
    # Coherent part: -i[H, ρ] = -i(H⊗I - I⊗H^T) vec(ρ)
    I = np.eye(D)
    Liouville += -1j * (np.kron(I, H) - np.kron(H.T, I))
    
    # Dissipative part
    Lambda = get_Lambda(Gamma)
    
    for m in range(D):
        for n in range(D):
            mn_idx = m + n*D  # Column-major vectorization
            
            if m == n:
                # Population dynamics
                for k in range(D):
                    if k != n:
                        # In-scattering: Γ_nk ρ_kk
                        Liouville[mn_idx, k + k*D] += Gamma[n, k]
                        # Out-scattering: -Γ_kn ρ_nn
                        Liouville[mn_idx, n + n*D] -= Gamma[k, n]
            else:
                # Coherence decay: -Λ_mn ρ_mn
                Liouville[mn_idx, mn_idx] -= Lambda[m, n]
        
    return Liouville


def red_H_idx(H, allowed_idx):
    """
    Reduce a Hamiltonian to a subspace specified by indices.
    
    Extracts the matrix elements H_ij for states i, j in the
    allowed subspace.
    
    Args:
        H: Full Hamiltonian (QuTiP Qobj or NumPy array)
        allowed_idx: List of indices for the allowed states
        
    Returns:
        ArrayLike: Reduced Hamiltonian matrix
    """
    N = len(allowed_idx)
    H_red = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            H_red[i, j] = H[allowed_idx[i], allowed_idx[j]]
    return H_red


def red_H(H, states):
    """
    Reduce a Hamiltonian to a subspace specified by state vectors.
    
    Calculates matrix elements ⟨ψ_i|H|ψ_j⟩ for all pairs of states
    in the given list.
    
    Args:
        H: Full Hamiltonian (QuTiP Qobj)
        states: List of state vectors (QuTiP Qobj kets)
        
    Returns:
        ArrayLike: Reduced Hamiltonian matrix
    """
    N = len(states)
    H_red = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            H_red[i, j] = (states[i].dag() * H * states[j])
    return H_red
