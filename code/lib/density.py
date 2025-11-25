"""
Density matrix module for quantum dot systems.

This module provides tools for describing open quantum systems
using the Lindblad master equation formalism. It implements the
time evolution of density matrices including dissipative effects
from coupling to reservoirs.

Key functionality:
- Lamb shift matrix calculation
- Density matrix equation of motion
- Vectorized Liouvillian superoperator construction

Physical background:
The Lindblad master equation describes the evolution of a density matrix ρ:
    dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})

where H is the Hamiltonian and L_k are Lindblad jump operators
describing coupling to the environment (e.g., electron tunneling
to/from reservoirs).
"""

from sympy import Symbol, Matrix, zeros, eye
from sympy.physics.quantum import TensorProduct as kron
import typing


# NOTE: All calculations use NATURAL UNITS (ℏ = 1)


def get_Lamb_matrix(G_matrix: Matrix) -> Matrix:
    """
    Calculate the Lamb shift matrix from the transition rate matrix.
    
    The Lamb shift describes energy renormalization due to coupling
    to the environment. For quantum dots, this represents the effect
    of virtual transitions to reservoir states.
    
    The Lamb shift contribution to coherence decay rates is:
        Λ_mn = (1/2) Σ_{k≠n} (Γ_km + Γ_kn)
    
    Args:
        G_matrix: Square matrix of transition rates Γ_ij
                  (rate of transition from state j to state i)
    
    Returns:
        Matrix: The Lamb shift matrix Λ_mn
    """
    # Verify input is square matrix
    assert G_matrix.shape[0] == G_matrix.shape[1]
    D = G_matrix.shape[0]
    L_matrix = zeros(D, D)
    
    for m in range(D):
        for n in range(D):
            L_mn = 0
            for k in range(D):
                if k != n:
                    # Sum contributions from all other states
                    L_mn += 1/2 * (G_matrix[k, m] + G_matrix[k, n])
            L_matrix[m, n] = L_mn
            
    return L_matrix


def get_density_equation(G_matrix: Matrix, H_matrix: Matrix) -> Matrix:
    """
    Generate the equation of motion for the density matrix.
    
    Constructs the right-hand side of the Lindblad master equation:
        dρ/dt = -i[H, ρ] + (incoherent terms)
    
    The incoherent terms include:
    - Population transfer between diagonal elements (Γ_nm rates)
    - Decoherence of off-diagonal elements (Lamb shift)
    
    Args:
        G_matrix: Transition rate matrix Γ_ij (same dimension as H_matrix)
        H_matrix: System Hamiltonian matrix
        
    Returns:
        Matrix: The derivative dρ/dt expressed in terms of ρ elements
                (symbolic matrix with ρ_mn as symbols)
    """
    # Validate input dimensions
    assert G_matrix.shape[0] == G_matrix.shape[1]
    assert H_matrix.shape[0] == H_matrix.shape[1]
    assert G_matrix.shape[0] == H_matrix.shape[0]
    D = G_matrix.shape[0]

    # Initialize result matrix for dρ/dt
    drho_matrix = zeros(D, D)
    
    # Create symbolic density matrix
    rho = zeros(D, D)
    for m in range(D):
        for n in range(D):
            rho[m, n] = Symbol('rho_' + str(m) + str(n))
            
    # Add coherent evolution: -i[H, ρ]
    drho_matrix += -1j * (H_matrix * rho - rho * H_matrix)
    
    # Add dissipative terms
    Lamb_matrix = get_Lamb_matrix(G_matrix)
    for m in range(D):
        for n in range(D):
            if m == n:
                # Diagonal elements: population dynamics
                # dρ_nn/dt = Σ_k (Γ_nk ρ_kk - Γ_kn ρ_nn)
                for k in range(D):
                    if k != n:
                        drho_matrix[m, n] += G_matrix[n, k] * rho[k, k] - G_matrix[k, n] * rho[n, n]
            else:
                # Off-diagonal elements: coherence decay
                # dρ_mn/dt -= Λ_mn ρ_mn
                drho_matrix[m, n] -= Lamb_matrix[m, n] * rho[m, n]

    return drho_matrix


def get_density_equation_vect(G_matrix: Matrix, H_matrix: Matrix) -> Matrix:
    """
    Generate the Liouvillian superoperator in vectorized form.
    
    The density matrix equation dρ/dt = L[ρ] can be rewritten in
    vectorized form as:
        d(vec(ρ))/dt = L_vec · vec(ρ)
    
    where vec(ρ) stacks columns of ρ into a vector and L_vec is the
    Liouvillian superoperator matrix.
    
    This is useful for:
    - Finding steady states (null space of L_vec)
    - Numerical time evolution
    - Spectral analysis of the dynamics
    
    The vectorization uses the identity:
        vec(AXB) = (B^T ⊗ A) vec(X)
    
    Args:
        G_matrix: Transition rate matrix
        H_matrix: System Hamiltonian matrix
        
    Returns:
        Matrix: The Liouvillian superoperator L_vec (D² × D² matrix)
    """
    # Validate input dimensions
    assert G_matrix.shape[0] == G_matrix.shape[1]
    assert H_matrix.shape[0] == H_matrix.shape[1]
    assert G_matrix.shape[0] == H_matrix.shape[0]
    D = G_matrix.shape[0]

    # Initialize Liouvillian superoperator
    L = zeros(D*D, D*D)
    
    # Add coherent part using vectorization identity
    # -i[H, ρ] -> -i(I ⊗ H - H^T ⊗ I) vec(ρ)
    I = eye(D)
    L += -1j * (kron(I, H_matrix) - kron(H_matrix.T, I))
    
    # Add dissipative part
    Lamb_matrix = get_Lamb_matrix(G_matrix)
    
    for m in range(D):
        for n in range(D):
            mn_idx = m + n*D  # Index of ρ_mn in vectorized ρ (column-major)
            
            if m == n:
                # Population dynamics: ρ_nn terms
                for k in range(D):
                    if k != n:
                        # In: Γ_nk ρ_kk
                        A = G_matrix[n, k]
                        ij_idx = k + k*D
                        L[mn_idx, ij_idx] += A
                        
                        # Out: -Γ_kn ρ_nn
                        A = G_matrix[k, n]
                        ij_idx = n + n*D
                        L[mn_idx, ij_idx] -= A
            else:
                # Coherence decay: -Λ_mn ρ_mn
                A = Lamb_matrix[m, n]
                ij_idx = mn_idx
                L[mn_idx, ij_idx] -= A
        
    return L
