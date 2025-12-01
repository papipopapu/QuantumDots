"""
Test suite for the qutipDots module.

Tests QuTiP-based numerical simulation tools including:
- Fermionic operators with Jordan-Wigner transformation
- State construction
- Hamiltonian reduction
- Liouvillian construction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'lib'))

import numpy as np
from numpy.testing import assert_allclose

from qutipDots import (
    f_destroy, f_create, eqdot_state, get_Lambda,
    get_Liouville, red_H_idx, red_H, get_state_index
)


class TestFermionicOperators:
    """Tests for fermionic operator construction."""
    
    def test_f_destroy_shape(self):
        """Test fermionic annihilation operator has correct shape."""
        c = f_destroy(3, 0)
        assert c.shape == (8, 8)  # 2^3 = 8
        
    def test_f_create_shape(self):
        """Test fermionic creation operator has correct shape."""
        c_dag = f_create(3, 0)
        assert c_dag.shape == (8, 8)
        
    def test_anticommutation_same_site(self):
        """Test {c_i, c_i†} = 1 (anticommutation relation)."""
        for N in [2, 3, 4]:
            for i in range(N):
                c = f_destroy(N, i)
                c_dag = f_create(N, i)
                
                anticomm = c * c_dag + c_dag * c
                identity = np.eye(2**N)
                
                assert_allclose(np.array(anticomm.full()), identity, atol=1e-10)
                
    def test_anticommutation_different_sites(self):
        """Test {c_i, c_j†} = 0 for i ≠ j."""
        N = 3
        c1 = f_destroy(N, 0)
        c2_dag = f_create(N, 1)
        
        anticomm = c1 * c2_dag + c2_dag * c1
        
        assert_allclose(np.array(anticomm.full()), np.zeros((2**N, 2**N)), atol=1e-10)
        
    def test_annihilation_squared_is_zero(self):
        """Test c_i * c_i = 0 (fermionic property)."""
        c = f_destroy(3, 0)
        c_sq = c * c
        
        assert_allclose(np.array(c_sq.full()), np.zeros((8, 8)), atol=1e-10)


class TestEqdotState:
    """Tests for Fock state construction."""
    
    def test_vacuum_state(self):
        """Test vacuum state |0,0,0⟩."""
        state = eqdot_state([0, 0, 0])
        
        assert state.shape == (8, 1)
        # First component should be 1
        assert_allclose(np.abs(state.full()[0][0]), 1.0)
        
    def test_single_occupation(self):
        """Test single occupation state."""
        state = eqdot_state([1, 0, 0])
        
        # Should have exactly one nonzero element
        full = np.array(state.full()).flatten()
        assert np.sum(np.abs(full) > 0.5) == 1
        
    def test_multiple_occupations(self):
        """Test state with multiple occupations."""
        state = eqdot_state([1, 1, 0])
        
        full = np.array(state.full()).flatten()
        assert np.sum(np.abs(full) > 0.5) == 1


class TestGetStateIndex:
    """Tests for the get_state_index helper function."""
    
    def test_vacuum_index(self):
        """Test index of vacuum state."""
        state = eqdot_state([0, 0, 0])
        idx = get_state_index(state)
        
        assert idx == 0
        
    def test_single_occupation_indices(self):
        """Test indices for single occupation states."""
        # Different occupation patterns should give different indices
        s000 = eqdot_state([0, 0, 0])
        s100 = eqdot_state([1, 0, 0])
        s010 = eqdot_state([0, 1, 0])
        s001 = eqdot_state([0, 0, 1])
        
        indices = [get_state_index(s) for s in [s000, s100, s010, s001]]
        
        # All indices should be unique
        assert len(set(indices)) == 4


class TestLambda:
    """Tests for Lamb shift matrix calculation."""
    
    def test_lambda_shape(self):
        """Test Lamb matrix has correct shape."""
        Gamma = np.array([[0, 0.1], [0.2, 0]])
        Lambda = get_Lambda(Gamma)
        
        assert Lambda.shape == (2, 2)
        
    def test_lambda_zeros_diagonal(self):
        """Test Lamb matrix with only off-diagonal Gamma."""
        Gamma = np.zeros((3, 3))
        Gamma[1, 0] = 0.1
        Gamma[0, 2] = 0.2
        
        Lambda = get_Lambda(Gamma)
        
        assert Lambda.shape == (3, 3)


class TestLiouvillian:
    """Tests for Liouvillian superoperator construction."""
    
    def test_liouvillian_shape(self):
        """Test Liouvillian has correct shape."""
        Gamma = np.array([[0, 0.1], [0.2, 0]])
        H = np.array([[1, 0], [0, 2]], dtype=np.complex128)
        
        L = get_Liouville(Gamma, H)
        
        assert L.shape == (4, 4)
        
    def test_liouvillian_trace_preservation(self):
        """Test that trace is preserved by Liouvillian dynamics."""
        Gamma = np.array([[0, 0.1], [0.2, 0]])
        H = np.array([[1, 0.1], [0.1, 2]], dtype=np.complex128)
        
        L = get_Liouville(Gamma, H)
        
        # The trace should be preserved: tr(L @ rho) = 0 for trace 1 rho
        # This means the sum of diagonal elements of L should be related correctly
        # More specifically: sum over j of L[j+j*D, :] should give zero rates
        
    def test_liouvillian_hermitian_hamiltonian(self):
        """Test with Hermitian Hamiltonian."""
        Gamma = np.zeros((3, 3))
        Gamma[1, 0] = 0.1
        Gamma[0, 2] = 0.2
        
        H = np.array([
            [0, 0.1, 0],
            [0.1, 1, 0.1],
            [0, 0.1, 2]
        ], dtype=np.complex128)
        
        L = get_Liouville(Gamma, H)
        
        assert L.shape == (9, 9)
        assert L.dtype == np.complex128


class TestReducedHamiltonian:
    """Tests for Hamiltonian reduction functions."""
    
    def test_red_H_idx(self):
        """Test reducing Hamiltonian by indices."""
        H = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ])
        
        H_red = red_H_idx(H, [0, 2])
        
        expected = np.array([[0, 2], [8, 10]])
        assert_allclose(H_red, expected)
        
    def test_red_H_with_states(self):
        """Test reducing Hamiltonian using state vectors."""
        # Create a simple number operator Hamiltonian
        c1 = f_destroy(3, 0)
        c1_dag = f_create(3, 0)
        
        H = c1_dag * c1  # Number operator for site 1
        
        states = [eqdot_state([0, 0, 0]), eqdot_state([1, 0, 0])]
        H_red = red_H(H, states)
        
        # <000|n_1|000> = 0, <100|n_1|100> = 1
        assert_allclose(H_red[0, 0], 0, atol=1e-10)
        assert_allclose(H_red[1, 1], 1, atol=1e-10)
        assert_allclose(H_red[0, 1], 0, atol=1e-10)
        assert_allclose(H_red[1, 0], 0, atol=1e-10)


class TestPhysicalConsistency:
    """Integration tests for physical consistency."""
    
    def test_tunneling_hamiltonian(self):
        """Test that tunneling Hamiltonian has correct structure."""
        N = 3
        tau = 0.1
        
        c1 = f_destroy(N, 0)
        c2 = f_destroy(N, 1)
        c3 = f_destroy(N, 2)
        c1_dag, c2_dag, c3_dag = c1.dag(), c2.dag(), c3.dag()
        
        H = -tau * (c1_dag * c2 + c2_dag * c1 + c2_dag * c3 + c3_dag * c2)
        
        # Hamiltonian should be Hermitian
        H_full = np.array(H.full())
        assert_allclose(H_full, H_full.conj().T, atol=1e-10)
        
    def test_lindblad_steady_state(self):
        """Test that Liouvillian admits a steady state."""
        from scipy.linalg import null_space
        
        Gamma = np.zeros((4, 4))
        Gamma[1, 0] = 0.01  # Injection
        Gamma[0, 3] = 0.01  # Extraction
        
        H = np.array([
            [0, 0, 0, 0],
            [0, 1, -0.1, 0],
            [0, -0.1, 1, -0.1],
            [0, 0, -0.1, 1]
        ], dtype=np.complex128)
        
        L = get_Liouville(Gamma, H)
        
        # Find null space (steady state)
        rho_ss = null_space(L)
        
        # Should have exactly one steady state
        assert rho_ss.shape[1] == 1
        
        # Steady state should have positive trace
        rho_ss_matrix = rho_ss.reshape((4, 4))
        trace = np.trace(rho_ss_matrix)
        assert np.abs(trace) > 1e-10
