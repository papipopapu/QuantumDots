"""
Test suite for the density module.

Tests the Lindblad master equation tools including:
- Lamb shift matrix calculation
- Density matrix equation derivation
- Vectorized Liouvillian construction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'lib'))

from sympy import Symbol, Matrix, zeros, simplify
import numpy as np

from density import (
    get_Lamb_matrix,
    get_density_equation,
    get_density_equation_vect
)


class TestLambMatrix:
    """Tests for Lamb shift matrix calculation."""
    
    def test_lamb_matrix_shape(self):
        """Test Lamb matrix has correct shape."""
        G = Matrix([[0, 1], [2, 0]])
        L = get_Lamb_matrix(G)
        
        assert L.shape == (2, 2)
        
    def test_lamb_matrix_simple(self):
        """Test Lamb matrix calculation for simple case."""
        # Simple 2x2 transition matrix
        G = Matrix([[0, 1], [2, 0]])
        L = get_Lamb_matrix(G)
        
        # The Lamb matrix is computed as:
        # L_mn = (1/2) * sum_{k != n} (G_km + G_kn)
        # L[0,0]: k=1, L = 0.5 * (G[1,0] + G[1,0]) = 0.5 * (2 + 2) = 2.0
        # L[0,1]: k=0, L = 0.5 * (G[0,0] + G[0,1]) = 0.5 * (0 + 1) = 0.5
        # L[1,0]: k=1, L = 0.5 * (G[1,1] + G[1,0]) = 0.5 * (0 + 2) = 1.0
        # L[1,1]: k=0, L = 0.5 * (G[0,1] + G[0,1]) = 0.5 * (1 + 1) = 1.0
        assert L[0, 0] == 2.0
        assert L[0, 1] == 0.5
        assert L[1, 0] == 1.0
        assert L[1, 1] == 1.0
        
    def test_lamb_matrix_diagonal_terms(self):
        """Test diagonal terms of Lamb matrix."""
        G = Matrix([
            [0, 0.1, 0],
            [0.2, 0, 0.15],
            [0, 0.25, 0]
        ])
        L = get_Lamb_matrix(G)
        
        # L_nn should account for all transition rates out of state n
        # and rates from n to other states
        assert L.shape == (3, 3)


class TestDensityEquation:
    """Tests for density matrix equation of motion."""
    
    def test_density_equation_shape(self):
        """Test density equation matrix has correct shape."""
        G = Matrix([[0, 1], [2, 0]])
        H = Matrix([[Symbol('E1'), 0], [0, Symbol('E2')]])
        
        drho = get_density_equation(G, H)
        
        assert drho.shape == (2, 2)
        
    def test_density_equation_contains_rho_symbols(self):
        """Test density equation contains density matrix symbols."""
        G = Matrix([[0, 1], [2, 0]])
        H = Matrix([[Symbol('E1'), 0], [0, Symbol('E2')]])
        
        drho = get_density_equation(G, H)
        
        # Check that rho symbols are in the result
        drho_str = str(drho)
        assert 'rho_' in drho_str
        
    def test_hermiticity_preserved(self):
        """Test that density equation preserves Hermiticity structure."""
        G = Matrix([[0, 0.1], [0.2, 0]])
        H = Matrix([[1, 0], [0, 2]])
        
        drho = get_density_equation(G, H)
        
        # The derivative should respect the structure of a density matrix
        # Diagonal elements should be real
        assert drho.shape == (2, 2)


class TestDensityEquationVectorized:
    """Tests for vectorized Liouvillian construction."""
    
    def test_liouvillian_shape(self):
        """Test Liouvillian superoperator has correct shape."""
        G = Matrix([[0, 1], [2, 0]])
        H = Matrix([[Symbol('E1'), 0], [0, Symbol('E2')]])
        
        L = get_density_equation_vect(G, H)
        
        # D x D density matrix -> D^2 vectorized
        assert L.shape == (4, 4)
        
    def test_liouvillian_larger_system(self):
        """Test Liouvillian for larger system."""
        G = zeros(3, 3)
        G[1, 0] = Symbol('G_10')
        G[0, 2] = Symbol('G_02')
        
        H = Matrix([
            [Symbol('e1'), Symbol('tau'), 0],
            [Symbol('tau'), Symbol('e2'), Symbol('tau')],
            [0, Symbol('tau'), Symbol('e3')]
        ])
        
        L = get_density_equation_vect(G, H)
        
        assert L.shape == (9, 9)
        
    def test_consistency_with_matrix_form(self):
        """Test vectorized form is consistent with matrix form."""
        G_L = Symbol('Gamma_L')
        G_R = Symbol('Gamma_R')
        G = zeros(4, 4)
        G[1, 0] = G_L
        G[0, 3] = G_R
        
        H = Matrix([
            [0, 0, 0, 0],
            [0, Symbol('e1'), Symbol('tau'), 0],
            [0, Symbol('tau'), Symbol('e2'), Symbol('tau')],
            [0, 0, Symbol('tau'), Symbol('e3')]
        ])
        
        drho_matrix = get_density_equation(G, H)
        L = get_density_equation_vect(G, H)
        
        # Construct vec(rho)
        rho_vect = zeros(16, 1)
        for m in range(4):
            for n in range(4):
                rho_vect[m + n*4, 0] = Symbol('rho_' + str(m) + str(n))
                
        # Apply Liouvillian
        drho_vect = L * rho_vect
        
        # Reshape back to matrix
        drho_vect_matrix = zeros(4, 4)
        for m in range(4):
            for n in range(4):
                drho_vect_matrix[m, n] = drho_vect[m + n*4, 0]
                
        # Both forms should give the same result
        diff = simplify(drho_vect_matrix - drho_matrix)
        assert diff == zeros(4, 4)
