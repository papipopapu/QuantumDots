"""
Test suite for the hamiltonian module.

Tests the second quantization framework including:
- Space class for Hilbert spaces
- Fermionic creation/annihilation operators
- Hamiltonian matrix calculations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'lib'))

from sympy import Symbol, Matrix
import numpy as np

from hamiltonian import (
    Space, c_internal, calc_Hamiltonian, calc_vac,
    conjugate, delta_site, split_c_list
)


class TestSpace:
    """Tests for the Space class."""
    
    def test_simple_space_creation(self):
        """Test creating a simple spin space."""
        spin = Space('spin', ['up', 'down'])
        assert spin.dim() == 1
        assert len(spin.span) == 2
        
    def test_tensor_product(self):
        """Test tensor product of two spaces."""
        spin = Space('spin', ['up', 'down'])
        location = Space('location', ['left', 'right'])
        spin_loc = spin * location
        
        assert spin_loc.dim() == 2
        assert len(spin_loc.span) == 4  # 2 x 2 = 4 states
        
    def test_creation_operators(self):
        """Test generation of creation operators."""
        location = Space('location', ['left', 'right'])
        creations = location.creations()
        
        assert len(creations) == 2
        assert all(c.create == True for c in creations)
        
    def test_annihilation_operators(self):
        """Test generation of annihilation operators."""
        location = Space('location', ['left', 'right'])
        annihilations = location.annihilations()
        
        assert len(annihilations) == 2
        assert all(c.create == False for c in annihilations)


class TestOperators:
    """Tests for fermionic operator functions."""
    
    def test_c_internal_dagger(self):
        """Test Hermitian conjugate of operators."""
        c = c_internal(create=False, site={'location': 'left'})
        c_dag = c.d()
        
        assert c_dag.create == True
        assert c_dag.site == c.site
        
    def test_delta_site_same(self):
        """Test delta_site returns True for same site."""
        c1 = c_internal(create=True, site={'spin': 'up', 'location': 'left'})
        c2 = c_internal(create=False, site={'spin': 'up', 'location': 'left'})
        
        assert delta_site(c1, c2) == True
        
    def test_delta_site_different(self):
        """Test delta_site returns False for different sites."""
        c1 = c_internal(create=True, site={'spin': 'up', 'location': 'left'})
        c2 = c_internal(create=True, site={'spin': 'down', 'location': 'left'})
        
        assert delta_site(c1, c2) == False
        
    def test_split_c_list(self):
        """Test splitting operators into creation/annihilation lists."""
        c1 = c_internal(create=True, site={'location': 'left'})
        c2 = c_internal(create=False, site={'location': 'right'})
        c3 = c_internal(create=True, site={'location': 'right'})
        
        creation_list, annihilation_list = split_c_list([c1, c2, c3])
        
        assert len(creation_list) == 2
        assert len(annihilation_list) == 1
        
    def test_conjugate(self):
        """Test conjugate reverses order and applies dagger."""
        c1 = c_internal(create=True, site={'location': 'left'})
        c2 = c_internal(create=False, site={'location': 'right'})
        
        conj = conjugate([c1, c2])
        
        assert len(conj) == 2
        # Order is reversed and daggers applied
        assert conj[0].create == True  # c2 dagger (False -> True)
        assert conj[1].create == False  # c1 dagger (True -> False)


class TestHamiltonian:
    """Tests for Hamiltonian calculations."""
    
    def test_simple_tunneling_hamiltonian(self):
        """Test simple tunneling Hamiltonian between two sites."""
        spin = Space('spin', ['up', 'down'])
        location = Space('location', ['left', 'right'])
        spin_loc = location * spin
        
        cLu_, cLd_, cRu_, cRd_ = spin_loc.creations()
        cLu, cLd, cRu, cRd = spin_loc.annihilations()
        
        tau = Symbol('tau')
        H = [(-tau, [cLu_, cRu]), (-tau, [cRu_, cLu])]
        
        basis = [
            [(1, [cLu_])],  # |L, up>
            [(1, [cRu_])],  # |R, up>
        ]
        
        H_matrix = calc_Hamiltonian(H, basis)
        
        # Off-diagonal elements should be -tau
        assert H_matrix[0, 1] == -tau
        assert H_matrix[1, 0] == -tau
        # Diagonal elements should be zero (no on-site energy)
        assert H_matrix[0, 0] == 0
        assert H_matrix[1, 1] == 0
        
    def test_on_site_energy(self):
        """Test Hamiltonian with on-site energies."""
        location = Space('location', ['1', '2'])
        c1_, c2_ = location.creations()
        c1, c2 = location.annihilations()
        
        e1 = Symbol('e1')
        e2 = Symbol('e2')
        
        H = [
            (e1, [c1_, c1]),
            (e2, [c2_, c2]),
        ]
        
        basis = [
            [(1, [c1_])],
            [(1, [c2_])],
        ]
        
        H_matrix = calc_Hamiltonian(H, basis)
        
        assert H_matrix[0, 0] == e1
        assert H_matrix[1, 1] == e2
        assert H_matrix[0, 1] == 0
        assert H_matrix[1, 0] == 0
        
    def test_coulomb_interaction(self):
        """Test Hamiltonian with Coulomb interaction."""
        spin = Space('spin', ['up', 'down'])
        location = Space('location', ['L'])
        spin_loc = location * spin
        
        cLu_, cLd_ = spin_loc.creations()
        cLu, cLd = spin_loc.annihilations()
        
        U = Symbol('U')
        
        H = [(U, [cLu_, cLu, cLd_, cLd])]
        
        # Two electrons on the same site
        basis = [[(1, [cLu_, cLd_])]]
        
        H_matrix = calc_Hamiltonian(H, basis)
        
        assert H_matrix[0, 0] == U


class TestVacuumExpectation:
    """Tests for vacuum expectation value calculations."""
    
    def test_vev_creation_annihilation_pair(self):
        """Test VEV of c†c gives 1 for same site."""
        location = Space('location', ['1'])
        c1_, = location.creations()
        c1, = location.annihilations()
        
        # <0| c1 c1† |0> = 1 (normal ordering)
        result = calc_vac([c1, c1_])
        assert result == 1
        
    def test_vev_different_sites(self):
        """Test VEV of operators on different sites is zero."""
        location = Space('location', ['1', '2'])
        c1_, c2_ = location.creations()
        c1, c2 = location.annihilations()
        
        # <0| c1 c2† |0> = 0
        result = calc_vac([c1, c2_])
        assert result == 0
        
    def test_vev_unequal_creation_annihilation(self):
        """Test VEV is zero when numbers don't match."""
        location = Space('location', ['1'])
        c1_, = location.creations()
        
        # <0| c1† |0> = 0 (creating from vacuum, no pairing)
        result = calc_vac([c1_])
        assert result == 0
