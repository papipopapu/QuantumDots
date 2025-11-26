"""
Triple Quantum Dot (TQD) with Spin-Orbit Coupling.

This example constructs the Hamiltonian for a linear TQD system with 
spin-orbit coupling. The spin-flip tunneling enables spin manipulation
without magnetic fields, which is important for spintronic applications.

Physical setup:
- 3 quantum dots with spin: (1↑, 1↓, 2↑, 2↓, 3↑, 3↓)
- Normal tunneling: τ_0 (spin-preserving)
- Spin-flip tunneling: τ_sf (spin-orbit induced)
- Coulomb interactions: U (intradot), V (interdot)
- Zeeman splitting: E_z

Two cases are studied:
1. Single electron (6 basis states)
2. Two electrons (15 basis states)

Author: Joel Martínez, 2023
Collaboration: Instituto de Ciencia de Materiales de Madrid (ICMM-CSIC)
"""

from hamiltonian import *
from sympy import Symbol, init_printing, latex, Matrix, simplify
import numpy as np

init_printing()

# Define spaces
spin = Space('spin', ['up', 'down'])
location = Space('location', ['1', '2', '3'])
spin_location = location * spin

# Get operators
c1u_, c1d_, c2u_, c2d_, c3u_, c3d_ = spin_location.creations()
c1u, c1d, c2u, c2d, c3u, c3d = spin_location.annihilations()

# Define Hamiltonian
# Define factors (here symbolic variables are used)
e1 = Symbol('epsilon_1')
e2 = Symbol('epsilon_2')
e3 = Symbol('epsilon_3')
tau_0 = Symbol('tau_0')
tau_sf = Symbol('\\tau_{sf}')
U = Symbol('U')
V = Symbol('V')
Ez = Symbol('E_z')

H = [
    # Energies
    (e1, [c1u_, c1u]),
    (e1, [c1d_, c1d]),
    (e2, [c2u_, c2u]),
    (e2, [c2d_, c2d]),
    (e3, [c3u_, c3u]),
    (e3, [c3d_, c3d]),
    # Tunneling no flip
    (-tau_0, [c1u_, c2u]),
    (-tau_0, [c2u_, c1u]),
    (-tau_0, [c2u_, c3u]),
    (-tau_0, [c3u_, c2u]),
    (-tau_0, [c1d_, c2d]),
    (-tau_0, [c2d_, c1d]),
    (-tau_0, [c2d_, c3d]),
    (-tau_0, [c3d_, c2d]),
    # Tunneling spin flip
    (-tau_sf, [c1u_, c2d]),
    (-tau_sf, [c2d_, c1u]),
    (-tau_sf, [c2u_, c1d]),
    (-tau_sf, [c1d_, c2u]),
    (-tau_sf, [c2u_, c3d]),
    (-tau_sf, [c3d_, c2u]),
    (-tau_sf, [c2d_, c3u]),
    (-tau_sf, [c3u_, c2d]),
    # Intradot Coulomb
    (U, [c1u_, c1u, c1d_, c1d]),
    (U, [c2u_, c2u, c2d_, c2d]),
    (U, [c3u_, c3u, c3d_, c3d]),
    # Interdot Coulomb
    (V, [c1u_, c1u, c2u_, c2u]),
    (V, [c1d_, c1d, c2d_, c2d]),
    (V, [c1u_, c1u, c3u_, c3u]),
    (V, [c1d_, c1d, c3d_, c3d]),
    (V, [c2u_, c2u, c3u_, c3u]),
    (V, [c2d_, c2d, c3d_, c3d]),
    # Zeeman
    (Ez/2, [c1u_, c1u]),
    (-Ez/2, [c1d_, c1d]),
    (Ez/2, [c2u_, c2u]),
    (-Ez/2, [c2d_, c2d]),
    (Ez/2, [c3u_, c3u]),
    (-Ez/2, [c3d_, c3d]),
]
# only allow 1 electron for now
basis = [ 
        [(1, [c1u_] )], # |10, 00, 00>
        [(1, [c1d_] )], # |01, 00, 00>
        [(1, [c2u_] )], # |00, 10, 00>
        [(1, [c2d_] )], # |00, 01, 00>
        [(1, [c3u_] )], # |00, 00, 10>
        [(1, [c3d_] )] # |00, 00, 01>
]
    
    
# Calculate Hamiltonian matrix
H_matrix = Matrix(calc_Hamiltonian(H, basis))
print("Hamiltonian matrix:")
print(latex(H_matrix))

# two electrons
basis = [
        [(1, [c1u_, c1d_] )], # |11, 00, 00>
        [(1, [c2u_, c2d_] )], # |00, 11, 00>
        [(1, [c3u_, c3d_] )], # |00, 00, 11>
        [(1, [c1u_, c2u_] )], # |10, 10, 00>
        [(1, [c1d_, c2d_] )], # |01, 01, 00>
        [(1, [c1u_, c3u_] )], # |10, 00, 10>
        [(1, [c1d_, c3d_] )], # |01, 00, 01>
        [(1, [c2u_, c3u_] )], # |00, 10, 10>
        [(1, [c2d_, c3d_] )], # |00, 01, 01>
        [(1, [c1u_, c2d_] )], # |10, 01, 00>
        [(1, [c1d_, c2u_] )], # |01, 10, 00>
        [(1, [c1u_, c3d_] )], # |10, 00, 01>
        [(1, [c1d_, c3u_] )], # |01, 00, 10>
        [(1, [c2u_, c3d_] )], # |00, 10, 01>
        [(1, [c2d_, c3u_] )], # |00, 01, 10>
]

H_matrix = Matrix(calc_Hamiltonian(H, basis))
""" print("Hamiltonian matrix:")
print(latex(H_matrix)) """