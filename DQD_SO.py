from hamiltonian import *
from sympy import Symbol, init_printing, latex, Matrix, simplify
import numpy as np
# Define spaces
spin = Space('spin', ['up', 'down'])
location = Space('location', ['left', 'right'])
spin_location = location * spin

# Get operators
cLu_, cLd_, cRu_, cRd_ = spin_location.creations()
cLu, cLd, cRu, cRd = spin_location.annihilations()

# Define Hamiltonian
# Define factors (here symbolic variables are used)
eR = Symbol('epsilon_R')
eL = Symbol('epsilon_L')

tau_0 = Symbol('tau')
tau_sf = Symbol('\\tau_{sf}')
U = Symbol('U')
V = Symbol('V')
Ez = Symbol('E_z') 

H = [
    # Energies
    (eL, [cLu_, cLu]),
    (eL, [cLd_, cLd]),
    (eR, [cRu_, cRu]),
    (eR, [cRd_, cRd]),
    # Tunneling no flip
    (-tau_0, [cLu_, cRu]),
    (-tau_0, [cRu_, cLu]),
    (-tau_0, [cLd_, cRd]),
    (-tau_0, [cRd_, cLd]),
    # Tunneling spin flip
    (-tau_sf, [cLu_, cRd]),
    (-tau_sf, [cRd_, cLu]),
    (-tau_sf, [cLd_, cRu]),
    (-tau_sf, [cRu_, cLd]),
    # Intradot Coulomb
    (U, [cLu_, cLu, cLd_, cLd]),
    (U, [cRu_, cRu, cRd_, cRd]),
    # Interdot Coulomb
    (V, [cLu_, cLu, cRu_, cRu]),
    (V, [cLd_, cLd, cRd_, cRd]),
    (V, [cLu_, cLu, cRd_, cRd]),
    (V, [cLd_, cLd, cRu_, cRu]),
    # Zeeman
    (Ez/2, [cLu_, cLu]),
    (-Ez/2, [cLd_, cLd]),
    (Ez/2, [cRu_, cRu]),
    (-Ez/2, [cRd_, cRd]),
]


# Define basis (creation operators acting on vacuum)
basis = [
    [(1, [cLu_, cLd_])], # |11, 00>
    [(1, [cRu_, cRd_])], # |00, 11>
    [(1, [cLu_, cRu_])], # |10, 10>
    [(1, [cLd_, cRd_])], # |01, 01>
    [(1, [cLu_, cRd_])], # |10, 01>
    [(1, [cLd_, cRu_])] # |01, 10>
]

# Calculate Hamiltonian matrix
H_matrix = calc_Hamiltonian(H, basis)

init_printing()


# Change basis to molecular basis
S11 = [0, 0, 0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]
S02 = [0, 1, 0, 0, 0, 0]
S20 = [1, 0, 0, 0, 0, 0]
T0 = [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]
Tm = [0, 0, 0, 1, 0, 0]
Tp = [0, 0, 1, 0, 0, 0]

U_basis_inv = np.array([S11, S02, S20, T0, Tm, Tp]).T
U_basis = np.linalg.inv(U_basis_inv)


H_matrix_U = U_basis @ H_matrix @ U_basis_inv


# lets try to do it with our basis function
U_basis = [
    [(1/np.sqrt(2), [cLu_, cRd_]), (-1/np.sqrt(2), [cLd_, cRu_])], # S11 1/sqrt(2)(|10, 01> - |01, 10>)
    [(1, [cRu_, cRd_])], # S02 |00, 11>
    [(1, [cLu_, cLd_])], # S20 |11, 00>
    [(1/np.sqrt(2), [cLu_, cRd_]), (1/np.sqrt(2), [cLd_, cRu_])], # T0 1/sqrt(2)(|10, 01> + |01, 10>)
    [(1, [cLd_, cRd_])], # T- |01, 01>
    [(1, [cLu_, cRu_])] # T+ |10, 10>
]

H_matrix_U_2 = calc_Hamiltonian(H, U_basis)

# you can check that both methods give the same result