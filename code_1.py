from hamiltonian import *
from sympy import Symbol, init_printing, latex, Matrix, simplify, exp, I, nsimplify
from sympy.physics.quantum import Dagger
import numpy as np
# expm
from scipy.linalg import expm
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
e1 = Symbol('epsilon_1', real=True, positive=True)
e2 = Symbol('epsilon_2', real=True, positive=True)
e3 = Symbol('epsilon_3', real=True, positive=True)
e = Symbol('epsilon', real=True, positive=True)
tau = Symbol('\\tau', real=True, positive=True)
tau_0 = Symbol('tau_0', real=True, positive=True)
tau_sf = Symbol('\\tau_{sf}' , real=True, positive=True)
tau_sf_conj = Symbol('\\tau_{sf}^*', real=True, positive=True)
alpha = Symbol('\\alpha', real=True, positive=True)
alpha_12 = Symbol('\\alpha_{12}', real=True, positive=True)
alpha_31 = Symbol('\\alpha_{13}', real=True, positive=True)
alpha_23 = Symbol('\\alpha_{23}', real=True, positive=True)




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
    (-tau_0, [c1u_, c3u]),
    (-tau_0, [c3u_, c1u]),
    (-tau_0, [c1d_, c3d]),
    (-tau_0, [c3d_, c1d]),
    # Tunneling spin flip
    (-tau_sf*exp(-I*alpha_12), [c1u_, c2d]),
    (-tau_sf*exp(I*alpha_12), [c2d_, c1u]),
    (tau_sf*exp(I*alpha_12), [c1d_, c2u]),
    (tau_sf*exp(-I*alpha_12), [c2u_, c1d]),
    (-tau_sf*exp(-I*alpha_23), [c2u_, c3d]),
    (-tau_sf*exp(I*alpha_23), [c3d_, c2u]),
    (tau_sf*exp(I*alpha_23), [c2d_, c3u]),
    (tau_sf*exp(-I*alpha_23), [c3u_, c2d]),
    
    (tau_sf*exp(-I*alpha_31), [c1u_, c3d]),
    (tau_sf*exp(I*alpha_31), [c3d_, c1u]),
    
    (-tau_sf*exp(I*alpha_31), [c1d_, c3u]),
    (-tau_sf*exp(-I*alpha_31), [c3u_, c1d])
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

sy = np.array([[0, -1j], [1j, 0]]) / 2 # sigma_y
sx = np.array([[0, 1], [1, 0]]) / 2 # sigma_x
sz = np.array([[1, 0], [0, -1]]) / 2 # sigma_z

Sx = np.kron(np.eye(3, 3), sx)
Sz = np.kron(np.eye(3, 3), sz)

# matrix made of s blocks 
Sy = np.kron(np.eye(3, 3), sy)

U = expm(-1j * np.pi * Sy)

Hu = U.T @ H_matrix @ U
print(latex(Hu))
print(latex(H_matrix))

