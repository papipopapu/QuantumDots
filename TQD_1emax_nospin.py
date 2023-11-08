from hamiltonian import *
from density import *

from sympy import Symbol, init_printing, latex, Matrix, zeros, simplify


# enable printing of latex
init_printing()

# Define spaces
location = Space('location', ['1','2','3'])

# Get operators
c1_, c2_, c3_ = location.creations()
c1, c2, c3 = location.annihilations()

# Define Hamiltonian

# Define factors (here symbolic variables are used)
e1 = Symbol('\epsilon_1')
e2 = Symbol('\epsilon_2')
e3 = Symbol('\epsilon_3')
tau = Symbol('tau')


H = [
    
    # (factor, [operators])
    
    # Energies
    (e1, [c1_, c1]),
    (e2, [c2_, c2]),
    (e3, [c3_, c3]),
    
    # Tunneling
    (-tau, [c1_, c2]),
    (-tau, [c2_, c1]),
    (-tau, [c2_, c3]),
    (-tau, [c3_, c2])
]

# Define basis (creation operators acting on vacuum)

basis = [
    [(1, [])], # |000>
    [(1, [c1_])], # |100>
    [(1, [c2_])], # |010>
    [(1, [c3_])], # |001>
]

# Calculate Hamiltonian matrix
H_matrix = Matrix(calc_Hamiltonian(H, basis))
print("Hamiltonian matrix:")
print(latex(H_matrix))

# create Gamma matrix
G_L = Symbol('Gamma_L')
G_R = Symbol('Gamma_R')
G_matrix = zeros(4, 4)

G_matrix[1, 0] = G_L
G_matrix[0, 3] = G_R

# we can get directly the matrix equation d/dt rho = L rho
drho_matrix = get_density_equation(G_matrix, H_matrix)
print("drho/dt matrix:")
print(latex(drho_matrix))

# or the vectorised form of L 
L = get_density_equation_vect(G_matrix, H_matrix)

# we
# can check that drho_matrix = matrix(L * vect(rho))
rho_vect = zeros(16, 1)
for m in range(4):
    for n in range(4):
        rho_vect[m + n*4, 0] = Symbol('rho_' + str(m) + str(n))
        
drho_vect = L * rho_vect
drho_vect_matrix = zeros(4, 4)
for m in range(4):
    for n in range(4):
        drho_vect_matrix[m, n] = drho_vect[m + n*4, 0]
        
assert(simplify(drho_vect_matrix - drho_matrix) == zeros(4, 4))



# tau microeV
# epsilon microeV
# U meV
# V ~microev
# t ~ 10 ns
# Gamma ~ tau / 10


