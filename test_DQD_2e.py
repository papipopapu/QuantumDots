from Hamiltonian import *
from sympy import Symbol, init_printing, latex, Matrix

# Define spaces
spin = Space('spin', ['up', 'down'])
location = Space('location', ['left', 'right'])
spin_location = location * spin

# Get operators
cLu_, cLd_, cRu_, cRd_ = spin_location.creations()
cLu, cLd, cRu, cRd = spin_location.annihilations()

cL_, cR_ = location.creations()
cL, cR = location.annihilations()

# Define Hamiltonian
# Define factors (here symbolic variables are used)
eRu = Symbol('\epsilon_{R\\uparrow}')
eRd = Symbol('\epsilon_{R\\downarrow}')
eLu = Symbol('\epsilon_{L\\uparrow}')
eLd = Symbol('\epsilon_{L\\downarrow}')

tau = Symbol('tau')
UL = Symbol('U_L')
UR = Symbol('U_R')
V = Symbol('V')

H = [
    # Energies
    (eLu, [cLu_, cLu]),
    (eLd, [cLd_, cLd]),
    (eRu, [cRu_, cRu]),
    (eRd, [cRd_, cRd]),
    # Tunneling
    (-tau, [cLu_, cRu]),
    (-tau, [cRu_, cLu]),
    (-tau, [cLd_, cRd]),
    (-tau, [cRd_, cLd]),
    # Intradot Coulomb
    (UL, [cLu_, cLu, cLd_, cLd]),
    (UR, [cRu_, cRu, cRd_, cRd]),
    # Interdot Coulomb
    (V, [cL_, cL, cR_, cR])
]

# Define basis (creation operators acting on vacuum)
basis = [
    [cLu_, cLd_], # |11, 00>
    [cRu_, cRd_], # |00, 11>
    [cLu_, cRu_], # |10, 10>
    [cLd_, cRd_], # |01, 01>
    [cLu_, cRd_], # |10, 01>
    [cLd_, cRu_] # |01, 10>
]

# Calculate Hamiltonian matrix
H_matrix = calc_Hamiltonian(H, basis)

init_printing()
print(latex(Matrix(H_matrix)))

