# DQD with 2e and spin in QuTiP
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# Variables
eRu = 1
eRd = 2
eLu = 3
eLd = 4
tau = 0.1
UL = 1
UR = 1
V = 1

# Build Hamiltonian with QuTiP
# Basis
# |11, 00>
s11_00 = tensor(basis(2, 1), basis(2, 1), basis(2, 0), basis(2, 0))
# |00, 11>
s00_11 = tensor(basis(2, 0), basis(2, 0), basis(2, 1), basis(2, 1))
# |10, 10>
s10_10 = tensor(basis(2, 1), basis(2, 0), basis(2, 1), basis(2, 0))
# |01, 01>
s01_01 = tensor(basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1))
# |10, 01>
s10_01 = tensor(basis(2, 1), basis(2, 0), basis(2, 0), basis(2, 1))
# |01, 10>
s01_10 = tensor(basis(2, 0), basis(2, 1), basis(2, 1), basis(2, 0))

allowed_states = [s11_00, s00_11, s10_10, s01_01, s10_01, s01_10]
allowed_idx = [s.data.nonzero()[0][0] for s in allowed_states]

# creation operators
cLu = tensor(sigmam(), qeye(2), qeye(2), qeye(2))
cLd = tensor(qeye(2), sigmam(), qeye(2), qeye(2))
cRu = tensor(qeye(2), qeye(2), sigmam(), qeye(2))
cRd = tensor(qeye(2), qeye(2), qeye(2), sigmam())

# annihilation operators
cLu_ = cLu.dag()
cLd_ = cLd.dag()
cRu_ = cRu.dag()
cRd_ = cRd.dag()

# Hamiltonian
H = eLu * cLu_ * cLu + eLd * cLd_ * cLd + eRu * cRu_ * cRu + eRd * cRd_ * cRd \
    - tau * (cLu_ * cRu + cRu_ * cLu + cLd_ * cRd + cRd_ * cLd) \
    + UL * cLu_ * cLu * cLd_ * cLd + UR * cRu_ * cRu * cRd_ * cRd \
    + V * (cLu_ * cLu * cRu_ * cRu + cLd_ * cLd * cRd_ * cRd \
    + cLu_ * cLu * cRd_ * cRd + cLd_ * cLd * cRu_ * cRu)
    
# Reduce Hamiltonian to allowed states
H_red = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        H_red[i, j] = H[allowed_idx[i], allowed_idx[j]] 


