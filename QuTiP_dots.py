import numpy as np
from qutip import *

def f_destroy(N, i):
    return tensor([sigmaz()] * i + [destroy(2)] + [identity(2)] * (N - i - 1))

def eqdot_state(occupations):
    N = len(occupations)
    return tensor([basis(2, occupations[i]) for i in range(N)])

# Basis
# |11, 00>
s11_00 = eqdot_state([1, 1, 0, 0])
# |00, 11>
s00_11 = eqdot_state([0, 0, 1, 1])
# |10, 10>
s10_10 = eqdot_state([1, 0, 1, 0])
# |01, 01>
s01_01 = eqdot_state([0, 1, 0, 1])
# |10, 01>
s10_01 = eqdot_state([1, 0, 0, 1])
# |01, 10>
s01_10 = eqdot_state([0, 1, 1, 0])

allowed_states = [s11_00, s00_11, s10_10, s01_01, s10_01, s01_10]
allowed_idx = [s.data.nonzero()[0][0] for s in allowed_states]

cLu = f_destroy(4, 0)
cLd = f_destroy(4, 1)
cRu = f_destroy(4, 2)
cRd = f_destroy(4, 3)

cLu_ = cLu.dag()
cLd_ = cLd.dag()
cRu_ = cRu.dag()
cRd_ = cRd.dag()

# Variables
eRu = 1
eRd = 2
eLu = 3
eLd = 4
tau = 0.1
UL = 1
UR = 1
V = 1

# Hamiltonian
H = eLu * cLu_ * cLu + eLd * cLd_ * cLd + eRu * cRu_ * cRu + eRd * cRd_ * cRd \
    - tau * (cLu_ * cRu + cRu_ * cLu + cLd_ * cRd + cRd_ * cLd) \
    + UL * cLu_ * cLu * cLd_ * cLd + UR * cRu_ * cRu * cRd_ * cRd \
    + V * (cLu_ * cLu * cRu_ * cRu + cLd_ * cLd * cRd_ * cRd \
    + cLu_ * cLu * cRd_ * cRd + cLd_ * cLd * cRu_ * cRu)
    
# Reduce Hamiltonian to allowed states
# Reduce Hamiltonian to allowed states
H_red = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        H_red[i, j] = H[allowed_idx[i], allowed_idx[j]] 
        
        
print(H_red)


