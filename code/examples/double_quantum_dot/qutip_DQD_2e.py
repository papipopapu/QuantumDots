"""
QuTiP-based Double Quantum Dot (DQD) simulation with 2 electrons.

This example uses QuTiP to numerically construct the Hamiltonian for a 
two-electron DQD system using Jordan-Wigner fermionic operators. The 
Hamiltonian is reduced to the 6-dimensional two-electron subspace.

Physical setup:
- 4 fermionic modes: Left (up, down), Right (up, down)
- 6 two-electron basis states
- Includes tunneling and Coulomb interactions

This demonstrates the numerical approach complementing the symbolic
calculations in DQD_2e.py

Author: Joel Martínez, 2023
Collaboration: Instituto de Ciencia de Materiales de Madrid (ICMM-CSIC)
"""

from qutipDots import *

# Define basis states for 2-electron subspace

s11_00 = eqdot_state([1, 1, 0, 0]) # |11, 00>
s00_11 = eqdot_state([0, 0, 1, 1]) # |00, 11>
s10_10 = eqdot_state([1, 0, 1, 0]) # |10, 10>
s01_01 = eqdot_state([0, 1, 0, 1]) # |01, 01>
s10_01 = eqdot_state([1, 0, 0, 1]) # |10, 01>
s01_10 = eqdot_state([0, 1, 1, 0]) # |01, 10>


allowed_states = [s11_00, s00_11, s10_10, s01_01, s10_01, s01_10]
allowed_idx = [get_state_index(s) for s in allowed_states]

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
H_red = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        H_red[i, j] = H[allowed_idx[i], allowed_idx[j]] 
        
        
print(H_red)

