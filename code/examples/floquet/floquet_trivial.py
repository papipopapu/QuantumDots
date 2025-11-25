from qutipDots import *
import matplotlib.pyplot as plt
from scipy.linalg import expm
# Basis

s11_00 = eqdot_state([1, 1, 0, 0]) # |11, 00>
s00_11 = eqdot_state([0, 0, 1, 1]) # |00, 11>
s10_10 = eqdot_state([1, 0, 1, 0]) # |10, 10>
s01_01 = eqdot_state([0, 1, 0, 1]) # |01, 01>
s10_01 = eqdot_state([1, 0, 0, 1]) # |10, 01>
s01_10 = eqdot_state([0, 1, 1, 0]) # |01, 10>


allowed_states = [s11_00, s00_11, s10_10, s01_01, s10_01, s01_10] # 2 electrons
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
T = 1
w = 2 * np.pi / T
VAC = 0
eRu = 1
eRd = 1
eLu = 1
eLd = 1
tau = 0.1
UL = 1
UR = 1


# Floquet amiltonian
H = eLu * cLu_ * cLu + eLd * cLd_ * cLd + eRu * cRu_ * cRu + eRd * cRd_ * cRd \
    - tau * (cLu_ * cRu + cRu_ * cLu + cLd_ * cRd + cRd_ * cLd) \
    + UL * cLu_ * cLu * cLd_ * cLd + UR * cRu_ * cRu * cRd_ * cRd \
        
# Periodic potential (to be multiplied by cos(wt))
HAC_0 = VAC * (cLu_ * cLu + cLd_ * cLd + cRu_ * cRu + cRd_ * cRd) / 2
        
H_red = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        H_red[i, j] = H[allowed_idx[i], allowed_idx[j]] 

# Diagonalize H_red to get floquet modes
mode_eps, modes_0_red = np.linalg.eig(H_red)

# Change to full space
modes_0 = np.zeros((16, 6), dtype=np.complex128)
for i, mode in enumerate(modes_0_red):
    for j, state in enumerate(allowed_states):
        modes_0[:, i] += mode[j] * state.data.toarray().flatten()



# Initial state
ts = np.linspace(0, 10 * T, 100)
psi_0 = s11_00

# Get <u_i(0)|psi(0)>
psi_0_floquet = modes_0_red[:, 0]


psi_t = np.zeros((6, len(ts)), dtype=np.complex128) 

for i, t in enumerate(ts):
    for mode in range(6):
        psi_t[:, i] += (psi_0_floquet[mode] * np.exp(-1j * mode_eps[mode] * t) * expm(-1j * HAC_0 * np.sin(w*t) / w).dot(modes_0[:, mode]))[allowed_idx]

abs_psi_t = np.abs(psi_t) ** 2

plt.plot(ts, abs_psi_t[0, :], label="state 1")
plt.plot(ts, abs_psi_t[1, :], label="state 2")
plt.plot(ts, abs_psi_t[2, :], label="state 3")
plt.plot(ts, abs_psi_t[3, :], label="state 4")
plt.plot(ts, abs_psi_t[4, :], label="state 5")
plt.plot(ts, abs_psi_t[5, :], label="state 6")

plt.legend()
plt.show()

# the evolution of the probabilities does not depend on VAC! (in hindsight, this is obvious)


