from qutipDots import *
from qutip import *
import matplotlib.pyplot as plt
from scipy.linalg import expm, null_space
# Basis
s0_0_0 = eqdot_state([0, 0, 0]) # |0, 0, 0>
s1_0_0 = eqdot_state([1, 0, 0]) # |1, 0, 0>
s0_1_0 = eqdot_state([0, 1, 0]) # |0, 1, 0>
s0_0_1 = eqdot_state([0, 0, 1]) # |0, 0, 1>

allowed_states = [s0_0_0, s1_0_0, s0_1_0, s0_0_1]
allowed_idx = [s.data.nonzero()[0][0] for s in allowed_states]

c1 = f_destroy(3, 0)
c2 = f_destroy(3, 1)
c3 = f_destroy(3, 2)

c1_ = c1.dag()
c2_ = c2.dag()
c3_ = c3.dag()

# Variables
e1 = 1
e2 = 1
e3 = 1
tau = 0.1

# Hamiltonian
H = e1 * c1_ * c1 + e2 * c2_ * c2 + e3 * c3_ * c3 \
    - tau * (c1_ * c2 + c2_ * c1 + c2_ * c3 + c3_ * c2)

# Reduce Hamiltonian to allowed states
H_red = np.zeros((4, 4), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        H_red[i, j] = H[allowed_idx[i], allowed_idx[j]]

# Create Gamma matrix
gL = tau / 10 
gR = tau / 10

Gamma = np.zeros((4, 4))
Gamma[1, 0] = gL
Gamma[0, 3] = gR

# Get Liouville matrix
Liouville = get_Liouville(Gamma, H_red)

# solve dynamics
rho0 = np.zeros((16, 1), dtype=np.complex128)
rho0[5] = 1
ts = np.linspace(0,1000, 1000)
rhos = np.array([expm(Liouville * t) @ rho0 for t in ts])



# plot diagonal elements
plt.plot(ts, rhos[:, 5].real, label='$\\rho_{11}$')
plt.plot(ts, rhos[:, 10].real, label='$\\rho_{22}$')
plt.plot(ts, rhos[:, 15].real, label='$\\rho_{33}$')

# plot trace
plt.plot(ts, np.array([np.trace(rho.real.reshape((4, 4))) for rho in rhos]), label='trace')



# find steady state
rho_ss = null_space(Liouville)
rho_ss = rho_ss / np.trace(rho_ss.real.reshape((4, 4)))

# plot diagonal elements
plt.plot(ts, np.array([rho_ss[5, 0] for t in ts]), '.', label='$\\rho_{11}$ ss')
plt.plot(ts, np.array([rho_ss[10, 0] for t in ts]), '-.',label='$\\rho_{22}$ ss')
plt.plot(ts, np.array([rho_ss[15, 0] for t in ts]), '--', label='$\\rho_{33}$ ss')

plt.legend()

plt.xlabel('Time')
plt.ylabel('Population')











plt.show()