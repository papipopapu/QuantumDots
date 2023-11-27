from qutipDots import *
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qutip import Qobj, floquet_modes, floquet_modes_table, floquet_modes_t_lookup, floquet_state_decomposition
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
T = 1.0
omega = 2.0 * np.pi / T
VAC = 10.0
eRu = 1.0
eRd = 1.0
eLu = 1.0
eLd = 1.0
tau = 0.1
UL = 1.0
UR = 1.0


# Hamiltonian non AC
H_0 = eLu * cLu_ * cLu + eLd * cLd_ * cLd + eRu * cRu_ * cRu + eRd * cRd_ * cRd \
    - tau * (cLu_ * cRu + cRu_ * cLu + cLd_ * cRd + cRd_ * cLd) \
    + UL * cLu_ * cLu * cLd_ * cLd + UR * cRu_ * cRu * cRd_ * cRd 
        
# Periodic potential (to be multiplied by cos(wt))
HAC = VAC * (cLu_ * cLu + cLd_ * cLd - cRu_ * cRu - cRd_ * cRd) / 2

# Reduce hamiltonian to 2 electron space
H_0 = Qobj(red_H(H_0, allowed_states))
HAC = Qobj(red_H(HAC, allowed_states))

args = {'w': omega}

H = [H_0, [HAC, lambda t, args: np.cos(args['w']*t)]] 

# Initial state |psi(0)>
psi_0 = s11_00

# Get |u_i(0)>
f_modes_0, f_energies = floquet_modes(H, T, args)

# Get <u_i(0)|psi(0)>
# Its just gonna be the first component of every mode
f_coeff = [f_mode_0.full()[0][0] for f_mode_0 in f_modes_0]

# Get look up table of modes in T (i.e., diagonalize U(T+t, t) for t in [0, T]
# , but also allow to look up for t > T from periodicity)
ts = np.linspace(0, 10*T, 1000)

f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, ts, H, T, args)


p_ex = np.zeros((6, len(ts)))
state_ex = s11_00
for n, t in enumerate(ts):
    f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
    psi_t = 0
    for i, f_mode_t in enumerate(f_modes_t):
        psi_t += np.exp(-1j * f_energies[i] * t) * f_coeff[i] * f_mode_t
    p_ex[:, n] = np.abs(psi_t.full(squeeze=True))**2
    
plt.figure()
plt.title('Reduced Hamiltonian')
for i in range(6):
    plt.plot(ts, p_ex[i, :], label=f'|{i}>')
plt.legend()




# Now without reducing the Hamiltonian
# Hamiltonian non AC
H_0 = eLu * cLu_ * cLu + eLd * cLd_ * cLd + eRu * cRu_ * cRu + eRd * cRd_ * cRd \
    - tau * (cLu_ * cRu + cRu_ * cLu + cLd_ * cRd + cRd_ * cLd) \
    + UL * cLu_ * cLu * cLd_ * cLd + UR * cRu_ * cRu * cRd_ * cRd 
        
# Periodic potential (to be multiplied by cos(wt))
HAC = VAC * (cLu_ * cLu + cLd_ * cLd - cRu_ * cRu - cRd_ * cRd) / 2

args = {'w': omega}

H = [H_0, [HAC, lambda t, args: np.cos(args['w']*t)]] 

# Initial state |psi(0)>
psi_0 = s11_00

# Get |u_i(0)>
f_modes_0, f_energies = floquet_modes(H, T, args)

# Get <u_i(0)|psi(0)>
f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi_0)


# Get look up table of modes in T (i.e., diagonalize U(T+t, t) for t in [0, T]
# , but also allow to look up for t > T from periodicity)
ts = np.linspace(0, 10*T, 1000)

f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, ts, H, T, args)


p_ex = np.zeros((16, len(ts)))
state_ex = s11_00
for n, t in enumerate(ts):
    f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
    psi_t = 0
    for i, f_mode_t in enumerate(f_modes_t):
        psi_t += np.exp(-1j * f_energies[i] * t) * f_coeff[i] * f_mode_t
    p_ex[:, n] = np.abs(psi_t.full(squeeze=True))**2
    
plt.figure()
plt.title('Full Hamiltonian')
for i, idx in enumerate(allowed_idx):
    plt.plot(ts, p_ex[idx, :], label=f'|{i}>')
plt.legend()
plt.show()