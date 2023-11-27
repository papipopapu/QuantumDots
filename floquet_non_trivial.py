from qutipDots import *
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qutip import Qobj, floquet_modes
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
VAC = 1
eRu = 1
eRd = 1
eLu = 1
eLd = 1
tau = 0.1
UL = 1
UR = 1


# Hamiltonian non AC
H_0 = eLu * cLu_ * cLu + eLd * cLd_ * cLd + eRu * cRu_ * cRu + eRd * cRd_ * cRd \
    - tau * (cLu_ * cRu + cRu_ * cLu + cLd_ * cRd + cRd_ * cLd) \
    + UL * cLu_ * cLu * cLd_ * cLd + UR * cRu_ * cRu * cRd_ * cRd \
        
# Effect of AC amplitude (potential with phase difference)
omega = 1.0 * 2*np.pi
VACs = np.linspace(0, 10, 100) * omega
T = (2*np.pi)/omega
q_energies = np.zeros((len(VACs), 16))
args = {'w': omega}
HAC_p = (cLu_ * cLu + cLd_ * cLd - cRu_ * cRu - cRd_ * cRd) / 2 
for idx, VAC in enumerate(VACs): 
  HAC = VAC * HAC_p # phase difference necessary for non-trivial Floquet states
  H = [H_0, [HAC, lambda t, args: np.cos(args['w']*t)]] 
  f_modes, f_energies = floquet_modes(H, T, args, True) 
  q_energies[idx,:] = f_energies 
plt.figure() 
for i in range(16):
    plt.plot(VACs/omega, q_energies[:,i], label=f'Floquet quasienergy {i}')
    
plt.legend()
plt.xlabel(r'$VAC/\omega$') 
plt.ylabel(r'Quasienergy') 
plt.title(r'Floquet quasienergies') 


# Same with reduced Hamiltonian
def reduced_hamiltonian(H, allowed_idx):
    N = len(allowed_idx)
    H_red = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_red[i, j] = H[allowed_idx[i], allowed_idx[j]] 
    return Qobj(H_red)

q_energies = np.zeros((len(VACs), 6))
HAC_p = reduced_hamiltonian(HAC_p, allowed_idx)
H_0 = reduced_hamiltonian(H_0, allowed_idx)
for idx, VAC in enumerate(VACs): 
  HAC = VAC * HAC_p # phase difference necessary for non-trivial Floquet states
  H = [H_0, [HAC, lambda t, args: np.cos(args['w']*t)]] 
  f_modes, f_energies = floquet_modes(H, T, args, True) 
  q_energies[idx,:] = f_energies 
plt.figure() 
for i in range(6):
    plt.plot(VACs/omega, q_energies[:,i], label=f'Floquet quasienergy {i}')
    
plt.legend()
plt.xlabel(r'$VAC/\omega$') 
plt.ylabel(r'Quasienergy') 
plt.title(r'Floquet quasienergies RED') 

plt.show()


# Dynamics

# Periodic potential (to be multiplied by cos(wt))
""" HAC = VAC * (cLu_ * cLu + cLd_ * cLd - cRu_ * cRu - cRd_ * cRd) / 2

args = {'w': w}
H = [H_0, [HAC, 'cos(w * t)']]
 """