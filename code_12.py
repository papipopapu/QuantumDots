from qutipDots import *
from numpy import exp
from qutip import tensor
import qutip as qt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
import numba as nb
from scipy.linalg import eig
import time
plt.style.use('science')


run = '10000tauZZ'


maxs = np.load('data/' + run + '.npy')

min = np.min(maxs[:, :, 3])
print(min)


@nb.njit
def evolve_U(U_dt, N, phi_0):
    rho_ii_t = np.zeros((N, 6), dtype=np.float64)
    for i in np.arange(N):
        phi_0 = np.dot(U_dt, phi_0)
        rho_ii_t[i, :] = np.abs(phi_0)**2
    return rho_ii_t
    
    
e1u = 0
e1d = 0
e2u = 0
e2d = 0
e3u = 0
e3d = 0
tau_0 = 1.0
tau_sf = 1.0

a_12 = 0
a_23 = a_12
a_31 = a_23

t23 = tau_0
deltas_tau = np.linspace(0, +9*t23, 100)
T0 = 2.094
tf = 10
tlist = np.linspace(0, tf*T0, 1000)
dt = tlist[1] - tlist[0]
N = len(tlist)

t12 = tau_0
t31 = tau_0
t23 = tau_0

t23_0 = tau_0
t12_0 = tau_0
t31_0 = tau_0



phi_0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.complex128)
phi_0 += np.random.rand(6)
phi_0 = phi_0/np.linalg.norm(phi_0)
print(phi_0)
# evolve superposed state

H = np.array([
    [e1u,                 0,                 -t12_0,                 -t12*exp(-1j*a_12),              -t31_0,                 t31*exp(-1j*a_31)],
    [0,                 e1d,                 t12*exp(1j*a_12),                 -t12_0,                 -t31*exp(1j*a_31),                 -t31_0],
    [-t12_0,    t12*exp(-1j*a_12),                 e2u,                 0,                 -t23_0,                 -t23*exp(-1j*a_23)],
    [-t12*exp(1j*a_12),                 -t12_0,                 0,                 e2d,                 t23*exp(1j*a_23),                 -t23_0],
    [-t31_0,                 -t31*exp(-1j*a_31),                 -t23_0,                 t23*exp(-1j*a_23),                 e3u,                 0],
    [t31*exp(1j*a_31),                 -t31_0,                 -t23*exp(1j*a_23),                 -t23_0,                 0,                 e3d]])

U_dt = expm(-1j*H*dt)

rho_ii_t = evolve_U(U_dt, N, phi_0)

rho_ii_t = np.abs(rho_ii_t)**2

# now subplot, 1x2, one for spin up, one for spin down
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.suptitle("Evolution of the diagonal elements of the density matrix", fontsize=30)

# Plot
axs[0].plot(tlist/T0, rho_ii_t[:, 0], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
axs[0].plot(tlist/T0, rho_ii_t[:, 2], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
axs[0].plot(tlist/T0, rho_ii_t[:, 4], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)
axs[1].plot(tlist/T0, rho_ii_t[:, 1], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
axs[1].plot(tlist/T0, rho_ii_t[:, 3], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
axs[1].plot(tlist/T0, rho_ii_t[:, 5], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)

# latex axis labels, move y label a bit to the left
axs[0].set_ylabel(r'$\rho_{\uparrow\uparrow}(t)$', fontsize=30, labelpad=25)
axs[1].set_ylabel(r'$\rho_{\downarrow\downarrow}(t)$', fontsize=30, labelpad=25)
axs[1].set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=30)

# only one legend for both subplots, located on top of both
axs[0].legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=3)

# change tick size
axs[0].tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
axs[1].tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)


# make lines thicker
axs[0].spines['bottom'].set_linewidth(1.5)
axs[0].spines['left'].set_linewidth(1.5)
axs[0].spines['top'].set_linewidth(1.5)
axs[0].spines['right'].set_linewidth(1.5)
axs[1].spines['bottom'].set_linewidth(1.5)
axs[1].spines['left'].set_linewidth(1.5)
axs[1].spines['top'].set_linewidth(1.5)
axs[1].spines['right'].set_linewidth(1.5)

# set subtitles, spin up and spin down, to the right margin of the subplots (outside of the plots)

axs[0].set_xlim(0, tf)
axs[1].set_xlim(0, tf)
axs[0].set_ylim(0, 1)
axs[1].set_ylim(0, 1)

# increase space between vertical subplots
# fig.subplots_adjust(hspace=0.2)

# add padding to x numbers
axs[0].tick_params(pad=10)
axs[1].tick_params(pad=10)



# now evolve each initial state separately and then add them up
rho_ii_t = np.zeros((N, 6), dtype=np.float64)
for i in range(6):
    phi_0p = np.zeros(6, dtype=np.complex128)
    phi_0p[i] = 1
    rho_ii_tp = evolve_U(U_dt, N, phi_0p)
    rho_ii_tp = np.abs(rho_ii_tp)**2
    
    rho_ii_t += rho_ii_tp * np.abs(phi_0[i])**2


# now subplot, 1x2, one for spin up, one for spin down
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.suptitle("superposition of initial states", fontsize=30)
# Plot
axs[0].plot(tlist/T0, rho_ii_t[:, 0], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
axs[0].plot(tlist/T0, rho_ii_t[:, 2], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
axs[0].plot(tlist/T0, rho_ii_t[:, 4], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)
axs[1].plot(tlist/T0, rho_ii_t[:, 1], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
axs[1].plot(tlist/T0, rho_ii_t[:, 3], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
axs[1].plot(tlist/T0, rho_ii_t[:, 5], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)

# latex axis labels, move y label a bit to the left
axs[0].set_ylabel(r'$\rho_{\uparrow\uparrow}(t)$', fontsize=30, labelpad=25)
axs[1].set_ylabel(r'$\rho_{\downarrow\downarrow}(t)$', fontsize=30, labelpad=25)
axs[1].set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=30)

# only one legend for both subplots, located on top of both
axs[0].legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=3)

# change tick size
axs[0].tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
axs[1].tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)


# make lines thicker
axs[0].spines['bottom'].set_linewidth(1.5)
axs[0].spines['left'].set_linewidth(1.5)
axs[0].spines['top'].set_linewidth(1.5)
axs[0].spines['right'].set_linewidth(1.5)
axs[1].spines['bottom'].set_linewidth(1.5)
axs[1].spines['left'].set_linewidth(1.5)
axs[1].spines['top'].set_linewidth(1.5)
axs[1].spines['right'].set_linewidth(1.5)

# set subtitles, spin up and spin down, to the right margin of the subplots (outside of the plots)

axs[0].set_xlim(0, tf)
axs[1].set_xlim(0, tf)
axs[0].set_ylim(0, 1)
axs[1].set_ylim(0, 1)

# increase space between vertical subplots
# fig.subplots_adjust(hspace=0.2)

# add padding to x numbers
axs[0].tick_params(pad=10)
axs[1].tick_params(pad=10)

plt.show()