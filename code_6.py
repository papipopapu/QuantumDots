# Triangular TQD with 1 electron and spin flip tunneling
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
# Basis
s10_00_00 = eqdot_state([1, 0, 0, 0, 0, 0]) # |10, 00, 00>
s01_00_00 = eqdot_state([0, 1, 0, 0, 0, 0]) # |01, 00, 00>
s00_10_00 = eqdot_state([0, 0, 1, 0, 0, 0]) # |00, 10, 00>
s00_01_00 = eqdot_state([0, 0, 0, 1, 0, 0]) # |00, 01, 00>
s00_00_10 = eqdot_state([0, 0, 0, 0, 1, 0]) # |00, 00, 10>
s00_00_01 = eqdot_state([0, 0, 0, 0, 0, 1]) # |00, 00, 01>

allowed_states = [s10_00_00, s01_00_00, s00_10_00, s00_01_00, s00_00_10, s00_00_01]

# Relevant operators
c1u = f_destroy(6, 0)
c1d = f_destroy(6, 1)
c2u = f_destroy(6, 2)
c2d = f_destroy(6, 3)
c3u = f_destroy(6, 4)
c3d = f_destroy(6, 5)

c1u_ = c1u.dag()
c1d_ = c1d.dag()
c2u_ = c2u.dag()
c2d_ = c2d.dag()
c3u_ = c3u.dag()
c3d_ = c3d.dag() 

# Variables
e1u = 0
e1d = 0
e2u = 0
e2d = 0
e3u = 0
e3d = 0
tau_0 = 1.0
tau_sf = 1.0

a_12 = 0

t12 = tau_sf
t23 = tau_sf
t31 = tau_sf


T_0 = 2.094
#tlist = np.linspace(0, 30000*T_0, 15000)
tlist = np.linspace(0, 10*T_0, 150)
dt = tlist[1] - tlist[0]
N = len(tlist)
deltas = np.linspace(0, 2*np.pi, 100)


phi_0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
# phi_0 = np.ones((6, 1), dtype=np.complex128) / np.sqrt(6)

store_dd = np.zeros((len(deltas), len(deltas), 6), dtype=np.float64)
rho_ii_t = np.zeros((len(tlist), 6), dtype=np.float64)

@nb.njit
def evolve_U(U_dt, N, phi_0):
    rho_ii_t = np.zeros((N, 6), dtype=np.float64)
    for i in np.arange(N):
        phi_0 = np.dot(U_dt, phi_0)
        rho_ii_t[i, :] = np.abs(phi_0)**2
    return rho_ii_t
    
  


for i, d1 in enumerate(deltas):
    # percentage done
    print("{:.2f}%".format(i/len(deltas)*100))
    for j, d2 in enumerate(deltas):
        # part 1
        
        a_23 = a_12 + d1
        a_31 = a_23 + d2
        
        H = np.array([
            [e1u,                 0,                 -tau_0,                 -tau_sf * exp(-1j*a_12),                 -tau_0,                 tau_sf * exp(-1j*a_31)],
            [0,                 e1d,                 tau_sf * exp(1j*a_12),                 -tau_0,                 -tau_sf * exp(1j*a_31),                 -tau_0],
            [-tau_0,    tau_sf * exp(-1j*a_12),                 e2u,                 0,                 -tau_0,                 -tau_sf * exp(-1j*a_23)],
            [-tau_sf * exp(1j*a_12),                 -tau_0,                 0,                 e2d,                 tau_sf * exp(1j*a_23),                 -tau_0],
            [-tau_0,                 -tau_sf * exp(-1j*a_31),                 -tau_0,                 tau_sf * exp(-1j*a_23),                 e3u,                 0],
            [tau_sf * exp(1j*a_31),                 -tau_0,                 -tau_sf * exp(1j*a_23),                 -tau_0,                 0,                 e3d]])
        

        
        # part 2
        
        U_dt = expm(-1j * H * dt)
        rho_ii_t = evolve_U(U_dt, N, phi_0)
        store_dd[j, i, :] = np.max(rho_ii_t, axis=0) # (j, i) = (x, y), lower
        
    
        

        
run = 'SZZ'

np.save('data/' + run + '.npy', store_dd)
#store_dd = np.load('data/' + run + '.npy')

# 6 colorplots, one for each state
fontsize = 15
with plt.style.context(['science']):
    #  plot for each state, except for the first one (which is always 1, so we only do 5)
    # but show it in 3x2 grid
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

    axs = [ax1, ax2, ax3, ax4, ax5]
    
    axs[0].set_title(r'$max(\rho_{1\downarrow 1\downarrow})$', fontsize=fontsize+1)
    axs[1].set_title(r'$max(\rho_{2\uparrow 2\uparrow})$', fontsize=fontsize+1)
    axs[2].set_title(r'$max(\rho_{2\downarrow 2\downarrow})$', fontsize=fontsize+1)
    axs[3].set_title(r'$max(\rho_{3\uparrow 3\uparrow})$', fontsize=fontsize+1)
    axs[4].set_title(r'$max(\rho_{3\downarrow 3\downarrow})$', fontsize=fontsize+1)

    

    for i in range(5):
        # colorbar for a range between 0 and 1
        im = axs[i].imshow(store_dd[:, :, i+1], origin='lower', cmap='viridis', vmin=0, vmax=1, extent=[0, 1, 0, 1])
        axs[i].set_xlabel(r'$\Delta_1/2\pi$', fontsize=fontsize)
        axs[i].set_ylabel(r'$\Delta_2/2\pi$', fontsize=fontsize)
        plt.colorbar(im, ax=axs[i])

            
    plt.tight_layout()
    plt.savefig('figs/' + run + '.png', dpi=300)
    plt.show()
    
    
# Se: tf=2.5*T0, nt=250
# Ses: tf=2.5*T0, nt=75
# S: tf=10*T0, nt=500
# Giga5000: tf=5000*T0, nt=10000
# GigaX500: tf=500*T0, nt=15000
# 1000S: tf=1000*T0, nt=15000
# 6000S: tf=6000*T0, nt=15000