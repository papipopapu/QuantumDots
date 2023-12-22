# Triangular TQD with 1 electron and spin flip tunneling
from qutipDots import *
from numpy import exp
from qutip import tensor
import qutip as qt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
from scipy.linalg import eig
plt.style.use('science')
import numpy as np


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
tlist = np.linspace(0, 30000*T_0, 15000)
deltas = np.linspace(0, 2*np.pi, 15)


phi_0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)


store_dd = np.load('data/6000S.npy')

fontsize = 25
with plt.style.context(['science']):

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    im = axs.imshow(np.tile(store_dd[:, :, 2], (3, 3)), cmap='viridis', extent=[0, 3, 0, 3], vmin=0, vmax=1, origin='lower')
    axs.set_xlabel(r'$\Delta_1/2\pi$', fontsize=fontsize)
    axs.set_ylabel(r'$\Delta_2/2\pi$', fontsize=fontsize, labelpad=10)
    axs.set_title(r'$\rho_{2\uparrow 2\uparrow}$', fontsize=fontsize+1, pad=15)
    
    cbar = fig.colorbar(im, ax=axs)
    axs.set_xticks([0, 1, 2, 3])
    axs.set_yticks([0, 1, 2, 3])
    
    axs.tick_params(axis='both', which='major', labelsize=fontsize)
        
    # increase colorbar tick size and set ticks at 0, 0.25, 0.5, 0.75, 1
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    
    
    
    
    plt.tight_layout()
    
    plt.savefig('figs/2tile6000S.png', dpi=300)
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    im = axs.imshow(np.tile(store_dd[:, :, 3], (3, 3)), cmap='viridis', extent=[0, 3, 0, 3], vmin=0, vmax=1, origin='lower')
    axs.set_xlabel(r'$\Delta_1/2\pi$', fontsize=fontsize)
    axs.set_ylabel(r'$\Delta_2/2\pi$', fontsize=fontsize, labelpad=10)
    axs.set_title(r'$\rho_{2\downarrow 2\downarrow}$', fontsize=fontsize+1, pad=15)
    
    cbar = fig.colorbar(im, ax=axs)
    axs.set_xticks([0, 1, 2, 3])
    axs.set_yticks([0, 1, 2, 3])
    
    axs.tick_params(axis='both', which='major', labelsize=fontsize)
        
    # increase colorbar tick size and set ticks at 0, 0.25, 0.5, 0.75, 1
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    
    
    
    
    plt.tight_layout()
    
    plt.savefig('figs/3tile6000S.png', dpi=300)
    
    
    
    
    
    
    
    
    
    
    
    plt.show()
    
    