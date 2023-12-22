# Triangular TQD with 1 electron and spin flip tunneling
from qutipDots import *
from numpy import exp
from qutip import tensor
import qutip as qt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import scienceplots
from scipy.linalg import eig, eigvalsh
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

deltas = np.linspace(0, 2*np.pi, 50)


phi_0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
# phi_0 = np.ones((6, 1), dtype=np.complex128) / np.sqrt(6)
 
store_dd = np.zeros((len(deltas), len(deltas)), dtype=np.float64)
if True:       
    for i, d1 in enumerate(deltas):
        # percentage done
        print("{:.2f}%".format(i/len(deltas)*100))
        for j, d2 in enumerate(deltas):
            a_23 = a_12 + d1
            a_31 = a_23 + d2
            H = (
            # energies
            e1u * c1u_ * c1u + e1d * c1d_ * c1d + e2u * c2u_ * c2u + e2d * c2d_ * c2d + e3u * c3u_ * c3u + e3d * c3d_ * c3d 
            # non spin flip tunneling
            - tau_0 * (c1u_ * c2u + c2u_ * c1u + c1d_ * c2d + c2d_ * c1d + c2u_ * c3u + c3u_ * c2u + c2d_ * c3d + c3d_ * c2d
                    + c1u_ * c3u + c3u_ * c1u + c1d_ * c3d + c3d_ * c1d) 
            # spin flip tunneling
            + tau_sf * (- t12 * exp(-1j*a_12) * c1u_ * c2d - t12 * exp(1j*a_12) * c2d_ * c1u + t12 * exp(1j*a_12) * c1d_ * c2u + t12 * exp(-1j*a_12) * c2u_ * c1d
                    - t23 * exp(-1j*a_23) * c2u_ * c3d - t23 * exp(1j*a_23) * c3d_ * c2u + t23 * exp(1j*a_23) * c2d_ * c3u + t23 * exp(-1j*a_23) * c3u_ * c2d
                    + t31 * exp(-1j*a_31) * c1u_ * c3d + t31 * exp(1j*a_31) * c3d_ * c1u - t31 * exp(1j*a_31) * c1d_ * c3u - t31 * exp(-1j*a_31) * c3u_ * c1d))
            
            H_red = red_H(H, allowed_states)
            evals, evecs = eig(H_red)
            print("/////////////////////////")
            for evec in evecs:
                print(evec)

np.save('data/en5.npy', store_dd)
#store_dd = np.load('data/en5.npy')
fontsize = 25

with plt.style.context(['science']):

    
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    im = axs.imshow(store_dd[:, :], cmap='plasma', extent=[0, 1, 0, 1], origin='lower', norm=colors.LogNorm())
    axs.set_xlabel(r'$\Delta_1/2\pi$', fontsize=fontsize)
    axs.set_ylabel(r'$\Delta_2/2\pi$', fontsize=fontsize, labelpad=10)
    axs.set_title(r'$min(|E_n|)$', fontsize=fontsize+1, pad=15)
    
    cbar = fig.colorbar(im, ax=axs)
    axs.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs.set_yticks([0, 0.25, 0.5, 0.75, 1])
    
    axs.tick_params(axis='both', which='major', labelsize=22)
        
    cbar.ax.tick_params(labelsize=20)

    
    
    
    
    plt.tight_layout()
    
    plt.savefig('figs/en5.png', dpi=300)
    
    
    plt.show()
    
    