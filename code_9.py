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
tau_sf = tau_0

a_12 = 0
a_23 = a_12 
a_31 = a_23 

t23 = tau_0


T_0 = 2.094
tlist = np.linspace(0, 6000 * T_0, 15000)
deltas = np.linspace(-t23, +t23, 20)


phi_0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
# phi_0 = np.ones((6, 1), dtype=np.complex128) / np.sqrt(6)

""" store_dd = np.zeros((len(deltas), len(deltas), 6), dtype=np.float64)
    
for i, d1 in enumerate(deltas):
    # percentage done
    print("{:.2f}%".format(i/len(deltas)*100))
    for j, d2 in enumerate(deltas):
        t12= t23 + d1
        t31 = t23 + d2

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
        # unitary evoluction
        U_t = np.zeros((len(tlist), 6, 6), dtype=np.complex128)
        U_t = expm(-1j * H_red * tlist[:, None, None])
        # calculate phi(t) = U(t) phi(0)
        phi_t = np.matmul(U_t, phi_0)
        
        rho_ii_t = np.zeros((len(tlist), 6), dtype=np.float64)
        rho_ii_t = np.abs(phi_t)**2
        
        max_rho_ii = np.max(rho_ii_t, axis=0)
        store_dd[j, i, :] = np.squeeze(max_rho_ii)
        
        

np.save('data2/6.npy', store_dd) """


run = '6'
store_dd = np.load('data2/'+run+'.npy')

# 6 colorplots, one for each state
fontsize = 15
with plt.style.context(['science']):
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs = axs.flatten()


    axs[0].set_title(r'$max(\rho_{1\uparrow 1\uparrow})$', fontsize=fontsize+1)
    axs[1].set_title(r'$max(\rho_{1\downarrow 1\downarrow})$', fontsize=fontsize+1)
    axs[2].set_title(r'$max(\rho_{2\uparrow 2\uparrow})$', fontsize=fontsize+1)
    axs[3].set_title(r'$max(\rho_{2\downarrow 2\downarrow})$', fontsize=fontsize+1)
    axs[4].set_title(r'$max(\rho_{3\uparrow 3\uparrow})$', fontsize=fontsize+1)
    axs[5].set_title(r'$max(\rho_{3\downarrow 3\downarrow})$', fontsize=fontsize+1)



    for i in range(6):
        # colorbar for a range between 0 and 1
        im = axs[i].imshow(store_dd[:, :, i], origin='lower', cmap='viridis', extent=[0, 2, 0, 2], vmin=0, vmax=1)
        axs[i].set_xlabel(r'$t_{12}[\tau_0]$', fontsize=fontsize)
        axs[i].set_ylabel(r'$t_{31}[\tau_0]$', fontsize=fontsize)
        fig.colorbar(im, ax=axs[i])
            
    plt.tight_layout()
    plt.savefig('figs/6.png', dpi=300)
    plt.show()
    
    
