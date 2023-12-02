# Triangular TQD with 1 electron and spin flip tunneling
from qutipDots import *
from numpy import exp
import qutip as qt
from scipy.linalg import expm
import matplotlib.pyplot as plt
import scienceplots
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
e1u = 1.0
e1d = 1.0
e2u = 1.0
e2d = 1.0
e3u = 1.0
e3d = 1.0
tau_0 = 1.0
tau_sf = 1.0
alphas = np.linspace(0, 2*np.pi, 10)

d1 = np.pi
d2 = np.pi

a_12 = 1.7 * np.pi
a_23 = a_12 + d1
a_31 = a_23 + d2

# Hamiltonian(s)
output = []

H = (
    # energies
    e1u * c1u_ * c1u + e1d * c1d_ * c1d + e2u * c2u_ * c2u + e2d * c2d_ * c2d + e3u * c3u_ * c3u + e3d * c3d_ * c3d 
    # non spin flip tunneling
    - tau_0 * (c1u_ * c2u + c2u_ * c1u + c1d_ * c2d + c2d_ * c1d + c2u_ * c3u + c3u_ * c2u + c2d_ * c3d + c3d_ * c2d
            + c1u_ * c3u + c3u_ * c1u + c1d_ * c3d + c3d_ * c1d) 
    # spin flip tunneling
    + tau_sf * (- exp(-1j*a_12) * c1u_ * c2d - exp(1j*a_12) * c2d_ * c1u + exp(1j*a_12) * c1d_ * c2u + exp(-1j*a_12) * c2u_ * c1d
            - exp(-1j*a_23) * c2u_ * c3d - exp(1j*a_23) * c3d_ * c2u + exp(1j*a_23) * c2d_ * c3u + exp(-1j*a_23) * c3u_ * c2d
            - exp(-1j*a_31) * c1u_ * c3d - exp(1j*a_31) * c3d_ * c1u + exp(1j*a_31) * c1d_ * c3u + exp(-1j*a_31) * c3u_ * c1d))





""" 
H_red = red_H(H, allowed_states)
You can check for time reversal symmetry
sy = np.array([[0, -1j], [1j, 0]]) / 2 # sigma_y 
Sy = np.kron(np.eye(3, 3), sy)
U = expm(-1j * np.pi * Sy)
Hu = U.T @ H_red @ U
print(Hu-np.conj(H))
"""

phi_0 = s10_00_00
rho_0 = phi_0 * phi_0.dag()

expect_ops = [c1u_ * c1u , c2u_ * c2u , c3u_ * c3u , c1d_ * c1d , c2d_ * c2d , c3d_ * c3d]
tlist = np.linspace(0, 20, 1000)
result = qt.mesolve(H, rho_0, tlist, [], expect_ops)
T_0 = 2.094 # period of oscillation of rho_11 for tau_sf = 0, T_0 = 2*pi/W_0, W_0 = 3*tau_0
tlist = tlist/T_0

with plt.style.context(['science']):
        """ fig, ax = plt.subplots(figsize=(8, 6))
        # Plot
        ax.plot(tlist, result.expect[0], label= r'$\rho_{11}$', c='c', linestyle='solid', linewidth=1.5)
        ax.plot(tlist, result.expect[1], label= r'$\rho_{22}$', color='orange', linestyle='dashed', linewidth=1.5)
        ax.plot(tlist, result.expect[2], label= r'$\rho_{33}$', color='black', linestyle='dotted', linewidth=1.5)

        # latex axis labels
        ax.set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=18)
        ax.set_ylabel(r'$\rho(t)$', fontsize=18)
        ax.legend(fontsize=18)
        
        # change tick size
        ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5)
        
        # make lines thicker
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        
        # set xlim 
        ax.set_xlim(0, 6)
        # save figure
        plt.savefig('figures/sf_a12=1,23pi+4,2pi,a23=13=4,2pi.png', dpi=300, bbox_inches='tight')
        plt.show() """
        # now subplot, 1x2, one for spin up, one for spin down
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Plot
        axs[0].plot(tlist, result.expect[0], label= r'$\rho_{11}$', c='c', linestyle='solid', linewidth=1.5)
        axs[0].plot(tlist, result.expect[1], label= r'$\rho_{22}$', color='orange', linestyle='dashed', linewidth=1.5)
        axs[0].plot(tlist, result.expect[2], label= r'$\rho_{33}$', color='black', linestyle='dotted', linewidth=1.5)
        axs[1].plot(tlist, result.expect[3], label= r'$\rho_{11}$', c='c', linestyle='solid', linewidth=1.5)
        axs[1].plot(tlist, result.expect[4], label= r'$\rho_{22}$', color='orange', linestyle='dashed', linewidth=1.5)
        axs[1].plot(tlist, result.expect[5], label= r'$\rho_{33}$', color='black', linestyle='dotted', linewidth=1.5)
        
        # latex axis labels
        axs[0].set_ylabel(r'$\rho(t)$', fontsize=18)
        axs[1].set_ylabel(r'$\rho(t)$', fontsize=18)
        axs[1].set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=18)

        # only one legend for both subplots, located on top of both
        axs[0].legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3)
        
        # change tick size
        axs[0].tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5)
        axs[1].tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5)
        
        
        # make lines thicker
        axs[0].spines['bottom'].set_linewidth(1.5)
        axs[0].spines['left'].set_linewidth(1.5)
        axs[0].spines['top'].set_linewidth(1.5)
        axs[0].spines['right'].set_linewidth(1.5)
        axs[1].spines['bottom'].set_linewidth(1.5)
        axs[1].spines['left'].set_linewidth(1.5)
        axs[1].spines['top'].set_linewidth(1.5)
        axs[1].spines['right'].set_linewidth(1.5)
        
        # set subtitles, spin up and spin down 
        axs[0].set_title(r'$m_s = +\frac{1}{2}$', fontsize=18, y=1.04)
        axs[1].set_title(r'$m_s = -\frac{1}{2}$', fontsize=18, y=1.04)
        
        
        axs[0].set_xlim(0, 6)
        axs[1].set_xlim(0, 6)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)
        
        # increase space between vertical subplots
        fig.subplots_adjust(hspace=0.2)
        
        # save figure
        plt.savefig('figures/Dsf_d1=d2=pi,a12=1,7pi', dpi=300, bbox_inches='tight')

        plt.show()

# special cases
# d1=pi, d2=0 -> la población (variando en cada spin) en dots 2, 3 es la misma, 
# d1=0, d2=pi -> lo mismo pero intercambiando el spin
# d1=pi, d2=pi -> la población en spin up dot 2 es la misma que en spin down dot 3, y viceversa
# d1=pi/2, d2=pi/2 ->la población (variando en cada spin) en dots 2, 3 es la misma, y además, la poblción en spin up es spin down invertida en el tiempo, también p11 llega a 1

