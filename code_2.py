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
e1u = 0
e1d = 0
e2u = 0
e2d = 0
e3u = 0
e3d = 0
tau_0 = 1
tau_sf = 1
alphas = np.linspace(0, 2*np.pi, 10)

d1 = 0
d2 = 0
# if sum=0, space time symmetry
# if sum=pi, space up-down symmetry
# if sum=pi/2, no symmetry
a_12 = np.pi
a_23 = a_12 + d1
a_31 = a_23 + d2


t12 = tau_0 
t23 = tau_0
t31 = tau_0 

""" phi_0 = s10_00_00 
rho_0 = phi_0 * phi_0.dag() """
phi_0 = np.array([1, 0,0,0,0,0], dtype=np.complex128) 


filename = 'p1u2u3uf_d1='+str(d1/np.pi)+'pi_d2='+str(d2/np.pi)+'pi_t12='+str(t12)+'_t23='+str(t23)+'_t31='+str(t31)+'_a12='+str(a_12/np.pi)+'pi_tau_0='+str(tau_0)+'_tau_sf='+str(tau_sf)+'_e1u='+str(e1u)+'_e1d='+str(e1d)+'_e2u='+str(e2u)+'_e2d='+str(e2d)+'_e3u='+str(e3u)+'_e3d='+str(e3d)+'.png'
filename = 'socpi'
# Hamiltonian(s)
output = []


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
#  can check for time reversal symmetry
sy = np.array([[0, -1j], [1j, 0]]) / 2 # sigma_y 
Sy = np.kron(np.eye(3, 3), sy)
U = expm(-1j * np.pi * Sy)
Hu = U.T @ H_red @ U
print(Hu-np.conj(H_red))
print(H_red)

""" He = qt.Qobj(H_red)
evals, evecs = He.eigenstates()
print(evals)
 """


T_0 = 2.094
tlist = np.linspace(0, 10*T_0, 500)
T_0 = 2.094 # period of oscillation of rho_11 for tau_sf = 0, T_0 = 2*pi/W_0, W_0 = 3*tau_0

def evolve_U(H, tlist, phi_0):
    U_t = np.zeros((len(tlist), 6, 6), dtype=np.complex128)
    U_t = expm(-1j * H * tlist[:, None, None])
    phi_t = np.matmul(U_t, phi_0)
    return phi_t

phi_t = evolve_U(H_red, tlist, phi_0)
rho_ii_t = np.zeros((len(tlist), 6), dtype=np.float64)
rho_ii_t = np.abs(phi_t)**2
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
        axs[0].plot(tlist/T_0, rho_ii_t[:, 0], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
        axs[0].plot(tlist/T_0, rho_ii_t[:, 2], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
        axs[0].plot(tlist/T_0, rho_ii_t[:, 4], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)
        axs[1].plot(tlist/T_0, rho_ii_t[:, 1], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
        axs[1].plot(tlist/T_0, rho_ii_t[:, 3], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
        axs[1].plot(tlist/T_0, rho_ii_t[:, 5], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)
        
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
        
        axs[0].set_xlim(0, 6)
        axs[1].set_xlim(0, 6)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)
        
        # increase space between vertical subplots
        # fig.subplots_adjust(hspace=0.2)
        
        # save figure
        plt.savefig('figs/polla', dpi=300, bbox_inches='tight')

        plt.show()

# special cases
# d1=pi, d2=0 -> la población (variando en cada spin) en dots 2, 3 es la misma, 
# d1=0, d2=pi -> lo mismo pero intercambiando el spin
# d1=pi, d2=pi -> la población en spin up dot 2 es la misma que en spin down dot 3, y viceversa
# d1=pi/2, d2=pi/2 ->la población (variando en cada spin) en dots 2, 3 es la misma, y además, la poblción en spin up es spin down invertida en el tiempo, también p11 llega a 1

# la elección de las fases es importante:
# si d1=d2=0, empezando de un estado 1u+2u+3u, la evolución de 1 y 3 es la misma, pero la de 2 es distinta (no hay spin down en el dot 2),
# si d1=d2=pi, ahora son 1 y 2 los que evolucionan igual,
# si d1=pi, d2=0, ahora son el 2 y el 3
# si d1=0, d2=pi, los 3 evolucionan igual (autoestados de H)