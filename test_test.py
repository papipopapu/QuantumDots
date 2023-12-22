from qutipDots import *
import qutip as qt
from numpy import exp
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis
from qutip import fcreate, fdestroy
plt.style.use('science')

e1 = 0
e2 = 0
e3 = 0
tau_0 = 1.0
tau_sf = 1.0
a_12 = 0
a_23 = 0
a_31 = 0
omega = 2 * np.sqrt(2) * tau_0
T0 = 2.094

g1 = tau_0 / 10
g2 = tau_0 / 10
g3 = tau_0 / 10

# Relevant operators
c1u_ = fcreate(6, 0)
c1d_ = fcreate(6, 1)
c2u_ = fcreate(6, 2)
c2d_ = fcreate(6, 3)
c3u_ = fcreate(6, 4)
c3d_ = fcreate(6, 5)



c1u = c1u_.dag()
c1d = c1d_.dag()
c2u = c2u_.dag()
c2d = c2d_.dag()
c3u = c3u_.dag()
c3d = c3d_.dag()

n1u = c1u_ * c1u
n1d = c1d_ * c1d
n2u = c2u_ * c2u
n2d = c2d_ * c2d
n3u = c3u_ * c3u
n3d = c3d_ * c3d


H = (
    # energies
    e1 * c1u_ * c1u + e1 * c1d_ * c1d + e2 * c2u_ * c2u + e2 * c2d_ * c2d + e3 * c3u_ * c3u + e3 * c3d_ * c3d 
    # non spin flip tunneling
    - tau_0 * (c1u_ * c2u + c2u_ * c1u + c1d_ * c2d + c2d_ * c1d + c2u_ * c3u + c3u_ * c2u + c2d_ * c3d + c3d_ * c2d
            + c1u_ * c3u + c3u_ * c1u + c1d_ * c3d + c3d_ * c1d) 
    # spin flip tunneling
    + tau_sf * (- exp(-1j*a_12) * c1u_ * c2d - exp(1j*a_12) * c2d_ * c1u + exp(1j*a_12) * c1d_ * c2u + exp(-1j*a_12) * c2u_ * c1d
                - exp(-1j*a_23) * c2u_ * c3d - exp(1j*a_23) * c3d_ * c2u + exp(1j*a_23) * c2d_ * c3u + exp(-1j*a_23) * c3u_ * c2d
                + exp(-1j*a_31) * c1u_ * c3d + exp(1j*a_31) * c3d_ * c1u - exp(1j*a_31) * c1d_ * c3u - exp(-1j*a_31) * c3u_ * c1d))



# solve dynamics
tlist = np.linspace(0, 6*T0, 1000)
phi0 = basis([2,2,2,2,2,2], [1,0,0,0,0,0])
rho0 = phi0 * phi0.dag()
# transform all elemnts to Qobj
H = qt.Qobj(H)
rho0 = qt.Qobj(rho0)
n1u = qt.Qobj(n1u)
n1d = qt.Qobj(n1d)
n2u = qt.Qobj(n2u)
n2d = qt.Qobj(n2d)
n3u = qt.Qobj(n3u)
n3d = qt.Qobj(n3d)



solve = True
if solve is True:
    # solve
    rhos = qt.mesolve(H, rho0, tlist, [], [n1u, n1d, n2u, n2d, n3u, n3d])

    with plt.style.context(['science']):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(tlist/T0, rhos.expect[0], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
        axs[0].plot(tlist/T0, rhos.expect[2], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
        axs[0].plot(tlist/T0, rhos.expect[4], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)
        axs[1].plot(tlist/T0, rhos.expect[1], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
        axs[1].plot(tlist/T0, rhos.expect[3], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
        axs[1].plot(tlist/T0, rhos.expect[5], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)

        # latex axis labels, move y label a bit to the left
        axs[0].set_ylabel(r'$\rho_{\uparrow\uparrow}(t)$', fontsize=30, labelpad=25)
        axs[1].set_ylabel(r'$\rho_{\downarrow\downarrow}(t)$', fontsize=30, labelpad=25)
        axs[1].set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=30)
        axs[0].legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=3)
        axs[0].tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
        axs[1].tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
        axs[0].spines['bottom'].set_linewidth(1.5)
        axs[0].spines['left'].set_linewidth(1.5)
        axs[0].spines['top'].set_linewidth(1.5)
        axs[0].spines['right'].set_linewidth(1.5)
        axs[1].spines['bottom'].set_linewidth(1.5)
        axs[1].spines['left'].set_linewidth(1.5)
        axs[1].spines['top'].set_linewidth(1.5)
        axs[1].spines['right'].set_linewidth(1.5)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)
        axs[0].set_xlim(0, 6)
        axs[1].set_xlim(0, 6)
        
    # now plot current


    plt.show()

