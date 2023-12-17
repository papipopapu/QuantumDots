from qutipDots import *
import qutip as qt
from numpy import exp
import matplotlib.pyplot as plt
import scienceplots
from qutip import fcreate, fdestroy, qeye, tensor
plt.style.use('science')
# tqd with no spin
e1 = 0
e2 = 0
e3 = 0
tau = 0.1
gR = tau / 10
gL = tau / 10
# H0 in the basis of |0, 0, 0>, |1, 0, 0>, |0, 1, 0>, |0, 0, 1>
H0 = np.array([
    [0, 0, 0, 0],
    [0, e1, -tau, 0],
    [0, -tau, e2, -tau],
    [0, 0, -tau, e3]
])
# lets write the basis states and creation/annihilation operators
s0_0_0 = np.array([1, 0, 0, 0])
s1_0_0 = np.array([0, 1, 0, 0])
s0_1_0 = np.array([0, 0, 1, 0])
s0_0_1 = np.array([0, 0, 0, 1])

c1_ = s1_0_0[:, np.newaxis] @ s0_0_0[np.newaxis, :]
c2_ = s0_1_0[:, np.newaxis] @ s0_0_0[np.newaxis, :]
c3_ = s0_0_1[:, np.newaxis] @ s0_0_0[np.newaxis, :]

c1 = c1_.T
c2 = c2_.T
c3 = c3_.T

n1 = c1_ @ c1
n2 = c2_ @ c2
n3 = c3_ @ c3

# collapse operators
into = np.sqrt(gL) * c1_
out = np.sqrt(gR) * c3

# solve dynamics
omega = 2*np.sqrt(2)*tau
T0 = 2*np.pi/omega # new time unit
tlist = np.linspace(0, 4*T0, 1000)
rho0 = s1_0_0[:, np.newaxis] @ s1_0_0[np.newaxis, :]
# transform all elemnts to Qobj
H0 = qt.Qobj(H0)
into = qt.Qobj(into)
out = qt.Qobj(out)
rho0 = qt.Qobj(rho0)
n1 = qt.Qobj(n1)
n2 = qt.Qobj(n2)
n3 = qt.Qobj(n3)
# solve
rhos = qt.mesolve(H0, rho0, tlist, [into, out], [n1, n2, n3])

# plot diagonal elements
with plt.style.context(['science']):
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    axs.plot(tlist/T0, rhos.expect[0], label='$\\rho_{11}$', c='c', linestyle='solid', linewidth=1.5)
    axs.plot(tlist/T0, rhos.expect[1], label='$\\rho_{22}$', color='orange', linestyle='dashed', linewidth=1.5)
    axs.plot(tlist/T0, rhos.expect[2], label='$\\rho_{33}$', color='black', linestyle='dotted', linewidth=1.5)
    axs.set_ylabel(r'$\rho_{\eta\eta}(t)$', fontsize=30, labelpad=25)
    axs.set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=30)
    axs.legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    axs.tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['left'].set_linewidth(1.5)
    axs.spines['top'].set_linewidth(1.5)
    axs.spines['right'].set_linewidth(1.5)
    axs.set_xlim(0, 4)
    axs.set_ylim(0, 1)
    
    


# now with spin and spin flip and no limit to number of electrons
# two drains (2, 3) and one source (1)

tau_0 = 1.0
tau_sf = 1.0
a_12 = 0
a_23 = 0
a_31 = 0

g1 = tau_0 / 10
g2 = tau_0 / 10
g3 = tau_0 / 10

Id = qeye([2, 2])
c1u_ = tensor(tensor(fcreate(2, 0), Id), Id)
c1d_ = tensor(tensor(fcreate(2, 1), Id), Id)
c2u_ = tensor(tensor(Id, fcreate(2, 0)), Id)
c2d_ = tensor(tensor(Id, fcreate(2, 1)), Id)
c3u_ = tensor(tensor(Id, Id), fcreate(2, 0))
c3d_ = tensor(tensor(Id, Id), fcreate(2, 1))

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


c1u_ = c1u.dag()
c1d_ = c1d.dag()
c2u_ = c2u.dag()
c2d_ = c2d.dag()
c3u_ = c3u.dag()
c3d_ = c3d.dag() 

n1u = c1u_ * c1u
n1d = c1d_ * c1d
n2u = c2u_ * c2u
n2d = c2d_ * c2d
n3u = c3u_ * c3u
n3d = c3d_ * c3d
I = g2 * (n2u + n2d) + g3 * (n3u + n3d)
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


# collapse operators
into1 = np.sqrt(g1) * c1u_ + np.sqrt(g1) * c1d_
out2 = np.sqrt(g2) * c2u + np.sqrt(g2) * c2d
out3 = np.sqrt(g3) * c3u + np.sqrt(g3) * c3d

# solve dynamics
tlist = np.linspace(0, 4*T0, 1000)
phi0 = basis([2,2,2,2,2,2], [1,0,0,0,0,0])
rho0 = phi0 * phi0.dag()
# transform all elemnts to Qobj
H = qt.Qobj(H)
into1 = qt.Qobj(into1)
out2 = qt.Qobj(out2)
out3 = qt.Qobj(out3)
rho0 = qt.Qobj(rho0)
# solve
rhos = qt.mesolve(H, rho0, tlist, [into1, out2, out3], [n1u, n1d, n2u, n2d, n3u, n3d, I])

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
    axs[0].set_xlim(0, 2)
    axs[1].set_xlim(0, 2)
    
# now plot current

with plt.style.context(['science']):
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    axs.plot(tlist/T0, rhos.expect[6], label= r'$I(t)$', c='c', linestyle='solid', linewidth=1.5)
    axs.set_ylabel(r'$I(t)$', fontsize=30, labelpad=25)
    axs.set_xlabel(r'$t[\Omega_0/2\pi]$', fontsize=30)
    axs.legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=3)
    axs.tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['left'].set_linewidth(1.5)
    axs.spines['top'].set_linewidth(1.5)
    axs.spines['right'].set_linewidth(1.5)
    axs.set_xlim(0, 2)



plt.show()

