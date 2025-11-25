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

c1_ = fcreate(3, 0)
c2_ = fcreate(3, 1)
c3_ = fcreate(3, 2)

c1 = fdestroy(3, 0)
c2 = fdestroy(3, 1)
c3 = fdestroy(3, 2)

n1 = c1_ * c1
n2 = c2_ * c2
n3 = c3_ * c3

H0 = (
    # energies
    e1 * c1_ * c1 + e2 * c2_ * c2 + e3 * c3_ * c3 
    # non spin flip tunneling
    - tau * (c1_ * c2 + c2_ * c1 + 
               c2_ * c3 + c3_ * c2 
            ) 
    )

# collapse operators
into = np.sqrt(gL) * c1_
out = np.sqrt(gR) * c3

# solve dynamics
omega = 2*np.sqrt(2)*tau
T0 = 2*np.pi/omega # new time unit
tlist = np.linspace(0, 4*T0, 1000)
s1_0_0 = basis([2,2,2], [1,0,0])
rho0 = s1_0_0 * s1_0_0.dag()
# transform all elemnts to Qobj
H0 = qt.Qobj(H0)
into = qt.Qobj(into)
out = qt.Qobj(out)
rho0 = qt.Qobj(rho0)
n1 = qt.Qobj(n1)
n2 = qt.Qobj(n2)
n3 = qt.Qobj(n3)
print(n3)
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
    
    # save
    plt.savefig('figs/dont_worky.png', dpi=300, bbox_inches='tight')


plt.show()

