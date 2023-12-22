from qutipDots import *
import qutip as qt
from numpy import exp
import matplotlib.pyplot as plt
import scienceplots
from qutip import fcreate, fdestroy, qeye, tensor
from tqdm import tqdm
plt.style.use('science')
# tqd with no spin
e1 = 0
e2 = 0
e3 = 0
tau = 0.1
gR = tau / 10
gL = tau / 10
omega = 2*np.sqrt(2)*tau
T0 = 2*np.pi/omega # new time unit

tau_0 = tau
tau_sf = tau
a_12 = 0
a_23 = 0
a_31 = 0

g1 = tau / 2
g2 = tau / 2
g3 = tau / 2

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

I = (n2u + n2d) + (n3u + n3d) # I/Gamma




# collapse operators
into1 = np.sqrt(g1) * c1u_ + np.sqrt(g1) * c1d_
out2 = np.sqrt(g2) * c2u + np.sqrt(g2) * c2d
out3 = np.sqrt(g3) * c3u + np.sqrt(g3) * c3d


phi0 = basis([2,2,2,2,2,2], [1,0,0,0,0,0])
rho0 = phi0 * phi0.dag()



N = 100
e1s = np.linspace(-10*tau, 10*tau, N)
Is = np.zeros(N)
rhos = np.zeros((N, 6))
rho0 = qt.Qobj(rho0)

for i, e1p in tqdm(enumerate(e1s)):
    e1 = e1p
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
    
    H = qt.Qobj(H)
    
    # calculate steady state current
    rhoss = qt.steadystate(H, [np.sqrt(g1) * c1u_ , np.sqrt(g1) * c1d_, np.sqrt(g2) * c2u , np.sqrt(g2) * c2d, np.sqrt(g3) * c3u , np.sqrt(g3) * c3d])
    Is[i] = qt.expect(I, rhoss)
    rhos[i, 0] = qt.expect(n1u, rhoss)
    rhos[i, 1] = qt.expect(n1d, rhoss)
    rhos[i, 2] = qt.expect(n2u, rhoss)
    rhos[i, 3] = qt.expect(n2d, rhoss)
    rhos[i, 4] = qt.expect(n3u, rhoss)
    rhos[i, 5] = qt.expect(n3d, rhoss)
    
    
# save data
np.save('data/I_Nemax_spinp.npy', Is)
#np.save('data/1000rhos_e1.npy', rhos)


#Is = np.load('data/I_Nemax_spinp.npy')
# plot
    

    
with plt.style.context(['science']):
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    axs.plot(e1s/tau, Is,  c='c', linestyle='solid', linewidth=1.5)
    axs.set_ylabel(r'$I/\Gamma$', fontsize=30, labelpad=25)
    axs.set_xlabel(r'$\epsilon_1/\tau$', fontsize=30)
    axs.tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['left'].set_linewidth(1.5)
    axs.spines['top'].set_linewidth(1.5)
    axs.spines['right'].set_linewidth(1.5)
    axs.set_xlim(-10, 10)
    axs.set_ylim(bottom=0)
    # save
    plt.savefig('figs/I_Nemax_spinp.png', dpi=300, bbox_inches='tight')

plt.show()

