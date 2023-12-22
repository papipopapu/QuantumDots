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

tau_0 = tau
tau_sf = tau
a_12 = 0
a_23 = 0
a_31 = 0

g1 = tau / 2
g3 = tau / 2

c1_ = fcreate(3, 0)
c2_ = fcreate(3, 1)
c3_ = fcreate(3, 2)


c1 = c1_.dag()
c2 = c2_.dag()
c3 = c3_.dag()


n1 = c1_ * c1
n2 = c2_ * c2
n3 = c3_ * c3


I = n3 # I/Gamma




# collapse operators
into1 = np.sqrt(g1) * c1_ 
#out2 = np.sqrt(g2) * c2u + np.sqrt(g2) * c2d
out3 = np.sqrt(g3) * c3

N = 300
e1s = np.linspace(-10*tau, 10*tau, N)
Is = np.zeros(N)
rhos = np.zeros((N, 6))


s_0_0_0 = basis([2,2,2], [0,0,0])
s_1_0_0 = basis([2,2,2], [1,0,0])
s_0_1_0 = basis([2,2,2], [0,1,0])
s_0_0_1 = basis([2,2,2], [0,0,1])
allowed = [s_0_0_0, s_1_0_0, s_0_1_0, s_0_0_1]

into1 = red_H(into1, allowed)
out3 = red_H(out3, allowed)
I = red_H(I, allowed)
n1 = red_H(n1, allowed)
n2 = red_H(n2, allowed)
n3 = red_H(n3, allowed)

into1 = qt.Qobj(into1)
out3 = qt.Qobj(out3)
I = qt.Qobj(I)
n1 = qt.Qobj(n1)
n2 = qt.Qobj(n2)
n3 = qt.Qobj(n3)
for i, e1p in tqdm(enumerate(e1s)):
    e1 = e1p
    H = (
    # energies
    e1 * c1_ * c1 + e2 * c2_ * c2 + e3 * c3_ * c3 
    # non spin flip tunneling
    - tau_0 * (c1_ * c2 + c2_ * c1 + 
               c2_ * c3 + c3_ * c2 +
               c1_ * c3 + c3_ * c1)
               
    )
    
    H = red_H(H, allowed)
    
    
    H = qt.Qobj(H)
    
    # calculate steady state current
    rhoss = qt.steadystate(H, [into1, out3])
    Is[i] = qt.expect(I, rhoss)
    rhos[i, 0] = qt.expect(n1, rhoss)
    rhos[i, 1] = qt.expect(n2, rhoss)
    rhos[i, 2] = qt.expect(n3, rhoss)

    
    
# save data
np.save('data/I_3emaxNospin.npy', Is)

#Is = np.load('data/Is.npy')
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
    # save
    plt.savefig('figs/I_3emaxNospin.png', dpi=300, bbox_inches='tight')

""" with plt.style.context(['science']):
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

    axs.plot(e1s/tau, rhos[:, 0], label= r'$\eta=1$', c='c', linestyle='solid', linewidth=1.5)
    axs.plot(e1s/tau, rhos[:, 2], label= r'$\eta=2$', color='orange', linestyle='dashed', linewidth=1.5)
    axs.plot(e1s/tau, rhos[:, 4], label= r'$\eta=3$', color='black', linestyle='dotted', linewidth=1.5)
   
    

    # latex axis labels, move y label a bit to the left
    axs.set_ylabel(r'$\rho$', fontsize=30, labelpad=25)

    axs.legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=3)
    axs.tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)

    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['left'].set_linewidth(1.5)
    axs.spines['top'].set_linewidth(1.5)
    axs.spines['right'].set_linewidth(1.5)
    #axs[0].set_ylim(0, 1)
    #axs[1].set_ylim(0, 1)
    axs.set_xlim(-10, 10)

    
    # save
    plt.savefig('figs/1000rhos(e1)_ppm_linear.png', dpi=300, bbox_inches='tight') """
plt.show()

