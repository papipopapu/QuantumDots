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
into1 = np.sqrt(g1) * c1_
out3 = np.sqrt(g3) * c3


I = n3 # I/Gamma

# to qobj
I = qt.Qobj(I)
n1 = qt.Qobj(n1)
n2 = qt.Qobj(n2)
n3 = qt.Qobj(n3)
into1 = qt.Qobj(into1)
out3 = qt.Qobj(out3)




N = 300
e1s = np.linspace(-10*tau, 10*tau, N)
Is = np.zeros(N)
rhos = np.zeros((N, 6))

for i, e1p in tqdm(enumerate(e1s)):
    e1 = e1p
    H = np.array([
    [0, 0, 0, 0],
    [0, e1, -tau, 0],
    [0, -tau, e2, -tau],
    [0, 0, -tau, e3]
    ])
        
    H = qt.Qobj(H)
    
    # calculate steady state current
    rhoss = qt.steadystate(H, [into1, out3])
    Is[i] = qt.expect(I, rhoss)
    rhos[i, 0] = qt.expect(n1, rhoss)
    rhos[i, 1] = qt.expect(n2, rhoss)
    rhos[i, 2] = qt.expect(n3, rhoss)

    
    
# save data
np.save('data/I_1emaxNospin_linear.npy', Is)



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
    axs.set_ylim(bottom=0)
    # save
    plt.savefig('figs/I_1emaxNospin_linear.png', dpi=300, bbox_inches='tight')

plt.show()

