from qutipDots import *
import qutip as qt
from numpy import exp
import matplotlib.pyplot as plt
import scienceplots
from qutip import fcreate, fdestroy, qeye, tensor, sigmaz
from tqdm import tqdm
plt.style.use('science')
# tqd with no spin
e1 = 0
e2 = 0
e3 = 0
tau = 0.1

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

s00_00_00 = np.array([1, 0, 0, 0, 0, 0, 0])
s10_00_00 = np.array([0, 1, 0, 0, 0, 0, 0])
s01_00_00 = np.array([0, 0, 1, 0, 0, 0, 0])
s00_10_00 = np.array([0, 0, 0, 1, 0, 0, 0])
s00_01_00 = np.array([0, 0, 0, 0, 1, 0, 0])
s00_00_10 = np.array([0, 0, 0, 0, 0, 1, 0])
s00_00_01 = np.array([0, 0, 0, 0, 0, 0, 1])



c1u_ = s10_00_00[:, np.newaxis] @ s00_00_00[np.newaxis, :]
c1d_ = s01_00_00[:, np.newaxis] @ s00_00_00[np.newaxis, :]
c2u_ = s00_10_00[:, np.newaxis] @ s00_00_00[np.newaxis, :]
c2d_ = s00_01_00[:, np.newaxis] @ s00_00_00[np.newaxis, :]
c3u_ = s00_00_10[:, np.newaxis] @ s00_00_00[np.newaxis, :]
c3d_ = s00_00_01[:, np.newaxis] @ s00_00_00[np.newaxis, :]

c1u = c1u_.T
c1d = c1d_.T
c2u = c2u_.T
c2d = c2d_.T
c3u = c3u_.T
c3d = c3d_.T

n1u = c1u_ @ c1u
n1d = c1d_ @ c1d
n2u = c2u_ @ c2u
n2d = c2d_ @ c2d
n3u = c3u_ @ c3u
n3d = c3d_ @ c3d


I = n3u + n3d + n2u + n2d
# I/Gamma




# collapse operators
into1 = np.sqrt(g1) * c1u_ + np.sqrt(g1) * c1d_
out2 = np.sqrt(g2) * c2u + np.sqrt(g2) * c2d
out3 = np.sqrt(g3) * c3u + np.sqrt(g3) * c3d

# to Qobj
into1 = qt.Qobj(into1)
out3 = qt.Qobj(out3)
out2 = qt.Qobj(out2)
I = qt.Qobj(I)
c1u_ = qt.Qobj(c1u_)
c1d_ = qt.Qobj(c1d_)
c2u_ = qt.Qobj(c2u_)
c2d_ = qt.Qobj(c2d_)
c3u_ = qt.Qobj(c3u_)
c3d_ = qt.Qobj(c3d_)
c1u = qt.Qobj(c1u)
c1d = qt.Qobj(c1d)
c2u = qt.Qobj(c2u)
c2d = qt.Qobj(c2d)
c3u = qt.Qobj(c3u)
c3d = qt.Qobj(c3d)
n1u = qt.Qobj(n1u)
n1d = qt.Qobj(n1d)
n2u = qt.Qobj(n2u)
n2d = qt.Qobj(n2d)
n3u = qt.Qobj(n3u)
n3d = qt.Qobj(n3d)


N = 300
global_phases = np.linspace(0,2*np.pi, N)
Is = np.zeros(N)
rhos = np.zeros((N, 6))


for i, gp in tqdm(enumerate(global_phases)):
    a_12 = gp
    a_23 = gp
    a_31 = gp
    H = (
    # energies
    e1 * c1u_ * c1u + e1 * c1d_ * c1d + e2 * c2u_ * c2u + e2 * c2d_ * c2d + e3 * c3u_ * c3u + e3 * c3d_ * c3d 
    # non spin flip tunneling
    - tau_0 * (c1u_ * c2u + c2u_ * c1u + c1d_ * c2d + c2d_ * c1d + c2u_ * c3u + c3u_ * c2u + c2d_ * c3d + c3d_ * c2d
            + c1u_ * c3u + c3u_ * c1u + c1d_ * c3d + c3d_ * c1d) 
    # spin flip tunneling
    + tau_sf * (- exp(-1j*a_12) * c1u_ * c2d - exp(1j*a_12) * c2d_ * c1u + exp(1j*a_12) * c1d_ * c2u + exp(-1j*a_12) * c2u_ * c1d
                - exp(-1j*a_23) * c2u_ * c3d - exp(1j*a_23) * c3d_ * c2u + exp(1j*a_23) * c2d_ * c3u + exp(-1j*a_23) * c3u_ * c2d
                + exp(-1j*a_31) * c1u_ * c3d + exp(1j*a_31) * c3d_ * c1u - exp(1j*a_31) * c1d_ * c3u - exp(-1j*a_31) * c3u_ * c1d)
    )
    
    # calculate steady state current

    rhoss = qt.steadystate(H, [np.sqrt(g1) * c1u_ , np.sqrt(g1) * c1d_, np.sqrt(g2) * c2u , np.sqrt(g2) * c2d, np.sqrt(g3) * c3u , np.sqrt(g3) * c3d])
    trace = rhoss.tr()
    diff = 1 - trace
    if diff > 1e-10:
        print('trace = ', trace)
        print('diff = ', diff)
        quit()
    Is[i] = qt.expect(I, rhoss)
    
    
# save data
np.save('data/I_1emax_Spin_1to2,3(alpha).npy', Is)


#Is = np.load('data/Is.npy')
# plot

    
with plt.style.context(['science']):
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    axs.plot(global_phases/(2*np.pi), Is,  c='c', linestyle='solid', linewidth=1.5)
    axs.set_ylabel(r'$I/\Gamma$', fontsize=30, labelpad=5)
    axs.set_xlabel(r'$\alpha/2\pi$', fontsize=30)
    axs.tick_params(axis='both', which='major', labelsize=30, length=8, width=1.5)
    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['left'].set_linewidth(1.5)
    axs.spines['top'].set_linewidth(1.5)
    axs.spines['right'].set_linewidth(1.5)
    # add padding to numbers in x and y axis
    axs.tick_params(pad=8)
    axs.set_xlim(0,1)
    #axs.set_ylim(bottom=0)
    # save
    plt.savefig('figs/I_1emax_Spin_1to2,3(alpha).png', dpi=300, bbox_inches='tight')


plt.show()

