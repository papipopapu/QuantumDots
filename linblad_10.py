from qutipDots import *
import qutip as qt
from numpy import exp
import matplotlib.pyplot as plt
import scienceplots
from qutip import fcreate, fdestroy, qeye, tensor, sigmaz
from tqdm import tqdm
from scipy.linalg import expm
import numba as nb
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

t_12_0 = tau
t_23_0 = tau
t_31_0 = tau

t_12_sf = tau
t_23_sf = tau
t_31_sf = tau


N = 0
taus = np.linspace(0, 10, N)
alphas = np.linspace(0, 2*np.pi, N)

@nb.njit()
def build_H(t_12_0, t_23_0, t_31_0, t_12_sf, t_23_sf, t_31_sf, a_12, a_23, a_31):
    H2 = np.zeros((6,6), dtype=np.complex128)
    H2[0,0] = e1
    H2[0,1] = 0
    H2[0,2] = -t_12_0
    H2[0,3] = -t_12_sf * np.exp(-1j*a_12)
    H2[0,4] = -t_31_0
    H2[0,5] = t_31_sf * np.exp(-1j*a_31)
    H2[1,0] = 0
    H2[1,1] = e1
    H2[1,2] = t_12_sf * np.exp(1j*a_12)
    H2[1,3] = -t_12_0
    H2[1,4] = -t_31_sf * np.exp(1j*a_31)
    H2[1,5] = -t_31_0
    H2[2,0] = -t_12_0
    H2[2,1] = t_12_sf * np.exp(-1j*a_12)
    H2[2,4] = -t_23_0
    H2[2,5] = -t_23_sf * np.exp(-1j*a_23)
    H2[3,0] = -t_12_sf * np.exp(1j*a_12)
    H2[3,1] = -t_12_0
    H2[3,2] = 0
    H2[3,3] = e2
    H2[3,4] = t_23_sf * np.exp(1j*a_23)
    H2[3,5] = -t_23_0
    H2[4,0] = -t_31_0
    H2[4,1] = -t_31_sf * np.exp(-1j*a_31)
    H2[4,2] = -t_23_0
    H2[4,3] = t_23_sf * np.exp(-1j*a_23)
    H2[4,4] = e3
    H2[4,5] = 0
    H2[5,0] = t_31_sf * np.exp(1j*a_31)
    H2[5,1] = -t_31_0
    H2[5,2] = -t_23_sf * np.exp(1j*a_23)
    H2[5,3] = -t_23_0
    H2[5,4] = 0
    H2[5,5] = e3
    
    return H2

@nb.njit()
def main(N):
    taus = np.linspace(5, 1, N)
    alphas = np.linspace(0, 2*np.pi, N)
    ps = np.zeros(18)
    ret = np.zeros(11)
    bruh = np.zeros(1, dtype=np.int64)
    idx = 0
    tol = 1e-2
    for t_12_0 in taus:
        for t_23_0 in taus:
            for t_31_0 in taus:
                for t_12_sf in taus:
                    for t_23_sf in taus:
                        for t_31_sf in taus:
                            for a_12 in alphas:
                                for a_23 in alphas:
                                    for a_31 in alphas:
                                            
                                        """ H2 = np.array([
                                        [e1, 0, -t_12_0, -t_12_sf*exp(-1j*a_12), -t_31_0, -t_31_sf*exp(-1j*a_31)],
                                        [0, e1, t_12_sf*exp(1j*a_12), -t_12_0, -t_31_sf*exp(1j*a_31), -t_31_0],
                                        [-t_12_0, t_12_sf*exp(-1j*a_12), e2, 0, -t_23_0, -t_23_sf*exp(-1j*a_23)],
                                        [-t_12_sf*exp(1j*a_12), -t_12_0, 0, e2, t_23_sf*exp(1j*a_23), -t_23_0],
                                        [-t_31_0, -t_31_sf*exp(-1j*a_31), -t_23_0, t_23_sf*exp(-1j*a_23), e3, 0],
                                        [-t_31_sf*exp(1j*a_31), -t_31_0, -t_23_sf*exp(1j*a_23), -t_23_0, 0, e3]
                                        ]) """
                                        
                                        
                                        H = build_H(t_12_0, t_23_0, t_31_0, t_12_sf, t_23_sf, t_31_sf, a_12, a_23, a_31)
                                                    
                                                                        
                                        vals, vecs = np.linalg.eigh(H)
                                        v1 = vecs[:,0]
                                        ps[0] = np.abs(v1[0])**2 + np.abs(v1[1])**2                                        
                                        ps[1] = np.abs(v1[2])**2 + np.abs(v1[3])**2
                                        ps[2] = np.abs(v1[4])**2 + np.abs(v1[5])**2
                                                                 
                                        v2 = vecs[:,1]
                                        ps[3] = np.abs(v2[0])**2 + np.abs(v2[1])**2
                                        ps[4] = np.abs(v2[2])**2 + np.abs(v2[3])**2
                                        ps[5] = np.abs(v2[4])**2 + np.abs(v2[5])**2
                                                               
                                        v3 = vecs[:,2]
                                        ps[6] = np.abs(v3[0])**2 + np.abs(v3[1])**2
                                        ps[7] = np.abs(v3[2])**2 + np.abs(v3[3])**2
                                        ps[8] = np.abs(v3[4])**2 + np.abs(v3[5])**2
                                        
                                        v4 = vecs[:,3]
                                        ps[9] = np.abs(v4[0])**2 + np.abs(v4[1])**2
                                        ps[10] = np.abs(v4[2])**2 + np.abs(v4[3])**2
                                        ps[11] = np.abs(v4[4])**2 + np.abs(v4[5])**2
                                        
                                        v5 = vecs[:,4]
                                        ps[12] = np.abs(v5[0])**2 + np.abs(v5[1])**2
                                        ps[13] = np.abs(v5[2])**2 + np.abs(v5[3])**2
                                        ps[14] = np.abs(v5[4])**2 + np.abs(v5[5])**2
                                        
                                        v6 = vecs[:,5]
                                        ps[15] = np.abs(v6[0])**2 + np.abs(v6[1])**2
                                        ps[16] = np.abs(v6[2])**2 + np.abs(v6[3])**2
                                        ps[17] = np.abs(v6[4])**2 + np.abs(v6[5])**2
                                        
                                        if np.any(ps < tol):
                                            ret[0] = np.argmin(ps)
                                            bruh[0] = ret[0]
                                            ret[1] = ps[bruh[0]]
                                            ret[2] = t_12_0
                                            ret[3] = t_23_0
                                            ret[4] = t_31_0
                                            ret[5] = t_12_sf
                                            ret[6] = t_23_sf
                                            ret[7] = t_31_sf
                                            ret[8] = a_12
                                            ret[9] = a_23
                                            ret[10] = a_31
                                            return ret                                         
                                        
                                        
                                        
                                        
                                        idx += 1
    return ret
                                        





""" H2 = np.array([
[e1, 0, -t_12_0, -t_31_sf*exp(-1j*a_12), -t_12_0, t_31_sf*exp(-1j*a_31)],
[0, e1, t_31_sf*exp(1j*a_12), -t_12_0, -t_31_sf*exp(1j*a_31), -t_12_0],
[-t_12_0, t_31_sf*exp(-1j*a_12), e2, 0, -t_23_0, -t_31_sf*exp(-1j*a_23)],
[-t_31_sf*exp(1j*a_12), -t_12_0, 0, e2, t_31_sf*exp(1j*a_23), -t_23_0],
[-t_12_0, -t_31_sf*exp(-1j*a_31), -t_23_0, t_31_sf*exp(-1j*a_23), e3, 0],
[t_31_sf*exp(1j*a_31), -t_12_0, -t_31_sf*exp(1j*a_23), -t_23_0, 0, e3]
]) """
e1 = 0
e2 = 0
e3 = 0
tau = 1#sqrt3

tau_0 = tau
tau_sf = tau *3 / np.sqrt(3)
a_12 = 0
a_23 = 0
a_31 = 0

t_12_0 = tau_0
t_23_0 = tau_0
t_31_0 = tau_0

t_12_sf = tau_sf
t_23_sf = tau_sf
t_31_sf = tau_sf


H2 = np.array([
[e1, 0, -t_12_0, -t_12_sf*exp(-1j*a_12), -t_31_0, t_31_sf*exp(-1j*a_31)],
[0, e1, t_12_sf*exp(1j*a_12), -t_12_0, -t_31_sf*exp(1j*a_31), -t_31_0],
[-t_12_0, t_12_sf*exp(-1j*a_12), e2, 0, -t_23_0, -t_23_sf*exp(-1j*a_23)],
[-t_12_sf*exp(1j*a_12), -t_12_0, 0, e2, t_23_sf*exp(1j*a_23), -t_23_0],
[-t_31_0, -t_31_sf*exp(-1j*a_31), -t_23_0, t_23_sf*exp(-1j*a_23), e3, 0],
[t_31_sf*exp(1j*a_31), -t_31_0, -t_23_sf*exp(1j*a_23), -t_23_0, 0, e3]
])
sy = np.array([[0, -1j], [1j, 0]]) / 2 # sigma_y 
Sy = np.kron(np.eye(3, 3), sy)
U = expm(-1j * np.pi * Sy)
Hu = U.T @ H2 @ U
print(Hu-np.conj(H2))

evals, evecs = np.linalg.eigh(H2)
# normalize evecs
for i in range(6):
    evecs[:,i] = evecs[:,i] / np.linalg.norm(evecs[:,i])
print(evals)
for e in evecs.T:
    print(e)
# now without spin

# add a row of zeros on top and a column of zeros on the left
H = np.zeros((7,7), dtype=np.complex128)
H[1:,1:] = H2



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


I = n3u + n3d #+ n2u + n2d
# I/Gamma




# collapse operators
into1 = np.sqrt(g1) * c1u_ + np.sqrt(g1) * c1d_
#out2 = np.sqrt(g2) * c2u + np.sqrt(g2) * c2d
out3 = np.sqrt(g3) * c3u + np.sqrt(g3) * c3d

# to Qobj
into1 = qt.Qobj(into1)
out3 = qt.Qobj(out3)
#out2 = qt.Qobj(out2)
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


# solve steady state
H = qt.Qobj(H)
rho_ss = qt.steadystate(H, [into1, out3])
Is = qt.expect(I, rho_ss)
print(rho_ss)