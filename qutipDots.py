import numpy as np
from qutip import tensor, basis, sigmaz, destroy, identity
from typing import List
from numpy.typing import ArrayLike



def f_destroy(N: int, i: int):
    # evil tensor magic
    return tensor([sigmaz()] * i + [destroy(2)] + [identity(2)] * (N - i - 1))

def f_create(N: int, i: int):
    return f_destroy(N, i).dag()

def eqdot_state(occupations: List[bool]):
    N = len(occupations)
    return tensor([basis(2, occupations[i]) for i in range(N)])

def get_Lambda(Gamma: ArrayLike) -> ArrayLike:
    # do checks outside man
    D = Gamma.shape[0]
    Lambda = np.zeros((D, D))
    
    for m in range(D):
        for n in range(D):
            Lambda_mn = 0
            for k in range(D):
                if k != n:
                    Lambda_mn += 1/2 * (Gamma[k, m] + Gamma[k, n])
            Lambda[m, n] = Lambda_mn
    
    return Lambda

def get_Liouville(Gamma: ArrayLike, H: ArrayLike) -> ArrayLike:
    D = Gamma.shape[0]
    Liouville = np.zeros((D*D, D*D), dtype=np.complex128)
    I = np.eye(D)
    Liouville += -1j * (np.kron(I, H) - np.kron(H.T, I))
    Lambda = get_Lambda(Gamma)
    
    for m in range(D):
        for n in range(D):
            mn_idx = m + n*D
            if m == n:
                for k in range(D):
                    if k != n:
                        Liouville[mn_idx, k + k*D] += Gamma[n, k]
                        Liouville[mn_idx, n + n*D] -= Gamma[k, n]
            else:
                Liouville[mn_idx, mn_idx] -= Lambda[m, n]
        
    return Liouville
    
    
def red_H_idx(H, allowed_idx):
    N = len(allowed_idx)
    H_red = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            H_red[i, j] = H[allowed_idx[i], allowed_idx[j]] 
    return H_red


def red_H(H, states):
    N = len(states)
    H_red = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            H_red[i, j] = (states[i].dag() * H * states[j]).full(squeeze=True)
    return H_red