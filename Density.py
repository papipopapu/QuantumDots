
from sympy import Symbol, Matrix, zeros, eye
from sympy.physics.quantum import TensorProduct as kron
import typing

# NATURAL UNITS!!!!
def get_Lamb_matrix(G_matrix: Matrix) -> Matrix:
    # make sure G_matrix is square
    assert G_matrix.shape[0] == G_matrix.shape[1]
    D = G_matrix.shape[0]
    L_matrix = zeros(D, D)
    
    for m in range(D):
        for n in range(D):
            L_mn = 0
            for k in range(D):
                """ if k != m:
                    L_mn += 1/2 * G_matrix[k, m]
                if k != n:
                    L_mn += 1/2 * G_matrix[k, n] """
                if k != n:
                    L_mn += 1/2 * (G_matrix[k, m] + G_matrix[k, n])
            L_matrix[m, n] = L_mn
            
    return L_matrix

def get_density_equation(G_matrix: Matrix, H_matrix: Matrix) -> Matrix:
    # make sure G_matrix is square and H_matrix is square and same size as G_matrix
    assert G_matrix.shape[0] == G_matrix.shape[1]
    assert H_matrix.shape[0] == H_matrix.shape[1]
    assert G_matrix.shape[0] == H_matrix.shape[0]
    D = G_matrix.shape[0]

    # we return the right side of the equation d/dt rho = L rho
    drho_matrix = zeros(D, D)
    # create and fill rho matrix with symbols
    rho = zeros(D, D)
    for m in range(D):
        for n in range(D):
            rho[m, n] = Symbol('rho_' + str(m) + str(n))
            
    # lets add the H part
    drho_matrix += -1j * (H_matrix * rho - rho * H_matrix)
    
    Lamb_matrix = get_Lamb_matrix(G_matrix)
    for m in range(D):
        for n in range(D):
            # lets add the G part
            if m == n:
                for k in range(D):
                    if k != n:
                        drho_matrix[m, n] += G_matrix[n, k] * rho[k, k] - G_matrix[k, n] * rho[n, n]
            # lets add the Lamb part
            else:
                drho_matrix[m, n] -= Lamb_matrix[m, n] * rho[m, n]
    


    return drho_matrix

def get_density_equation_vect(G_matrix: Matrix, H_matrix: Matrix) -> Matrix:
    
    # make sure G_matrix is square and H_matrix is square and same size as G_matrix
    assert G_matrix.shape[0] == G_matrix.shape[1]
    assert H_matrix.shape[0] == H_matrix.shape[1]
    assert G_matrix.shape[0] == H_matrix.shape[0]
    D = G_matrix.shape[0]

    # we only return L from the equation d/dt vect(rho) = matrix(L) vect(rho)
    L = zeros(D*D, D*D)
        
    
    # lets add the H part, using the kronecker product vectorization identity vec(AXB) = (B^T \otimes A) vec(X)
    I = eye(D)
    L += -1j * (kron(I, H_matrix) - kron(H_matrix.T, I))
    Lamb_matrix = get_Lamb_matrix(G_matrix)
    
    for m in range(D):
        for n in range(D):
            mn_idx = m + n*D # index of rho_mn in vectorized rho, stacked column-wise
            # lets add the G part
            if m == n:
                for k in range(D):
                    if k != n:
                        # rho_mn += A * rho_ij 
                        
                        # first term, rho_mn += G_nk * rho_kk
                        A = G_matrix[n, k]
                        ij_idx = k + k*D
                        L[mn_idx, ij_idx] += A
                        
                        # second term, rho_mn -= G_kn * rho_nn
                        A = G_matrix[k, n]
                        ij_idx = n + n*D
                        L[mn_idx, ij_idx] -= A
            # lets add the Lamb part
            else:
                # rho_mn += A * rho_ij 
                
                # only term, rho_mn -= L_mn * rho_mn
                A = Lamb_matrix[m, n]
                ij_idx = mn_idx
                L[mn_idx, ij_idx] -= A
        
    return L
    
    
    

