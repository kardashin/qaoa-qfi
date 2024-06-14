import numpy as np
from numpy import array, copy, full, concatenate, dot, vdot, kron, outer, eye, zeros, ones, diag, sqrt, log2, cos, sin, exp, trace, shape, reshape, arange, linspace, real, pi, mean, median, std
from numpy.random import shuffle, normal, uniform, randint
from scipy.stats import gaussian_kde, unitary_group
from numpy.linalg import matrix_rank, inv, pinv, eig, svd
from scipy.linalg import expm, eigh, norm
from scipy.optimize import minimize, differential_evolution
from math import ceil
from functools import reduce, partial
from itertools import product
from random import sample


X = array([[0.,1.],
           [1.,0.]]) # X Pauli matrix
Y = array([[0.,-1.j],
           [1.j, 0.]]) # Y Pauli matrix
Z = array([[1., 0.],
           [0.,-1.]]) # Z Pauli matrix
I = array([[1.,0.],
           [0.,1.]]) # 2x2 identity matrix


def Hx_gen(n_qubits):
    Hx = zeros((2**n_qubits, 2**n_qubits))
    for q in range(n_qubits):
        X_op = [I]*q + [X] + [I]*(n_qubits-q-1)
        Hx = Hx + reduce(kron, X_op)
    return Hx

def Hz_ising_gen(n_qubits, bc="closed"):
    Hzz = zeros((2**n_qubits, 2**n_qubits))
    for q in range(n_qubits-1):
        Hzz = Hzz + reduce(kron, [I]*q + [Z, Z] + [I]*(n_qubits-q-2))
    if bc == "closed":
        Hzz = Hzz + reduce(kron, [Z] + [I]*(n_qubits-2) + [Z])
    return Hzz    


### for this code, credits go to Ernesto Luis Campos Espinoza and Akshay Vishwanathan! ###

def mixer_list(n_qubits):
    def split(x, k):
        return x.reshape((2**k, -1))
    def sym_swap(x):
        return np.asarray([x[-1], x[-2], x[1], x[0]])
    x_list = []
    t1 = np.asarray([arange(2**(n_qubits-1), 2**n_qubits), arange(0, 2**(n_qubits-1))])
    t1 = t1.flatten()
    x_list.append(t1.flatten())
    t2 = t1.reshape(4, -1)
    t3 = sym_swap(t2)
    t1 = t3.flatten()
    x_list.append(t1)
    k = 1
    while k < (n_qubits - 1):
        t2 = split(t1, k)
        t2 = np.asarray(t2)
        t1 = []
        for y in t2:
            t3 = y.reshape((4, -1))
            t4 = sym_swap(t3)
            t1.append(t4.flatten())
        t1 = np.asarray(t1)
        t1 = t1.flatten()
        x_list.append(t1)
        k += 1        
    return x_list

def apply_Hx(n_qubits, x_list, statevector):
    statevector_new = zeros(len(statevector), dtype=complex)
    for i in range(n_qubits):
        statevector_swap = statevector[x_list[i]]
        statevector_new = statevector_swap + statevector_new
    return statevector_new

def apply_H(H_diag, statevector):
    return H_diag*statevector

def apply_beta(n_qubits, x_list, beta, statevector):
    c = cos(beta)
    s = sin(beta)
    statevector_new = copy(statevector)
    for i in range(n_qubits):
        statevector_swap = statevector_new[x_list[i]]
        statevector_new = -1j*s*statevector_swap + c*statevector_new
    return statevector_new

def apply_gamma(H_diag, gamma, statevector):
    return exp(-1j*gamma*H_diag)*statevector


def qaoa_qfi_matrix(pars, n_qubits, x_list, H_diag, state_ini):
    
    n_pars = len(pars)
    p = int(n_pars/2)
    QFI_matrix = zeros((n_pars, n_pars))
    
    statevector = copy(state_ini)
    statevectors_der = []
    
    for i in range(p):
        
        n_pars_i = 2*(i+1)
    
        statevector_der_gamma = -1j*apply_H(H_diag, statevector)
        statevector_der_gamma = apply_gamma(H_diag, pars[2*i], statevector_der_gamma)     
        statevector_der_gamma = apply_beta(n_qubits, x_list, pars[2*i+1], statevector_der_gamma)
        
        statevector = apply_gamma(H_diag, pars[2*i], statevector)     
        statevector = apply_beta(n_qubits, x_list, pars[2*i+1], statevector)
        
        statevector_der_beta = -1j*apply_Hx(n_qubits, x_list, statevector) 
        
        statevectors_der.append(statevector_der_gamma)
        statevectors_der.append(statevector_der_beta)
        
        for j in range(n_pars_i - 2):
            statevectors_der[j] = apply_gamma(H_diag, pars[2*i], statevectors_der[j])     
            statevectors_der[j] = apply_beta(n_qubits, x_list, pars[2*i+1], statevectors_der[j])     
            
        for a in range(n_pars_i):
            for b in range(n_pars_i-2, n_pars_i):
                term_1 = vdot(statevectors_der[a], statevectors_der[b])
                term_2 = vdot(statevectors_der[a], statevector)*vdot(statevector, statevectors_der[b])
                QFI_ab = 4*real(term_1 - term_2)
                QFI_matrix[a][b] = QFI_matrix[b][a] = QFI_ab
            
    return QFI_matrix



def find_qaoa_eqd(x_list, n_qubits, H_diag, state_ini, tol=None, verbose=True):
    
    p = 1
    QFI_matrix = zeros((2, 2))
    pars = uniform(0, 2*pi, 2)
    max_rank = 0
    
    statevector = copy(state_ini)
    statevectors_der = []
    
    eqd_found = False
    while eqd_found == False:

        if verbose == True:
            print("p: %5d | EQD: %5d" % (p, max_rank), end="\r")
    
        n_pars = 2*p       

        statevector_der_gamma = -1j*apply_H(H_diag, statevector)
        statevector_der_gamma = apply_gamma(H_diag, pars[-2], statevector_der_gamma)     
        statevector_der_gamma = apply_beta(n_qubits, x_list, pars[-1], statevector_der_gamma)
        
        statevector = apply_gamma(H_diag, pars[-2], statevector)     
        statevector = apply_beta(n_qubits, x_list, pars[-1], statevector)
        
        statevector_der_beta = -1j*apply_Hx(n_qubits, x_list, statevector) 
        
        statevectors_der.append(statevector_der_gamma)
        statevectors_der.append(statevector_der_beta)
        
        for j in range(n_pars - 2):
            statevectors_der[j] = apply_gamma(H_diag, pars[-2], statevectors_der[j])     
            statevectors_der[j] = apply_beta(n_qubits, x_list, pars[-1], statevectors_der[j])     
            
        for a in range(n_pars):
            for b in range(n_pars-2, n_pars):
                term_1 = vdot(statevectors_der[a], statevectors_der[b])
                term_2 = vdot(statevectors_der[a], statevector)*vdot(statevector, statevectors_der[b])
                QFI_ab = 4*real(term_1 - term_2)
                QFI_matrix[a][b] = QFI_matrix[b][a] = QFI_ab
            
        rank_p = matrix_rank(QFI_matrix, tol=tol, hermitian=True)
                                    
        if rank_p > max_rank:
            max_rank = rank_p
            QFI_matrix = np.append(QFI_matrix, zeros((n_pars, 2)), axis=1)
            QFI_matrix = np.append(QFI_matrix, zeros((2, n_pars+2)), axis=0)
            pars = concatenate((pars, uniform(0, 2*pi, 2)))
            p += 1
        else:
            eqd_found = True
            QFI_matrix = QFI_matrix[:-2, :-2]
            p = ceil(max_rank/2)
            
    if verbose == True:
        print("p: %5d | EQD: %5d" % (p, max_rank), end="\r")

    return max_rank
