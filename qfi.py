import numpy as np
from numpy import array, copy, full, concatenate, vdot, kron, zeros, cos, sin, exp, arange, real, pi, sign, sqrt, sort
from numpy.random import uniform, randint, choice
from numpy.linalg import matrix_rank
from scipy.optimize import minimize
from functools import reduce
from itertools import product
from time import time
from networkx.generators.random_graphs import erdos_renyi_graph
import sys

### QAOA code ###
### Thanks to Akshay Vishwanathan and Luis Ernesto Campos Espinoza! ###

def mixer_list(n):
    def split(x, k):
        return x.reshape((2**k, -1))
    def sym_swap(x):
        return np.asarray([x[-1], x[-2], x[1], x[0]])
    x_list = []
    t1 = np.asarray([arange(2**(n-1), 2**n), arange(0, 2**(n-1))])
    t1 = t1.flatten()
    x_list.append(t1.flatten())
    t2 = t1.reshape(4, -1)
    t3 = sym_swap(t2)
    t1 = t3.flatten()
    x_list.append(t1)
    k = 1
    while k < (n - 1):
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

def apply_Hx(n, x_list, statevector):
    statevector_new = zeros(len(statevector), dtype=complex)
    for i in range(n):
        statevector_swap = statevector[x_list[i]]
        statevector_new = statevector_swap + statevector_new
    return statevector_new

def apply_H(H_diag, statevector):
    return H_diag*statevector

def apply_beta(n, x_list, beta, statevector):
    c = cos(beta)
    s = sin(beta)
    statevector_new = copy(statevector)
    for i in range(n):
        statevector_swap = statevector_new[x_list[i]]
        statevector_new = -1j*s*statevector_swap + c*statevector_new
    return statevector_new

def apply_gamma(H_diag, gamma, statevector):
    return exp(-1j*gamma*H_diag)*statevector

def qaoa_vec(pars, n, x_list, H_diag, state_ini):
    statevector = copy(state_ini)
    p = int(len(pars)/2)
    for j in range(p):
        k = 2*j
        statevector = apply_gamma(H_diag, pars[k], statevector)
        statevector = apply_beta(n, x_list, pars[k+1], statevector)
    return statevector
    

### QFI code ###
### Thanks to Andrey Kardashin! ###

def qaoa_qfi_matrix(pars, n, x_list, H_diag, state_ini):
    
    n_pars = len(pars)
    p = int(n_pars/2)
    QFI_matrix = zeros((n_pars, n_pars))
    
    statevector = copy(state_ini)
    statevectors_der = []
    
    for i in range(p):
        
        n_pars_i = 2*(i+1)
    
        statevector_der_gamma = -1j*apply_H(H_diag, statevector)
        statevector_der_gamma = apply_gamma(H_diag, pars[2*i], statevector_der_gamma)     
        statevector_der_gamma = apply_beta(n, x_list, pars[2*i+1], statevector_der_gamma)
        
        statevector = apply_gamma(H_diag, pars[2*i], statevector)     
        statevector = apply_beta(n, x_list, pars[2*i+1], statevector)
        
        statevector_der_beta = -1j*apply_Hx(n, x_list, statevector) 
        
        statevectors_der.append(statevector_der_gamma)
        statevectors_der.append(statevector_der_beta)
        
        for j in range(n_pars_i - 2):
            statevectors_der[j] = apply_gamma(H_diag, pars[2*i], statevectors_der[j])     
            statevectors_der[j] = apply_beta(n, x_list, pars[2*i+1], statevectors_der[j])     
            
        for a in range(n_pars_i):
            for b in range(n_pars_i - 2, n_pars_i):
                term_1 = vdot(statevectors_der[a], statevectors_der[b])
                term_2 = vdot(statevectors_der[a], statevector)*vdot(statevector, statevectors_der[b])
                QFI_ab = 4*(term_1 - term_2).real
                QFI_matrix[a][b] = QFI_matrix[b][a] = QFI_ab
            
    return QFI_matrix


def find_qaoa_eqd(x_list, n, H_diag, state_ini, tol=None, verbose=True):
    
    p = 1
    QFI_matrix = zeros((2, 2))
    pars = uniform(0, 2*pi, 2)
    max_rank = 0
    
    statevector = copy(state_ini)
    statevectors_der = []

    eqd_found = False
    while eqd_found == False and max_rank <= 2**(n + 1):

        if verbose == True:
            print("p_c: %5d | QFIM rank: %5d" % (p, max_rank), end="\r")
    
        n_pars = 2*p       

        statevector_der_gamma = -1j*apply_H(H_diag, statevector)
        statevector_der_gamma = apply_gamma(H_diag, pars[-2], statevector_der_gamma)     
        statevector_der_gamma = apply_beta(n, x_list, pars[-1], statevector_der_gamma)
        
        statevector = apply_gamma(H_diag, pars[-2], statevector)     
        statevector = apply_beta(n, x_list, pars[-1], statevector)
        
        statevector_der_beta = -1j*apply_Hx(n, x_list, statevector) 
        
        statevectors_der.append(statevector_der_gamma)
        statevectors_der.append(statevector_der_beta)
        
        for j in range(n_pars - 2):
            statevectors_der[j] = apply_gamma(H_diag, pars[-2], statevectors_der[j])     
            statevectors_der[j] = apply_beta(n, x_list, pars[-1], statevectors_der[j])     
            
        for a in range(n_pars):
            for b in range(n_pars - 2, n_pars):
                term_1 = vdot(statevectors_der[a], statevectors_der[b])
                term_2 = vdot(statevectors_der[a], statevector)*vdot(statevector, statevectors_der[b])
                QFI_ab = 4*(term_1 - term_2).real
                QFI_matrix[a][b] = QFI_matrix[b][a] = QFI_ab
            
        rank_p = matrix_rank(QFI_matrix, tol=tol, hermitian=True)
                                    
        if rank_p > max_rank:
            max_rank = rank_p
            QFI_matrix = np.append(QFI_matrix, zeros((n_pars, 2)), axis=1)
            QFI_matrix = np.append(QFI_matrix, zeros((2, n_pars + 2)), axis=0)
            pars = concatenate((pars, uniform(0, 2*pi, 2)))
            p += 1
        else:
            eqd_found = True
            QFI_matrix = QFI_matrix[:-2, :-2]
            p = int(max_rank/2)
            
    if verbose == True:
        print("p_c: %5d | QFIM rank: %5d" % (p, max_rank), end="\r")

    if max_rank > 2**(n + 1):
        print("\nQFIM rank is larger than 2**(n + 1)! Try to change tol.")

    return max_rank


def ring_ham_maxcut_diag(n):
    d = 2**n
    I_diag = array([1, 1])
    Z_diag = array([1, -1])
    H_diag = zeros(d)
    for q in range(n):
        op = [I_diag]*n
        op[q] = Z_diag
        op[(q + 1)%n] = Z_diag
        H_diag = H_diag + reduce(kron, op)
    return H_diag

def edges_to_ham(n, edges):
    d = 2**n
    I_diag = array([1, 1])
    Z_diag = array([1, -1])
    H_diag = zeros(d, dtype=int)
    for edge in edges:
        op = [I_diag]*n
        op[edge[0]] = Z_diag
        op[edge[1]] = Z_diag
        H_diag += reduce(kron, op)
    return H_diag

def gen_ksat_instance(n, m, k):
    clauses = []
    i = 0
    c = 0
    while i < m and c < 500:
        a = sort(choice(arange(1, n + 1), k, replace=False))
        b = choice([-1, 1], k, replace=True)
        clause = list(a*b)
        if clause not in clauses:
            clauses.append(clause)
            i += 1
        c += 1
    if c == 500:
        print(f"Can't generate {m} unique clauses, returned {len(clauses)}!")
    return clauses

def ksat_to_ham_diag(n, clauses):
    d = 2**n
    H_diag = zeros(d)
    for clause in clauses:
        op = [array([1, 1])]*n
        for var in clause:
            if sign(var) == 1:
                op[abs(var) - 1] = array([1, 0])
            else:
                op[abs(var) - 1] = array([0, 1])
        op = reduce(kron, op)
        H_diag += op
    return H_diag


def gen_ksat_instance_seeded(n, m, k, seeds):
    """ seeds must contain at least m naturals"""
    clauses = []
    i = 0
    c = 0
    while i < m and c < 500:
        if c < len(seeds):
            np.random.seed(seeds[c])
        else:
            np.random.seed(None)
        a, b = sort(choice(arange(1, n + 1), k, replace=False)), choice([-1, 1], k, replace=True)
        clause = list(a*b)
        if clause not in clauses:
            clauses.append(clause)
            i += 1
        c += 1
    if c == 500:
        print(f"Can't generate {m} unique clauses, returned {len(clauses)}!")
    return clauses
    
    
def ksat_to_ham_diag(n, clauses):
    d = 2**n
    H_diag = zeros(d)
    for clause in clauses:
        op = [array([1, 1])]*n
        for var in clause:
            if sign(var) == 1:
                op[abs(var)-1] = array([1, 0])
            else:
                op[abs(var)-1] = array([0, 1])
        op = reduce(kron, op)
        H_diag += op
    return H_diag


# def fun(x):#, n, x_list, H_diag, state_ini, fvals, fval_cont):
#     statevector = qaoa_vec(x, n, x_list, H_diag, state_ini)
#     f = vdot(statevector, H_diag*statevector).real
#     fval_cont[0] = f
#     return f

# def callback(x):#, n, x_list, H_diag, state_ini, fvals, fval_cont):
#     fvals.append(fval_cont[0])
#     if len(fvals)%10 == 0:
#         print("\t\tIteration: %d | Cost: %.8f" %(len(fvals), fval_cont[0]), end="\r")
#         data_array = array([n, k, m, p_min, p_max, s, r, H_diag, fvals, funs, errors, nits, nfevs, times, xfs, EQD], dtype=object)
#         np.save(path + file_name, data_array, allow_pickle=True)
#     return None