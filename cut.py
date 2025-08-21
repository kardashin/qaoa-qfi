from qfi import *

def fun(x, n, x_list, H_diag, state_ini, fvals, fval_cont):
    statevector = qaoa_vec(x, n, x_list, H_diag, state_ini)
    f = vdot(statevector, H_diag*statevector).real
    fval_cont[0] = f
    return f

def callback(x):#, n, x_list, H_diag, state_ini, fvals, fval_cont):
    fvals.append(fval_cont[0])
    if len(fvals)%10 == 0:
        print("\t\tIteration: %d | Cost: %.8f" %(len(fvals), fval_cont[0]), end="\r")
        data_array = array([n, prob, p_min, p_max, s, r, edges, H_diag, fvals, funs, errors, nits, nfevs, times, xfs, EQD], dtype=object)
        np.save(path + file_name, data_array, allow_pickle=True)
    return None

if __name__ == '__main__':

    n = int(sys.argv[1])
    prob = float(sys.argv[2])
    p_min = int(sys.argv[3])
    p_max = int(sys.argv[4])
    s = int(sys.argv[5])
    r = int(sys.argv[6])
    path = sys.argv[7]
    
    d = 2**n
    x_list = mixer_list(n)
    state_ini = full(d, 1/sqrt(d))

    graph = erdos_renyi_graph(n, prob, seed=int(s + 1000*prob)) # lame seed, but ensures that the graphs with lower prob are not subgraphs for the graphs with higher prob  
    edges = list(graph.edges())
    H_diag = edges_to_ham(n, edges)
    cost_min_exact = min(H_diag)
    EQD = find_qaoa_eqd(x_list, n, H_diag, state_ini, tol=None, verbose=True)
    
    np.random.seed(s + 98765*r) # lame, but works: want to run the same graph with different initials
    x0_blank = uniform(0, 2*pi, 2*(p_max - p_min + 1)).reshape((p_max - p_min + 1), 2)

    funs = zeros(p_max - p_min + 1)
    errors = zeros(p_max - p_min + 1)
    nits = zeros(p_max - p_min + 1)
    nfevs = zeros(p_max - p_min + 1)
    xfs = [zeros(2*p) for p in range(p_min, p_max + 1)]
    times = zeros(p_max - p_min + 1)

    if p_min == 1:
        xf = array([])
    else:
        xf = concatenate(x0_blank[:p_min - 1])

    file_name = "maxcut-random_graphs-n=%d-prob=%.2f-p=(%d,%d)-s=%d-r=%d-prevsol" %(n, prob, p_min, p_max, s, r)
#    print(file_name)
    
    fvals = []
    fval_cont = [None]

    args = (n, x_list, H_diag, state_ini, fvals, fval_cont)
    
    c = 0
    for p in range(p_min, p_max + 1):
        print("\tp:", p)
        
        x0 = concatenate([xf, x0_blank[c]])
        time_start = time()
        result = minimize(x0=x0, fun=fun, callback=callback, method="BFGS", args=args)
        time_finish = time() - time_start
        xf = result.x
        error = abs(cost_min_exact - result.fun)

        print("\n\t\tFinished in %d seconds" %time_finish)
        print("\t\tError: %.9f" %error)
        
        funs[c:] = result.fun
        errors[c:] = error
        nits[c] = result.nit
        nfevs[c] = result.nfev
        times[c] = time_finish
        xfs[c] = result.x

        if abs(cost_min_exact - result.fun) <= 1e-8:
            print("Converged!")
            break
        else:
            c += 1

    data_array = array([n, prob, p_min, p_max, s, r, edges, H_diag, array(fvals), funs, errors, nits, nfevs, times, xfs, EQD], dtype=object)
    np.save(path + file_name, data_array, allow_pickle=True)
