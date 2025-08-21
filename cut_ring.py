from qfi import *

def fun(x, n, x_list, H_diag, state_ini, fvals, fval_cont):
    statevector = qaoa_vec(x, n, x_list, H_diag, state_ini)
    f = vdot(statevector, H_diag*statevector).real
    fval_cont[0] = f
    return f


def callback(x):
    fvals.append(fval_cont[0])
    if len(fvals)%10 == 0:
        print("\t\tIteration: %d | Cost: %.8f" %(len(fvals), fval_cont[0]), end="\r")
        data_array = array([n, p_min, p_max, r, array(fvals), array(funs), array(errors), array(nits), array(nfevs), array(times), xfs], dtype=object)
        np.save(path + file_name, data_array, allow_pickle=True)
    return None


if __name__ == '__main__':

    n = int(sys.argv[1])
    p_min = int(sys.argv[2])
    p_max = int(sys.argv[3])
    r = int(sys.argv[4])
    path = sys.argv[5]
    
    d = 2**n
    x_list = mixer_list(n)
    state_ini = full(d, 1/sqrt(d))

    H_diag = ring_ham_maxcut_diag(n)
    cost_min_exact = min(H_diag)
    
    np.random.seed(r)
    x0_blank = uniform(0, 2*pi, 2*(p_max - p_min + 1)).reshape((p_max - p_min + 1), 2)
    print(x0_blank)

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

    file_name = "ring_maxcut-n=%d-p=(%d,%d)-r=%d" %(n, p_min, p_max, r)
    
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

        print("\t\tFinished in %d seconds" %time_finish)
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

    data_array = array([n, p_min, p_max, r, array(fvals), array(funs), array(errors), array(nits), array(nfevs), array(times), xfs], dtype=object)
    np.save(path + file_name, data_array, allow_pickle=True)
