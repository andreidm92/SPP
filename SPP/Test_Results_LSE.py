import numpy as np
from TestFunctions import LogSumExp_difalpha as LogSumExp
from TestFunctions.TestFunctions import TestFunction
from Solvers.GradientMethods import FGM_internal
import comparison


from comparison import method_comparison, create_methods_dict


def get_LSE(N=100, M=2, alpha=0.001, beta=0.5e-3, k=100, seed=241):
    """
    The function is to get class of LSE. All parameters mathes the
    parameters above.

    N is dimensional of primal problem
    M is number of constraints. It is also the dual dimension
    alpha is parameter of LogSumExp problem
    beta is coefficient of l2-regularization for primal problem

    k is parameter for to generate matrix B
    """
    np.random.seed(M * N)
    alpha_ = np.random.uniform(size=(N,), low=-alpha, high=alpha)
    alpha = np.max(np.abs(alpha_))
    B = []
    for i in range(M):
        b = np.random.uniform(low=-k, high=0, size=(N,)) * np.sign(alpha_)
        B.append(b)
    B = np.array(B)
    # B = np.random.uniform(low = -k, high = 0, size = (M, N))
    c = np.ones((M,))
    x_ = np.zeros((N,))
    f_ = LogSumExp.r(alpha, beta).get_value(x_)
    gamma = c.max()
    y_max = f_ / gamma
    Q = [[0, y_max] for i in range(M)]
    Q = np.array(Q)
    size_domain = np.linalg.norm(Q[:, 0] - Q[:, 1])
    LSE = TestFunction(r=LogSumExp.h(c, size_domain=size_domain),
                       F=LogSumExp.F_tilde(B, size_domain),
                       h=LogSumExp.r(alpha_, beta, size_domain),
                       solver=FGM_internal,
                       get_start_point=lambda x: (1 / beta * -x.dot(B), alpha * np.sqrt(N) / beta))
    return LSE, Q, alpha_, B, c, beta

def parse_results(history, alpha_, B, c, beta, keys = ["Ellipsoids", "Dichotomy", "FGM"]):
    s_key = ""
    s_feas = ""
    s_time = ""
    s_value = ""
    for key in keys:
        s_key += key + "\t$"
        lambda_, x = history[key][-1][0]
        s_feas += str((B @ x <= c).any()) + "\t\t$"
        s_time += "%.3f"%(history[key][-1][1] - history[key][0][1]) + "\t\t$"
        s_value+= "%.3f"%(np.log(1 + np.exp(alpha_*x).sum()) + beta * np.linalg.norm(x)**2) + "\t\t$"
    print(s_key)
    print(s_feas)
    print(s_time)
    print(s_value)

M = 2
N = 100

eps = 1e-3

LSE, Q, alpha_, B, c, beta = get_LSE(N, M)
history = {}
methods = create_methods_dict(LSE, np.mean(Q, axis = 1),
                              np.linalg.norm(Q[:,0]-Q[:,1]),
                              Q, eps, history, time_max = 20,
                             stop_cond_args = [alpha_, B, c, beta])
comparison.method_comparison(methods)
parse_results(history, alpha_, B, c, beta)


