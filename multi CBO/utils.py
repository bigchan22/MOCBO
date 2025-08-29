import numpy as np
import scipy.optimize as sco

def Lmin(x, L):
    Lmin = 1e6
    for i in range(len(x)):
        if (Lmin > L(x[i])):
            Lmin = L(x[i])
    return Lmin, x[i]


def simplex_proj(x):
    b = 1.
    y = np.copy(x)
    u = sorted(y) # 작은 수부터 순서 정렬.
    n = len(y)
    i = n - 1
    cumsum = u[i]
    t = (cumsum - b) / (n - i)
    while (t < u[i - 1]):
        i = i - 1
        cumsum += u[i]
        t = (cumsum - b) / (n - i)
        if i == 0:
            break
    y = y - t
    y[y < 0] = 0
    return y

def get_opts(L, D):
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(D)) # (0,1) 을 D개만큼 복사.

    opts = sco.minimize(L, D * [1. / D], method='SLSQP',
                        bounds=bnds, constraints=cons)
    #     lambda x : L(x,retmean,retcov)
    #     x0 = opts.x
    #     srvol = np.sqrt(np.dot(x0.T, np.dot(retcov * 252, x0)))
    #     srmean = np.sum(retmean * x0) * 252
    return opts

def avg_L(x, L):
    N = len(x)
    value_L = np.zeros(N)
    for i in range(N):
        value_L[i] = L(x[i])
    return np.sum(value_L) / N
def avg_x(x):
    return np.sum(x, axis=0) / len(x)