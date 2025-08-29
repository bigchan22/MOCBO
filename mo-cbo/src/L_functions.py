import numpy as np
def rast(x):
    return ((x ** 2 - 10 * np.cos(x*2*np.pi) + 10).sum()) / len(x)

# obj is stdev
def get_vol(x, retmean, retcov): # 변동성 (낮을수록 좋음)
    a = np.sqrt(np.dot(x.T, np.dot(retcov * 252, x)))
    if np.isnan(a):
        print(x)
        print(retcov)
        raise ValueError
    return a


# obj is sharpe ratio
def get_neg_sharpe(x, retmean, retcov): # 높을수록 좋음 >> - 추가
    w = x  # simplexproj(x)
    pret = np.sum(retmean * w) * 252
    pvol = np.sqrt(np.dot(w.T, np.dot(retcov * 252, w)))
    return -pret / pvol


def get_wealth(x, rets): # 단기 return
    w = 1.
    # print(x)
    # print(retmeans)
    for ret in rets:
        # print(retmean)
        # print(retmean*x)
        # print(np.sum(retmean * x))
        # raise ValueError
        w = np.sum(ret * x) * w + w
    return -w


def get_logwealth(x, rets):
    # w = 1.
    return -np.sum(np.log(np.matmul(rets , x) + 1)) # -는 minimize 툴에 넣을 거니까.
    # for ret in rets:
    #     # print(retmean)
    #     # print(retmean*x)
    #     # print(np.sum(retmean * x))
    #     # raise ValueError
    #     np.log(np.sum(ret * x) + 1)
    # return -w