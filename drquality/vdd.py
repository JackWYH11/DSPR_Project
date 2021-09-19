# from drquality import *
import time

from drquality.utils import *


def run(Data, Projection, K):
    t = time.time()
    dist2datan, datadistOK, datadistOKIndex = dist2N(Data, K)
    oo = time.time() - t
    # print("VDD step1 time:", oo)

    tt = time.time()
    dist2projectn, projdistOK = dist2AtIndices(Projection, datadistOKIndex)
    kk = time.time() - tt
    # print("VDD step2 time:", kk)

    ff = time.time()
    tmp1 = np.sqrt(np.square(np.mat(datadistOK)[:, 1:K + 1]).sum(axis=1))
    tmp2 = np.sqrt(np.square(np.mat(projdistOK)[:, 1:K + 1]).sum(axis=1))
    normdatadistOK = np.tile(tmp1, (1, K))
    normprojdistOK = np.tile(tmp2, (1, K))

    firstterm = np.mat(datadistOK)[:, 1:K + 1] / normdatadistOK
    firstterm[np.isnan(firstterm)] = 0

    secondterm = np.mat(projdistOK)[:, 1:K + 1] / normprojdistOK
    secondterm[np.isnan(secondterm)] = 0

    res = ((np.sqrt(np.square(firstterm - secondterm).sum(axis=1))).sum(axis=0) / len(Data))[0, 0]
    pp = time.time() - ff
    # print("VDD step3 time:", pp)

    return float(res)


if __name__ == "__main__":
    data = [[3, 2, 1], [4, 5, 6], [9, 8, 7], [10, 11, 12]]
    pjt = [[3, 1], [4, 6], [9, 7], [10, 12]]
    VDD = run(data, pjt, 2)
    print("VDD:", VDD)
