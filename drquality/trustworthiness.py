import time

import numpy as np

from drquality.utils import dist2N


def run(Data, Projection, K):
    t = time.time()
    projected_d2, projected_d2N, projected_d2index = dist2N(Projection, K)
    oo = time.time() - t
    # print("Trustworthiness step1 time:", oo)

    tt = time.time()
    original_d2, original_d2N, original_d2index = dist2N(Data)
    kk = time.time() - tt
    # print("Trustworthiness step2 time:", kk)

    ff = time.time()
    ranks = np.zeros((len(Data), K))
    for i in range(len(Data)):
        for j in range(1, K + 1):
            results = original_d2index[i].index(projected_d2index[i][j]) - (K + 1)
            ranks[i, j - 1] = results if results >= 0 else 0
    pp = time.time() - ff
    # print("Trustworthiness step3 time:", pp)

    N = len(Data)
    Ak = 2 / (N * K * (2 * N - 3 * K - 1))
    T = 1 - Ak * ranks.sum()
    elapsedtime = [oo, kk, pp]
    return T


if __name__ == '__main__':
    data = [[3, 2, 1], [4, 5, 6], [9, 8, 7], [10, 11, 12]]
    pjt = [[4, 1], [5, 6], [6, 7], [10, 11]]
    k = 3
    print(run(data, pjt, k))
