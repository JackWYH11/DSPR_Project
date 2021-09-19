import time

import numpy as np

from drquality.utils import dist2N


def MRREdata(Data, Projection, K):
    t = time.time()
    projected_d2, projected_d2N, projected_d2index = dist2N(Projection, K)
    oo = time.time() - t
    # print("MRREdata step1 time:", oo)

    tt = time.time()
    original_d2, original_d2N, original_d2index = dist2N(Data)
    kk = time.time() - tt
    # print("MRREdata step2 time:", kk)

    ff = time.time()
    ranks_latent = np.zeros((len(Data), K))
    ranks_data = np.zeros((len(Data), K))
    for i in range(len(Data)):
        for j in range(1, K + 1):
            ranks_latent[i, j - 1] = j - 1
            ranks_data[i, j - 1] = original_d2index[i].index(projected_d2index[i][j]) - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        ranks = np.abs(ranks_latent - ranks_data) / ranks_data
    ranks[np.isinf(ranks)] = 0
    ranks[np.isnan(ranks)] = 0
    pp = time.time() - ff
    # print("MRREdata step3 time:", pp)

    N = len(Data)
    sumnorm = 0
    for t in range(K):
        sumnorm += abs(2 * K - N - 1) / K
    Ak = 1 / (N * sumnorm)
    C = Ak * ranks.sum()
    return C


def MRRElatent(Data, Projection, K):
    t = time.time()
    projected_d2, projected_d2N, projected_d2index = dist2N(Projection)
    oo = time.time() - t
    # print("MRRElatent step1 time:", oo)

    tt = time.time()
    original_d2, original_d2N, original_d2index = dist2N(Data, K)
    kk = time.time() - tt
    # print("MRRElatent step2 time:", kk)

    ff = time.time()
    ranks_latent = np.zeros((len(Data), K))
    ranks_data = np.zeros((len(Data), K))
    for i in range(len(Data)):
        for j in range(1, K + 1):
            ranks_data[i, j - 1] = j - 1
            ranks_latent[i, j - 1] = projected_d2index[i].index(original_d2index[i][j]) - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        ranks = np.abs(ranks_data - ranks_latent) / ranks_latent
    ranks[np.isinf(ranks)] = 0
    ranks[np.isnan(ranks)] = 0
    pp = time.time() - ff
    # print("MRRElatent step3 time:", pp)

    N = len(Data)
    sumnorm = 0
    for t in range(K):
        sumnorm += abs(2 * K - N - 1) / K
    Ak = 1 / (N * sumnorm)
    C = Ak * ranks.sum()
    return C


if __name__ == '__main__':
    data = [[3, 2, 1], [4, 5, 6], [9, 8, 7], [10, 11, 12]]
    pjt = [[4, 1], [5, 6], [6, 7], [10, 11]]
    k = 3
    print(MRREdata(data, pjt, k))
    print(MRRElatent(data, pjt, k))
