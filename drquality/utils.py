import sys
import numpy as np


def dist2N(x, k=None):
    ndata, dimx = len(x), len(x[0])
    ncentres, dimc = len(x), len(x[0])
    if dimx != dimc:
        print('Data dimension does not match dimension of centres')
        sys.exit(0)
    now_x = np.mat(x)
    x1 = (np.ones((ncentres, 1)) * np.square(now_x).T.conjugate().sum(axis=0)).T.conjugate()
    x2 = np.ones((ndata, 1)) * np.square(now_x).T.conjugate().sum(axis=0)
    x3 = np.multiply(2, (now_x * now_x.T.conjugate()))
    n2 = (x1 + x2 - x3)
    n2[np.less(n2, 0)] = 0
    if k:
        tempn2, tempindsortn2 = np.sort(n2, axis=1), np.argsort(n2, axis=1)
        sortn2 = tempn2[:, 0: k+1]
        indsortn2 = tempindsortn2[:, 0: k+1]
        # print(sortn2, indsortn2)
    else:
        sortn2, indsortn2 = np.sort(n2, axis=1), np.argsort(n2, axis=1)
        # print(sortn2, indsortn2)

    return n2.tolist(), sortn2.tolist(), indsortn2.tolist()


def dist2AtIndices(x, Indices):
    ndata, dimx = len(x), len(x[0])
    ncentres, dimc = len(x), len(x[0])
    if dimx != dimc:
        print('Data dimension does not match dimension of centres')
        sys.exit(0)
    now_x = np.mat(x)
    x1 = (np.ones((ncentres, 1)) * np.square(now_x).T.conjugate().sum(axis=0)).T.conjugate()
    x2 = np.ones((ndata, 1)) * np.square(now_x).T.conjugate().sum(axis=0)
    x3 = np.multiply(2, (now_x * now_x.T.conjugate()))
    n2, sortn2, indsortn2 = [], [], []
    n2 = (x1 + x2 - x3)
    n2[np.less(n2, 0)] = 0
    n2AtIndices = np.zeros((len(Indices), len(Indices[0])))
    for i in range(len(Indices)):
        n2AtIndices[i, :] = n2[i, Indices[i]]
    return n2, n2AtIndices


if __name__ == "__main__":
    # print(dist2N([[3, 2, 1], [4, 5, 6], [9, 8, 7], [10, 11, 12]]))
    data, _, ids = dist2N([[3, 2, 1], [4, 5, 6], [9, 8, 7], [10, 11, 12]])
    print(dist2AtIndices(data, ids))
