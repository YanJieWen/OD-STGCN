# @Time    : 2021/12/21 11:17
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : cal_graph
# @Project Name :new_code

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
#according to Gaussian kernel function to convert adjancency
def weight_matrix(file_path, sigma2=0.1, epsilon=0.5):
    w =pd.read_excel(file_path,header=None).values
    n = w.shape[0]
    W = w / 10000.
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
def build_laplacttion(w):#Normalized graph Laplacian function,2*L/lambda-I
    n, d = np.shape(w)[0], np.sum(w, axis=1)
    L=-w
    # print('the matrix is :',L)
    L[np.diag_indices_from(L)] = d
    for i in range(n):#the equation is I-sqrt(D)*W*sqrt(D)
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    return np.mat(L-np.identity(n))
def cheb_ploy(L, Ks, n):#the method to neibor chebnet
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))
    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    else:
        return np.asarray(L0)
def first_approx(W,n):#GCN
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    return np.mat(np.identity(n) + sinvD * A * sinvD)




