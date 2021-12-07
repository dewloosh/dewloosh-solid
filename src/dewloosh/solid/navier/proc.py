# -*- coding: utf-8 -*-
from dewloosh.math.linalg.linalg import inv3x3
from dewloosh.core.squeeze import squeeze
import numpy as np
from numpy.linalg import solve as npsolve
from numpy import ndarray
from numba import njit, prange


@squeeze(True)
def linsolve(LHS: ndarray, RHS: ndarray, *args, **kwarg):
    if len(LHS.shape) > 2:
        return linsolve_M(LHS, RHS)
    else:
        return linsolve_K(LHS, RHS)


@njit(nogil=True, parallel=True, cache=True)
def linsolve_K(A: ndarray, B: ndarray):
    nLHS, nMN = A.shape
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, nMN))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[i, j, k] = B[i, k] / A[j, k]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve_M(A: ndarray, B: ndarray):
    nLHS, nMN = A.shape[:2]
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, nMN, 3))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[i, j, k] = inv3x3(A[j, k]) @ B[i, k]
    return res