# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from dewloosh.math.linalg.sparse.csr import csr_matrix
from dewloosh.solid.fem.utils import irows_icols_bulk_filtered
from numba import types as nbtypes
from numba.typed import Dict as nbDict


__cache = True
nbint64 = nbtypes.int64
nbint64A = nbint64[:]
nbfloat64A = nbtypes.float64[:]


def cells_around(centers: np.ndarray, r_min: float, *args,
                 as_csr=False, **kwargs):
    conn, widths = _cells_around_(centers, r_min)
    if as_csr:
        data, inds, indptr, shp = dict_to_spdata(conn, widths)
        return csr_matrix(data=data, indices=inds, indptr=indptr, shape=shp)
    return conn


@njit(nogil=True, cache=__cache)
def _cells_around_(centers: np.ndarray, r_min: float):
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbint64A,
    )
    nE = len(centers)
    normsbuf = np.zeros(nE, dtype=centers.dtype)
    widths = np.zeros(nE, dtype=np.int64)
    for iE in range(nE):
        normsbuf[:] = norms(centers - centers[iE])
        res[iE] = np.where(normsbuf <= r_min)[0]
        widths[iE] = len(res[iE])
    return res, widths


def filter_stiffness(K_bulk: np.ndarray, edofs: np.ndarray,
                     vals: np.ndarray, *args, tol=1e-12, **kwargs):
    inds = np.where(vals > tol)[0]
    rows, cols = irows_icols_bulk_filtered(edofs, inds)
    return K_bulk[inds, :, :].flatten(), (rows.flatten(), cols.flatten())


@njit(nogil=True, cache=__cache)
def dict_to_spdata(d: dict, widths: np.ndarray):
    N = int(np.sum(widths))
    nE = len(widths)
    data = np.zeros(N, dtype=np.int64)
    inds = np.zeros_like(data)
    indptr = np.zeros(nE+1, dtype=np.int64)
    _c = 0
    wmax = 0
    for i in range(len(d)):
        w = widths[i]
        if w > wmax:
            wmax = w
        c_ = _c + w
        data[_c: c_] = d[i]
        inds[_c: c_] = np.arange(w)
        indptr[i+1] = c_
        _c = c_
    return data, inds, indptr, (nE, wmax)


@njit(nogil=True, cache=__cache)
def get_filter_factors(centers: np.ndarray, neighbours: nbDict,
                       r_min: float):
    nE = len(centers)
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbfloat64A,
    )
    for iE in range(nE):
        res[iE] = r_min - norms(centers[neighbours[iE]] - centers[iE])
    return res


@njit(nogil=True, cache=__cache)
def get_filter_factors_csr(centers: np.ndarray, neighbours: csr_matrix,
                           r_min: float):
    nE = len(centers)
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbfloat64A,
    )
    for iE in range(nE):
        ni, _ = neighbours.row(iE)
        res[iE] = r_min - norms(centers[ni] - centers[iE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def norms(a: np.ndarray):
    nI = len(a)
    res = np.zeros(nI, dtype=a.dtype)
    for iI in prange(len(a)):
        res[iI] = np.dot(a[iI], a[iI])
    return np.sqrt(res)


@njit(nogil=True, parallel=True, cache=__cache)
def norm(a: np.ndarray):
    return np.sqrt(np.dot(a, a))


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_bulk(K: np.ndarray, weights: np.ndarray):
    nE = len(weights)
    res = np.zeros_like(K, dtype=K.dtype)
    for i in prange(nE):
        res[i, :] = weights[i] * K[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_1d(data: np.ndarray, weights: np.ndarray):
    nE = len(weights)
    dsize = int(len(data)/nE)
    res = np.zeros_like(data, dtype=data.dtype)
    for i in prange(nE):
        res[i*dsize: (i+1)*dsize] = data[i*dsize: (i+1)*dsize] * weights[i]
    return res


def index_of_closest_point(coords: np.ndarray, target=np.ndarray):
    return np.argmin(np.linalg.norm(coords - target, axis=1))


@njit(nogil=True, parallel=True, cache=__cache)
def compliances_bulk(K: np.ndarray, U: np.ndarray, gnum: np.ndarray):
    nE, nNE = gnum.shape
    res = np.zeros(nE, dtype=K.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                res[iE] += K[iE, i, j] * U[gnum[iE, i]] * U[gnum[iE, j]]
    return res
