# -*- coding: utf-8 -*-
from pyoneer.mech.fem.model.solid import Solid
from pyoneer.mech.material.utils import HMH_3d
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True

_NSTRE_ = 6
_NDOFN_ = 3


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(dshp: ndarray, jac: ndarray):
    nE = jac.shape[0]
    nP, nN = dshp.shape[:2]
    nTOTV = nN * _NDOFN_
    B = np.zeros((nE, nP, _NSTRE_, nTOTV), dtype=dshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            gdshp = dshp[iP] @ np.linalg.inv(jac[iE, iP])
            for i in prange(nN):
                B[iE, iP, 0, 0 + i*_NDOFN_] = gdshp[i, 0]
                B[iE, iP, 1, 1 + i*_NDOFN_] = gdshp[i, 1]
                B[iE, iP, 2, 2 + i*_NDOFN_] = gdshp[i, 2]
                B[iE, iP, 3, 1 + i*_NDOFN_] = gdshp[i, 2]
                B[iE, iP, 3, 2 + i*_NDOFN_] = gdshp[i, 1]
                B[iE, iP, 4, 0 + i*_NDOFN_] = gdshp[i, 2]
                B[iE, iP, 4, 2 + i*_NDOFN_] = gdshp[i, 0]
                B[iE, iP, 5, 0 + i*_NDOFN_] = gdshp[i, 1]
                B[iE, iP, 5, 1 + i*_NDOFN_] = gdshp[i, 0]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def HMH(estrs: np.ndarray):
    nE, nP = estrs.shape[:2]
    res = np.zeros((nE, nP), dtype=estrs.dtype)
    for iE in prange(nE):
        for jNE in prange(nP):
            res[iE, jNE] = HMH_3d(estrs[iE, jNE])
    return res


class Solid3d(Solid):

    NDOFN = 3
    NSTRE = 6

    @classmethod
    def strain_displacement_matrix(cls, *args, dshp=None, jac=None, **kwargs):
        return strain_displacement_matrix(dshp, jac)

    @classmethod
    def HMH(cls, data, *args, **kwargs):
        return HMH(data)
