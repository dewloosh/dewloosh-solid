# -*- coding: utf-8 -*-
from dewloosh.geom.line import Line
from dewloosh.geom.utils import lengths_of_lines2
from dewloosh.solid.fem.elem import FiniteElement
from dewloosh.math.numint import GaussPoints as Gauss
from dewloosh.solid.fem.model.beam import BernoulliBeam
import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, cache=__cache)
def shp_bernoulli(r):
    return np.array([
        (1 - r) / 2,
        (1 + r) / 2,
        -(-2 * r - 4) * (r - 1)**2 / 8,
        (r - 1)**2 * (r + 1) / 4,
        -(2 * r - 4) * (r + 1)**2 / 8,
        (r - 1) * (r + 1)**2 / 4
        ])
    
    
@njit(nogil=True, parallel=True, cache=__cache)
def shp_bernoulli_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 6), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shp_bernoulli(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp_bernoulli(r):
    return np.array([
        [-1 / 2, 0],
        [1 / 2, 0], 
        [3 * (r - 1) * (r + 1) / 4, 3 * r / 2],
        [(-3 * r - 1) * (r - 1) / 4, 3 * r / 2 - 1 / 2],
        [-3 * (r - 1) * (r + 1) / 4, -3 * r / 2],
        [-(-3 * r + 1) * (r + 1) / 4, 3 * r / 2 + 1 / 2]
        ])
    

@njit(nogil=True, parallel=True, cache=__cache)
def dshp_bernoulli_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 6, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_bernoulli(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def gdshp_bernoulli_bulk(dshp: ndarray, jac: ndarray):
    nP = dshp.shape[0]
    nE = jac.shape[0]
    res = np.zeros((nE, nP, 6, 2), dtype=dshp.dtype)
    for iE in prange(nE):
        for jP in prange(nP):
            res[iE, jP, :, 0] = dshp[jP, :, 0] / jac[iE, jP, 0, 0]
            res[iE, jP, :, 1] = dshp[jP, :, 1] / jac[iE, jP, 0, 0]**2
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix(shp: ndarray, gdshp: ndarray):
    nE, nP = gdshp.shape[:2]
    N = np.zeros((nE, nP, 6, 12), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            N[iE, iP, 0, 0] = shp[iP, 0]
            N[iE, iP, 0, 6] = shp[iP, 1]
            N[iE, iP, 1, 1] = shp[iP, 2]
            N[iE, iP, 1, 5] = shp[iP, 3]
            N[iE, iP, 1, 7] = shp[iP, 4]
            N[iE, iP, 1, 11] = shp[iP, 5]
            N[iE, iP, 2, 2] = shp[iP, 2]
            N[iE, iP, 2, 4] = -shp[iP, 3]
            N[iE, iP, 2, 8] = shp[iP, 4]
            N[iE, iP, 2, 10] = -shp[iP, 5]
            N[iE, iP, 3, 3] = shp[iP, 0]
            N[iE, iP, 3, 9] = shp[iP, 1]
            N[iE, iP, 4, 2] = -gdshp[iE, iP, 2, 0]
            N[iE, iP, 4, 4] = gdshp[iE, iP, 3, 0]
            N[iE, iP, 4, 8] = -gdshp[iE, iP, 4, 0]
            N[iE, iP, 4, 10] = gdshp[iE, iP, 5, 0]
            N[iE, iP, 5, 1] = gdshp[iE, iP, 2, 0]
            N[iE, iP, 5, 5] = gdshp[iE, iP, 3, 0]
            N[iE, iP, 5, 7] = gdshp[iE, iP, 4, 0]
            N[iE, iP, 5, 11] = gdshp[iE, iP, 5, 0]
    return N


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(gdshp: ndarray):
    nE, nP = gdshp.shape[:2]
    B = np.zeros((nE, nP, 4, 12), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            B[iE, iP, 0, 0] = gdshp[iE, iP, 0, 0]
            B[iE, iP, 0, 6] = gdshp[iE, iP, 1, 0]
            B[iE, iP, 1, 3] = gdshp[iE, iP, 0, 0]
            B[iE, iP, 1, 9] = gdshp[iE, iP, 1, 0]
            B[iE, iP, 2, 2] = -gdshp[iE, iP, 2, 1]
            B[iE, iP, 2, 4] = gdshp[iE, iP, 3, 1]
            B[iE, iP, 2, 8] = -gdshp[iE, iP, 4, 1]
            B[iE, iP, 2, 10] = gdshp[iE, iP, 5, 1]
            B[iE, iP, 3, 1] = gdshp[iE, iP, 2, 1]
            B[iE, iP, 3, 5] = gdshp[iE, iP, 3, 1]
            B[iE, iP, 3, 7] = gdshp[iE, iP, 4, 1]
            B[iE, iP, 3, 11] = gdshp[iE, iP, 5, 1]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_det_bulk(jac: ndarray):
    nE, nG = jac.shape[:2]
    res = np.zeros((nE, nG), dtype=jac.dtype)
    for iE in prange(nE):
        res[iE, :] = jac[iE, :, 0, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk(dshp: ndarray, ecoords: ndarray):
    lengths = lengths_of_lines2(ecoords)
    nE = ecoords.shape[0]
    nG = dshp.shape[0]
    res = np.zeros((nE, nG, 1, 1), dtype=dshp.dtype)
    for iE in prange(nE):
        res[iE, :, 0, 0] = lengths[iE] / 2
    return res


class Bernoulli(Line, BernoulliBeam, FiniteElement):
    
    qrule = 'full'
    quadrature = {
        'full' : Gauss(2)
        }
        
    def shape_function_derivatives(self, pcoords, *args, **kwargs):
        if isinstance(pcoords, ndarray):
            return dshp_bernoulli_bulk(pcoords)
        else:
            return dshp_bernoulli(pcoords)
        
    @classmethod
    def shape_function_matrix(cls, *args, pcoords=None, dshp=None, jac=None, **kwargs):
        shp = shp_bernoulli_bulk(pcoords)
        gdshp = gdshp_bernoulli_bulk(dshp, jac)
        return shape_function_matrix(shp, gdshp)
    
    @classmethod
    def strain_displacement_matrix(cls, *args, dshp=None, jac=None, **kwargs):
        gdshp = gdshp_bernoulli_bulk(dshp, jac)
        return strain_displacement_matrix(gdshp)
    
    def jacobian_matrix(self, *args, dshp=None, ecoords=None, **kwargs):
        return jacobian_matrix_bulk(dshp, ecoords)
    
    def jacobian(self, *args, jac=None, **kwargs):
        return jacobian_det_bulk(jac)
    
        
       
    
