# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
from collections import Iterable
from typing import Union

from dewloosh.math.array import atleast1d
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.geom.cells import L2 as Line

from .utils import to_range
from ..elem import FiniteElement
from ..model.beam import BernoulliBeam


__all__ = ['Bernoulli']


__cache = True

ArrayOrFloat = Union[ndarray, float]


@njit(nogil=True, cache=__cache)
def shape_function_values(x, x1, x2):
    """
    Evaluates the shape functions at a point x
    in the range [x1, x2]. 
    """
    return np.array([
        # for u_s and theta_x
        (x - x2)/(x1 - x2),
        (-x + x1)/(x1 - x2),
        # for v_s and theta_z
        (x - x2)**2*(-2*x + 3*x1 - x2)/(x1 - x2)**3,
        (x - x1)*(x - x2)**2/(x1 - x2)**2,
        (x - x1)**2*(2*x + x1 - 3*x2)/(x1 - x2)**3,
        (x - x1)**2*(x - x2)/(x1 - x2)**2,
        # for w_s and theta_y
        (x - x2)**2*(-2*x + 3*x1 - x2)/(x1 - x2)**3,
        (-x + x1)*(x - x2)**2/(x1 - x2)**2,
        (x - x1)**2*(2*x + x1 - 3*x2)/(x1 - x2)**3,
        (-x + x2)*(x - x1)**2/(x1 - x2)**2
        ])
    

@njit(nogil=True, cache=__cache)
def shp_loc(r):
    """
    Evaluates the shape functions at a point x
    in the range [-1, 1]. 
    """
    return shape_function_values(r, -1, 1)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_loc_bulk(pcoords: ndarray):
    """
    Evaluates the shape functions at several points
    in the range [-1, 1].
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 10), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shp_loc(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_1(x, x1, x2):
    """
    Evaluates the first derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    return np.array([
        1/(x1 - x2),
        -1/(x1 - x2),
        -6*(x - x1)*(x - x2)/(x1 - x2)**3,
        -(x - x2)*(-3*x + 2*x1 + x2)/(x1 - x2)**2,
        6*(x - x1)*(x - x2)/(x1 - x2)**3,
        -(x - x1)*(-3*x + x1 + 2*x2)/(x1 - x2)**2,
        -6*(x - x1)*(x - x2)/(x1 - x2)**3,
        (x - x2)*(-3*x + 2*x1 + x2)/(x1 - x2)**2,
        6*(x - x1)*(x - x2)/(x1 - x2)**3,
        (x - x1)*(-3*x + x1 + 2*x2)/(x1 - x2)**2
        ])
    

@njit(nogil=True, cache=__cache)
def shape_function_derivatives_2(x, x1, x2):
    """
    Evaluates the second derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    return np.array([
        0,
        0,
        6*(-2*x + x1 + x2)/(x1 - x2)**3,
        2*(3*x - x1 - 2*x2)/(x1 - x2)**2,
        6*(2*x - x1 - x2)/(x1 - x2)**3,
        2*(3*x - 2*x1 - x2)/(x1 - x2)**2,
        6*(-2*x + x1 + x2)/(x1 - x2)**3,
        2*(-3*x + x1 + 2*x2)/(x1 - x2)**2,
        6*(2*x - x1 - x2)/(x1 - x2)**3,
        2*(-3*x + 2*x1 + x2)/(x1 - x2)**2
        ])
    

@njit(nogil=True, cache=__cache)
def shape_function_derivatives_3(x, x1, x2):
    """
    Evaluates the third derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    return np.array([
        0,
        0,
        -12/(x1 - x2)**3,
        6/(x1 - x2)**2,
        12/(x1 - x2)**3,
        6/(x1 - x2)**2,
        -12/(x1 - x2)**3,
        -6/(x1 - x2)**2,
        12/(x1 - x2)**3,
        -6/(x1 - x2)**2
        ])
    

@njit(nogil=True, cache=__cache)
def shape_function_derivatives(x, x1, x2):
    """
    Evaluates the derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    res = np.zeros((10, 3))
    res[:, 0] = shape_function_derivatives_1(x, x1, x2)
    res[:, 1] = shape_function_derivatives_2(x, x1, x2)
    res[:, 2] = shape_function_derivatives_3(x, x1, x2)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_loc_bulk(pcoords: ndarray):
    """
    Evaluates the derivatives of the shape 
    functions at several points in the range [-1, 1]. 
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 10, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_derivatives(pcoords[iP], -1, 1)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def gdshp_bulk_v1(pcoords: ndarray, lengths: ndarray):
    """pcoords is expected in the range [0, 1]"""
    nP = pcoords.shape[0]
    nE = lengths.shape[0]
    res = np.zeros((nE, nP, 10, 3), dtype=pcoords.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_derivatives(pcoords[iP]*lengths[iE], 
                                                     0, lengths[iE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def gdshp_bulk_v2(dshp: ndarray, jac: ndarray):
    nP = dshp.shape[0]
    nE = jac.shape[0]
    res = np.zeros((nE, nP, 10, 3), dtype=dshp.dtype)
    for iE in prange(nE):
        for jP in prange(nP):
            res[iE, jP, :, 0] = dshp[jP, :, 0] / jac[iE, jP, 0, 0]
            res[iE, jP, :, 1] = dshp[jP, :, 1] / jac[iE, jP, 0, 0]**2
            res[iE, jP, :, 2] = dshp[jP, :, 2] / jac[iE, jP, 0, 0]**3
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix(shp: ndarray, gdshp: ndarray):
    nE, nP = gdshp.shape[:2]
    N = np.zeros((nE, nP, 6, 12), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            # u_s
            N[iE, iP, 0, 0] = shp[iP, 0]
            N[iE, iP, 0, 6] = shp[iP, 1]
            # v_s
            N[iE, iP, 1, 1] = shp[iP, 2]
            N[iE, iP, 1, 5] = shp[iP, 3]
            N[iE, iP, 1, 7] = shp[iP, 4]
            N[iE, iP, 1, 11] = shp[iP, 5]
            # w_s
            N[iE, iP, 2, 2] = shp[iP, 6]
            N[iE, iP, 2, 4] = shp[iP, 7]
            N[iE, iP, 2, 8] = shp[iP, 8]
            N[iE, iP, 2, 10] = shp[iP, 9]
            # theta_x
            N[iE, iP, 3, 3] = shp[iP, 0]
            N[iE, iP, 3, 9] = shp[iP, 1]
            # theta_y
            N[iE, iP, 4, 2] = - gdshp[iE, iP, 6, 0]
            N[iE, iP, 4, 4] = - gdshp[iE, iP, 7, 0]
            N[iE, iP, 4, 8] = - gdshp[iE, iP, 8, 0]
            N[iE, iP, 4, 10] = - gdshp[iE, iP, 9, 0]
            # theta_z
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
            # 
            B[iE, iP, 0, 0] = gdshp[iE, iP, 0, 0]
            B[iE, iP, 0, 6] = gdshp[iE, iP, 1, 0]
            #
            B[iE, iP, 1, 3] = gdshp[iE, iP, 0, 0]
            B[iE, iP, 1, 9] = gdshp[iE, iP, 1, 0]
            #
            B[iE, iP, 2, 2] = -gdshp[iE, iP, 6, 1]
            B[iE, iP, 2, 4] = -gdshp[iE, iP, 7, 1]
            B[iE, iP, 2, 8] = -gdshp[iE, iP, 8, 1]
            B[iE, iP, 2, 10] = -gdshp[iE, iP, 9, 1]
            #
            B[iE, iP, 3, 1] = gdshp[iE, iP, 2, 1]
            B[iE, iP, 3, 5] = gdshp[iE, iP, 3, 1]
            B[iE, iP, 3, 7] = gdshp[iE, iP, 4, 1]
            B[iE, iP, 3, 11] = gdshp[iE, iP, 5, 1]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def body_load_vector(values: ndarray, shp: ndarray, gdshp: ndarray,
                     djac: ndarray, w: ndarray):
    nE, nG = djac.shape
    res = np.zeros((nE, 12), dtype=values.dtype)
    for iG in range(nG):
        for iE in prange(nE):
            vals = values[iE, 0] * shp[iG, 0] + \
                values[iE, 1] * shp[iG, 1]
            # sum Fx
            res[iE, 0] += shp[iG, 0] * vals[0] * djac[iE, iG] * w[iG]
            res[iE, 6] += shp[iG, 1] * vals[0] * djac[iE, iG] * w[iG]
            # sum Fy
            res[iE, 1] += shp[iG, 2] * vals[1] * djac[iE, iG] * w[iG]
            res[iE, 5] += shp[iG, 3] * vals[1] * djac[iE, iG] * w[iG]
            res[iE, 7] += shp[iG, 4] * vals[1] * djac[iE, iG] * w[iG]
            res[iE, 11] += shp[iG, 5] * vals[1] * djac[iE, iG] * w[iG]            
            res[iE, 1] -= gdshp[iE, iG, 2, 0] * vals[4] * djac[iE, iG] * w[iG]
            res[iE, 5] -= gdshp[iE, iG, 3, 0] * vals[4] * djac[iE, iG] * w[iG]
            res[iE, 7] -= gdshp[iE, iG, 4, 0] * vals[4] * djac[iE, iG] * w[iG]
            res[iE, 11] -= gdshp[iE, iG, 5, 0] * vals[4] * djac[iE, iG] * w[iG]
            # sum Fz
            res[iE, 2] += shp[iG, 6] * vals[2] * djac[iE, iG] * w[iG]
            res[iE, 4] += shp[iG, 7] * vals[2] * djac[iE, iG] * w[iG]
            res[iE, 8] += shp[iG, 8] * vals[2] * djac[iE, iG] * w[iG]
            res[iE, 10] += shp[iG, 9] * vals[2] * djac[iE, iG] * w[iG]
            res[iE, 2] -= gdshp[iE, iG, 6, 0] * vals[5] * djac[iE, iG] * w[iG]
            res[iE, 4] -= gdshp[iE, iG, 7, 0] * vals[5] * djac[iE, iG] * w[iG]
            res[iE, 8] -= gdshp[iE, iG, 8, 0] * vals[5] * djac[iE, iG] * w[iG]
            res[iE, 10] -= gdshp[iE, iG, 9, 0] * vals[5] * djac[iE, iG] * w[iG]
            # sum Mx
            res[iE, 3] += shp[iG, 0] * vals[3] * djac[iE, iG] * w[iG]
            res[iE, 9] += shp[iG, 1] * vals[3] * djac[iE, iG] * w[iG]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calc_element_forces_bulk(dofsol: ndarray, B: ndarray, 
                             D: ndarray, shp: ndarray, 
                             gdshp: ndarray, body_forces: ndarray):
    """
    Calculates internal forces from dof solution.
    """
    nE = dofsol.shape[0]
    nP = B.shape[1]
    res = np.zeros((6, nE, nP), dtype=dofsol.dtype)
    for i in prange(nE):
        for j in prange(nP):
            pyy, pzz = body_forces[i, 0, 4:] * shp[j, 0] + \
                body_forces[i, 1, 4:] * shp[j, 1]
            N, T, My, Mz = D[i] @ B[i, j] @ dofsol[i]
            res[0, i, j] = N
            # Vy
            res[1, i, j] = -D[i, 3, 3] * (\
                gdshp[i, j, 2, 2] * dofsol[i, 1] + \
                gdshp[i, j, 3, 2] * dofsol[i, 5] + \
                gdshp[i, j, 4, 2] * dofsol[i, 7] + \
                gdshp[i, j, 5, 2] * dofsol[i, 11]) - pzz
            # Vz
            res[2, i, j] = -D[i, 2, 2] * (\
                gdshp[i, j, 6, 2] * dofsol[i, 2] + \
                gdshp[i, j, 7, 2] * dofsol[i, 4] + \
                gdshp[i, j, 8, 2] * dofsol[i, 8] + \
                gdshp[i, j, 9, 2] * dofsol[i, 10]) + pyy
            res[3, i, j] = T
            res[4, i, j] = My
            res[5, i, j] = Mz
    return res


njit(nogil=True, parallel=True, cache=__cache)
def interp_element_forces_bulk(edata: ndarray, pcoords: ndarray):
    """
    Approximates internal forces in several elements at several points, 
    from previously calculated internal force data.
    
    The points are exprected in the range [0, 1]. 
    """
    nE = edata.shape[0]
    nP = pcoords.shape[0]
    res = np.zeros((nE, nP, 6))
    for iP in prange(nP):
        shp_i = 1 - pcoords[iP]
        shp_j = pcoords[iP]
        for iE in prange(nE):
            res[iE, iP, :] = edata[iE, 0] * shp_i + edata[iE, 1] * shp_j
    return res


class Bernoulli2(Line, BernoulliBeam, FiniteElement):
    
    qrule = 'full'
    quadrature = {
        'full' : Gauss(2)
        }
    
    @classmethod
    def shape_function_values(cls, *args, pcoords: ArrayOrFloat, 
                              rng: Iterable=None, **kwargs) -> ndarray:
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        pcoords = to_range(pcoords, source=rng, target=[-1, 1])
        if isinstance(pcoords, ndarray):
            return shp_loc_bulk(pcoords)
        else:
            return shp_loc(pcoords)
        
    def shape_function_derivatives(self, *args, pcoords,
                                   rng: Iterable=None, lengths: ndarray=None, 
                                   jac: ndarray=None, **kwargs):
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = to_range(pcoords, source=rng, target=[-1, 1])
        if lengths is not None:
            return gdshp_bulk_v1(pcoords, lengths)
        elif jac is not None:
            return gdshp_bulk_v2(dshp_loc_bulk(pcoords), jac)   
        else:
            if isinstance(pcoords, ndarray):
                return dshp_loc_bulk(pcoords)
            else:
                return shape_function_derivatives(pcoords, -1, 1)
        
    @classmethod
    def shape_function_matrix(cls, *args, pcoords=None, rng=None, **kwargs):
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        shp = cls.shape_function_values(pcoords=pcoords, rng=rng)
        gdshp = cls.shape_function_derivatives(*args, pcoords=pcoords, 
                                               rng=rng, **kwargs)
        return shape_function_matrix(shp, gdshp)
    
    @classmethod
    def strain_displacement_matrix(cls, *args, pcoords: ArrayOrFloat, 
                                   jac=None, rng=None, **kwargs):
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        gdshp = cls.shape_function_derivatives(pcoords=pcoords, rng=rng, jac=jac)
        return strain_displacement_matrix(gdshp)
    
    
    @classmethod
    def _body_load_vector(cls, values: ndarray, shp: ndarray, gdshp: ndarray,
                          djac: ndarray, w: ndarray) -> ndarray:
        return body_load_vector(values, shp, gdshp, djac, w)
    
    @classmethod
    def _calc_element_forces_bulk(cls, dofsol: ndarray, B: ndarray, D: ndarray, 
                                  shp: ndarray, gdshp: ndarray, 
                                  body_forces: ndarray) -> ndarray:
        return calc_element_forces_bulk(dofsol, B, D, shp, gdshp, body_forces)
    
    @classmethod
    def _interp_element_forces_bulk(cls, edata: ndarray, pcoords: ndarray,
                                    rng: Iterable=None) -> ndarray:
        pcoords = to_range(pcoords, source=rng, target=[0, 1])
        return interp_element_forces_bulk(edata, pcoords)
    
    
        
       
    
